from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from Core.vectordb import sematic_search
from Core.llm import get_query_embedding
from utils.logger import get_logger
from dotenv import load_dotenv
from langsmith import traceable
import os

logger = get_logger(__name__)
history = []

repo_id = "google/gemma-4-31B-it"
rewriter_repo_id = "Qwen/Qwen2.5-7B-Instruct"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "../prompt_query.txt"), "r") as file:
    query_prompt = file.read()
with open(os.path.join(BASE_DIR, "../prompt_text.txt"), "r") as file:
    prompt = file.read()

template = ChatPromptTemplate(
    [
        ("system", "You are a helpful AI bot."),
        ("human", prompt),
    ]
)
query_template = ChatPromptTemplate(
    [
        MessagesPlaceholder(variable_name="history"),
        ("human", query_prompt)
    ]
)

load_dotenv()

def create_model(max_token: int = 1024, temperature=0.1):
    """Initializes and returns the primary ChatHuggingFace model for generation tasks."""
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        provider="auto",
        max_new_tokens=max_token
    )  # type: ignore
    model = ChatHuggingFace(llm=endpoint, temperature=temperature)
    return model


def create_rewriter_model(max_token: int = 256, temperature=0.0):
    """Initializes and returns a dedicated ChatHuggingFace model for query rewriting."""
    endpoint = HuggingFaceEndpoint(
        repo_id=rewriter_repo_id,
        task="text-generation",
        provider="auto",
        max_new_tokens=max_token
    )  # type: ignore
    model = ChatHuggingFace(llm=endpoint, temperature=temperature)
    return model

model = create_model()
query_rewriter = create_rewriter_model()

@traceable(name="query_rewrtting", run_type="llm")
def rewrite_query(query: str, history, query_resolver):
    """Refreshes and contextualizes the user query using conversation history and a rewriter model."""
    prompt = query_template.invoke({"history": history, "question": query})
    result = query_resolver.invoke(prompt)
    return result.content

@traceable(name="llm-inference", run_type="llm")
def inference_llm(query: str, top_k: int = 3):
    """Executes the RAG pipeline by rewriting the query, searching the vector DB, and generating an answer."""
    context_query = rewrite_query(query, history, query_rewriter)
    if not context_query or context_query.strip() == "":
        context_query = query
        logger.info("Query rewrite returned empty, using original query")
    logger.info(f"Debugg The User Query, new query is {context_query}")
    query_embedding = get_query_embedding(query=context_query)  # type: ignore
    sematic_search_result = sematic_search(query_embedding=query_embedding, top_k=top_k)
    result = "\n\n".join(sematic_search_result[0])
    logger.info(f"Extraxt The relevant knowledge from pdf")
    llm_input = template.invoke(
        {
            "pdf_knowledge": result,
            "user_input": context_query
        }
    )
    history.append(HumanMessage(content=query))  # type: ignore
    response = model.invoke(llm_input)
    history.append(AIMessage(content=response.content))  # type: ignore
    logger.info(f"Model Response is {response.content}")
    return response.content
