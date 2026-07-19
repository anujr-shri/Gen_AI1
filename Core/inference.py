"""RAG pipeline module: rewrites queries, retrieves context, and generates answers."""

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langsmith import traceable

from Core.vectordb import context_searching
from utils.logger import get_logger

logger = get_logger(__name__)

# In-memory conversation history (list of HumanMessage / AIMessage)
history = []

# --- Model configuration ---
repo_id = "google/gemma-4-31B-it"
rewriter_repo_id = "Qwen/Qwen2.5-7B-Instruct"

# --- Load prompt templates from disk ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "../prompt_query.txt"), "r") as file:
    query_prompt = file.read()

with open(os.path.join(BASE_DIR, "../prompt_text.txt"), "r") as file:
    prompt = file.read()

# Main generation prompt: system instruction + human prompt (with PDF context)
template = ChatPromptTemplate(
    [
        ("system", "You are a helpful AI bot."),
        ("human", prompt),
    ]
)

# Query-rewriting prompt: prior conversation history + the rewrite instruction
query_template = ChatPromptTemplate(
    [
        MessagesPlaceholder(variable_name="history"),
        ("human", query_prompt),
    ]
)

load_dotenv()


def create_model(max_token: int = 1024, temperature: float = 0.1):
    """Initialize and return the primary ChatHuggingFace model used for answer generation."""
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        provider="auto",
        max_new_tokens=max_token,
    )  # type: ignore
    return ChatHuggingFace(llm=endpoint, temperature=temperature)


def create_rewriter_model(max_token: int = 256, temperature: float = 0.0):
    """Initialize and return a dedicated ChatHuggingFace model used for query rewriting."""
    endpoint = HuggingFaceEndpoint(
        repo_id=rewriter_repo_id,
        task="text-generation",
        provider="auto",
        max_new_tokens=max_token,
    )  # type: ignore
    return ChatHuggingFace(llm=endpoint, temperature=temperature)


# Instantiate models once at import time
model = create_model()
query_rewriter = create_rewriter_model()


@traceable(name="query_rewrtting", run_type="llm")
def rewrite_query(query: str, history, query_resolver):
    """
    Rewrite/contextualize the user's query using conversation history.

    This lets follow-up questions ("what about the second one?") be turned
    into standalone queries suitable for semantic search.
    """

    prompt = query_template.invoke({"history": history, "question": query})
    result = query_resolver.invoke(prompt)
    return result.content


@traceable(name="llm_inference", run_type="llm")
def inference_llm(query: str, top_k: int = 3):
    """
    Run the full RAG pipeline:
      1. Rewrite the query using conversation history.
      2. Embed the rewritten query and retrieve relevant chunks from the vector DB.
      3. Generate an answer grounded in the retrieved context.
      4. Update conversation history with the new turn.
    """

    # Step 1: contextualize the query

    context_query = rewrite_query(query, history, query_rewriter)
    if not context_query or context_query.strip() == "":
        context_query = query
        logger.info(f"Query rewrite returned empty, using original query {query}")
    logger.info(f"Debug: user query rewritten to: {context_query}")

    # Step 2: retrieve relevant context from the vector store
    semantic_search_result = context_searching(query, top_k)
    result = "\n\n".join(semantic_search_result[0])
    logger.info("Extracted relevant knowledge from PDF")

    # Step 3: generate the answer
    llm_input = template.invoke(
        {
            "pdf_knowledge": result,
            "user_input": context_query,
        }
    )

    history.append(HumanMessage(content=query))  # type: ignore
    response = model.invoke(llm_input)
    history.append(AIMessage(content=response.content))  # type: ignore

    logger.info(f"Model response: {response.content}")
    return response.content