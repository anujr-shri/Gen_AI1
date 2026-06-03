from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from Core.vectordb import sematic_search
from Core.llm import get_query_embedding
from utils.logger import get_logger
from dotenv import load_dotenv

logger = get_logger(__name__)
history = []

repo_id = "google/gemma-4-31B-it"

with open("prompt_query.txt", "r") as file:
    query_prompt = file.read()

with open("prompt_text.txt", "r") as file:
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
    
def create_model(max_token: int = 256, temperature=0.1):
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        provider="auto",
        max_new_tokens=max_token
    )  # type: ignore

    model = ChatHuggingFace(llm=endpoint, temperature=temperature)
    return model

model = create_model()

def rewrite_query(query: str, history, query_resolver):
    prompt = query_template.invoke({"history" : history, "question" : query})
    result = query_resolver.invoke(prompt)
    return result.content

def inference_llm(query: str, top_k: int = 3):
    
    context_query = rewrite_query(query, history, model)
    logger.info(f"Debugg The User Query, new query is {context_query}")

    query_embedding = get_query_embedding(query=context_query) # type: ignore
    sematic_search_result = sematic_search(query_embedding=query_embedding, top_k=top_k)
    result = "\n\n".join(sematic_search_result[0])
    logger.info(f"Extraxt The relevant knowledge from pdf")
        
    llm_input = template.invoke(
            {
                "pdf_knowledge" : result,
                "user_input" : context_query
            }
    )

    history.append(HumanMessage(content=query)) # type: ignore
    response = model.invoke(llm_input)
    history.append(AIMessage(content=response.content))  # type: ignore
    return response.content


        
    
