from utils.logger import get_logger
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

load_dotenv()

logger_inst = get_logger(__name__)

def get_document_embedding(chunks):

    model = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL
    )
    
    document_embedding = model.embed_documents([chunk.page_content for chunk in chunks])
    logger_inst.info("Successfully get the Embedding from the document")

    return document_embedding

def get_query_embedding(query: str):

    model = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL
    )
    
    query_embedding = model.embed_query(query)
    logger_inst.info("Successfully Embed The Query")

    return query_embedding





