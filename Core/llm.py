"""Vector Embedding Generation Module.

This module initializes the connection to Hugging Face embedding models 
to convert both raw document text chunks and incoming user queries into 
dense vector representations for downstream semantic search.
"""

from utils.logger import get_logger
from langsmith import traceable
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

# Specific model selected for text embedding mapping
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

load_dotenv()

logger_inst = get_logger(__name__)

@traceable(name="get_document_embedding", run_type="embedding")
def get_document_embedding(chunks):
    """Generates dense vector embeddings for a collection of text chunks.

    Iterates through LangChain document structures, extracts their raw string 
    contents, and batches them to the Hugging Face endpoint to generate bulk 
    embeddings during data ingestion.
    """
    model = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL
    )
    
    document_embedding = model.embed_documents([chunk.page_content for chunk in chunks])
    logger_inst.info("Successfully get the Embedding from the document")

    return document_embedding

@traceable(name="get_query_embedding", run_type="embedding")
def get_query_embedding(query: str):
    """Generates a single vector embedding for the incoming user search query.

    Uses the same underlying embedding model to project the user's refined query 
    into the exact same vector space as the documents, enabling similarity lookups.
    """
    model = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL
    )
    
    query_embedding = model.embed_query(query)
    logger_inst.info("Successfully Embed The Query")

    return query_embedding
