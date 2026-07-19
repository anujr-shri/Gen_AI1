"""Vector Embedding Generation Module.

This module initializes the connection to Hugging Face embedding models
to convert both raw document text chunks and incoming user queries into
dense vector representations for downstream semantic search.
"""

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langsmith import traceable

from utils.logger import get_logger

logger_inst = get_logger(__name__)
load_dotenv()

# Specific model selected for text embedding mapping
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

@traceable(name="get_query_embedding", run_type="embedding")
def get_query_embedding(query: str):
    """
    Generate a single vector embedding for the incoming user search query.

    Uses the same underlying embedding model to project the user's refined query
    into the exact same vector space as the documents, enabling similarity lookups.
    """
    model = HuggingFaceEndpointEmbeddings(model=EMBEDDING_MODEL)

    query_embedding = model.embed_query(query)
    logger_inst.info("Successfully embedded the query")

    return query_embedding