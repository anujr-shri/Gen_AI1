import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_postgres import PGVector
from utils.logger import get_logger
from langsmith import traceable
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

logger_instance = get_logger(__name__)

# ---- Model And Vector DB Configuration ----
collection_name = "pdf-qa-project"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
VDB_URL = os.environ["PGVECTOR_CONNECTION_STRING"]

# Load the reranker (tokenizer + model) once at import time instead of on
# every call inside reranking_wrt_query — this was previously re-downloading
# and re-instantiating the model on every single query, which is slow and
# wasteful. Loaded lazily via a module-level cache the first time it's needed.
_reranker_tokenizer = None
_reranker_model = None


def _get_reranker():
    """Lazily load and cache the cross-encoder reranker (tokenizer + model)."""
    global _reranker_tokenizer, _reranker_model
    if _reranker_tokenizer is None or _reranker_model is None:
        logger_instance.info(f"Loading reranker model: {reranker_model_name}")
        _reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
        _reranker_model.eval()  # inference mode, disables dropout etc.
    return _reranker_tokenizer, _reranker_model


# ---- Return The Embedding Model ----
def get_embedding_model(model_id: str):
    """Initialize and return a HuggingFace endpoint embedding model."""
    logger_instance.info(f"Initializing embedding model: {model_id}")
    model = HuggingFaceEndpointEmbeddings(
        model=model_id
    )
    return model


# ---- Initializing The PG Vector Store ----
vector_store = PGVector(
    embeddings=get_embedding_model(EMBEDDING_MODEL),
    collection_name=collection_name,
    connection=VDB_URL,
    use_jsonb=True,
)


# --- Add Chunks to Vector DB ---
def add_documents_vdb(docs):
    """Add a list of LangChain Document objects to the PGVector store.

    Each document's metadata must contain an "id" key, which is used as the
    vector store id (so re-adding the same doc id upserts rather than
    duplicates).
    """
    if not docs:
        logger_instance.warning("add_documents_vdb called with an empty docs list — nothing to add")
        return

    try:
        ids = [doc_id for doc_id in range(len(docs))]
    except KeyError as e:
        logger_instance.error(f"Document missing required metadata field 'id': {e}")
        raise

    vector_store.add_documents(docs, ids=ids)
    logger_instance.info(f"Added {len(docs)} document(s) to PG vector store")


# --- Context Reranking ---
def reranking_wrt_query(retrieved_context, query):
    """Re-rank retrieved documents against the query using a cross-encoder.

    Args:
        retrieved_context: list of LangChain Document objects returned by
            the retriever (e.g. from an MMR search).
        query: the original user query string.

    Returns:
        A list of (Document, score) tuples sorted by descending relevance
        score.
    """
    if not retrieved_context:
        logger_instance.warning("reranking_wrt_query called with no candidate documents")
        return []

    tokenizer, classifier = _get_reranker()

    # Cross-encoder expects (query, passage) pairs
    query_context_pair = [(query, doc.page_content) for doc in retrieved_context]

    features = tokenizer(
        query_context_pair,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():  
        scores = classifier(**features).logits.squeeze().cpu().numpy() 

    # Handle the edge case of a single candidate doc, where squeeze() can
    # collapse a 1-element array down to a 0-d scalar
    if scores.ndim == 0:
        scores = scores.reshape(1)

    doc_scored_pair = list(zip(retrieved_context, scores))
    ranked_context = sorted(doc_scored_pair, key=lambda x: x[1], reverse=True)

    logger_instance.info(f"Reranked {len(ranked_context)} candidate document(s) for query")
    return ranked_context


# --- Semantic Searching ---
@traceable(name="context_retrieving", run_type="retriever")
def context_searching(query: str, top_k: int):
    """Retrieve and rerank the top_k most relevant chunks for a query.

    Over-fetches (top_k + 7) candidates via MMR search on the vector store,
    then reranks them with a cross-encoder and trims down to top_k so the
    final results are both diverse (MMR) and precisely ranked (cross-encoder).
    """
    
    k = top_k + 7
    logger_instance.info(f"Searching vector store for query='{query}' (fetching k={k} candidates)")

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k}, 
    )

    retrieved_context = retriever.invoke(query)
    logger_instance.info(f"Retrieved {len(retrieved_context)} candidate document(s) from vector store")

    context_with_score = reranking_wrt_query(retrieved_context, query)

    top_results = context_with_score[:top_k]
    logger_instance.info(f"Returning top {len(top_results)} reranked result(s)")
    return top_results