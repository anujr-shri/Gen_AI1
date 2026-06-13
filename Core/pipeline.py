from Core.llm import get_document_embedding
from Core.vectordb import store_embedding_vectordb
from Core.pdf_processing import split_document
from Core.inference import inference_llm
from utils.logger import get_logger
from langsmith import traceable

logger_inst = get_logger(__name__)

@traceable(name="build_pipeline", run_type="chain")
def build_pipeline(path: list[str]):
    """Executes the complete document ingestion and vector storage workflow.

    This function ties together the data preparation steps: it extracts and 
    chunks text from the given file paths, generates dense vector embeddings 
    for those chunks, and commits both the raw text and embeddings to the 
    vector database for future similarity search retrieval.
    """
    chunks = split_document(path)
    if not chunks:
        raise ValueError(f"No chunks extracted from pdf")

    embeddings = get_document_embedding(chunks)
    if not embeddings:
        raise ValueError("Embedding generation failed")

    store_embedding_vectordb(chunks, embeddings)
    logger_inst.info("Pipeline complete — %d chunks stored", len(chunks))
