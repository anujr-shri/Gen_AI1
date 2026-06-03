from Core.llm import get_document_embedding
from Core.vectordb import store_embedding_vectordb
from Core.pdf_processing import split_document
from Core.inference import inference_llm
from utils.logger import get_logger

logger_inst = get_logger(__name__)

def build_pipeline(path: list[str]):
    chunks = split_document(path)
    if not chunks:
        raise ValueError(f"No chunks extracted from pdf")

    embeddings = get_document_embedding(chunks)
    if not embeddings:
        raise ValueError("Embedding generation failed")

    store_embedding_vectordb(chunks, embeddings)
    logger_inst.info("Pipeline complete — %d chunks stored", len(chunks))


