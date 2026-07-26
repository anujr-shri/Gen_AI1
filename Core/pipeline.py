from Core.vectordb import add_documents_vdb
from Core.pdf_processing import split_document
from utils.logger import get_logger
from langsmith import traceable

logger = get_logger(__name__)


@traceable(name="build_pipeline", run_type="chain")
def build_pipeline(paths: list[str], testing: bool = False) -> int:
    """Ingest documents into the vector store.

    Splits each document into text chunks, then stores those chunks
    (embeddings generated downstream by `add_documents_vdb`) in the
    vector database for later similarity search.

    Args:
        paths: File paths to the source documents.

    Returns:
        The number of chunks stored.

    Raises:
        ValueError: If `paths` is empty, or no chunks could be extracted.
        RuntimeError: If chunking or vector storage fails unexpectedly.
    """

    if not paths:
        raise ValueError("build_pipeline called with an empty paths list")

    logger.info("Starting pipeline for %d document(s): %s", len(paths), paths)

    # Step 1: split documents into chunks

    try:
        chunks = split_document(paths)
    except FileNotFoundError:
        logger.exception("One or more source files not found: %s", paths)
        raise
    except Exception as exc:
        logger.exception("Failed to split documents: %s", paths)
        raise RuntimeError(f"Document splitting failed for {paths}") from exc

    if not chunks:
        logger.error("No chunks extracted from documents: %s", paths)
        raise ValueError(f"No chunks extracted from documents: {paths}")

    logger.info("Extracted %d chunks from %d document(s)", len(chunks), len(paths))

    # Step 2: store chunks (and embeddings) in the vector database

    try:
        add_documents_vdb(chunks)
    except Exception as exc:
        logger.exception("Failed to store %d chunks in vector DB", len(chunks))
        raise RuntimeError("Vector DB storage failed") from exc
    

    logger.info("Pipeline complete — %d chunks stored", len(chunks))
    return len(chunks)