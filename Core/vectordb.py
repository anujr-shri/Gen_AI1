import chromadb
from utils.logger import get_logger
from langsmith import traceable

logger_instance = get_logger(__name__)

collection_name = "pdf-qa-project"

def get_collection():
    """Initializes or connects to the local ChromaDB collection.

    Using the ephemeral Client ensures an in-memory vector database 
    instance is configured and ready for handling operational read/write actions.
    """
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="pdf-qa-project")
    return collection

@traceable(name="store_embedding_vectordb", run_type="retriver")
def store_embedding_vectordb(chunks, embeddings):
    """Commits text segments, metadata, and generated vector arrays to ChromaDB.

    Maps the structural attributes (unique generated IDs, position metadata, 
    raw document strings, and dense embeddings) into the target collection 
    to enable downstream vector index retrieval.
    """
    collection = get_collection()

    collection.add(
        metadatas=[{"chunk_index": i} for i in range(len(chunks))],
        documents=[chunk.page_content for chunk in chunks],
        embeddings=embeddings,
        ids=[f"id{i}" for i in range(len(chunks))]
    )

    logger_instance.info("Sucessfully Stored The Embedding of document in vector db")

@traceable(name="semantic-search", run_type="retriver")
def sematic_search(query_embedding, top_k):
    """Queries the vector index for data blocks most similar to the user prompt.

    Performs a vector distance comparison using the incoming query embedding array 
    and returns the top matches from the stored text documents.
    """
    collection = get_collection()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    docs_result = [result for result in results['documents']] # type: ignore

    return docs_result
