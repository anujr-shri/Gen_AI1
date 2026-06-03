import chromadb
from utils.logger import get_logger

logger_instance = get_logger(__name__)

collection_name = "pdf-qa-project"

def get_collection():
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="pdf-qa-project")
    return collection

def store_embedding_vectordb(chunks, embeddings):
    collection = get_collection()

    collection.add(
        metadatas=[{"chunk_index": i} for i in range(len(chunks))],
        documents=[chunk.page_content for chunk in chunks],
        embeddings=embeddings,
        ids=[f"id{i}" for i in range(len(chunks))]
    )

    logger_instance.info("Sucessfully Stored The Embedding of document in vector db")

def sematic_search(query_embedding, top_k):
    collection = get_collection()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    docs_result = [result for result in results['documents']] # type: ignore

    return docs_result



