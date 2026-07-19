from fastapi import FastAPI, UploadFile, File, HTTPException
from Core.pipeline import build_pipeline
from Core.inference import inference_llm
from typing import List
import tempfile
import os
from pydantic import BaseModel
from utils.logger import get_logger
from fastapi.middleware.cors import CORSMiddleware

backend_logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(BASE_DIR, "../uploaded_file")
backend_logger.info(f"file path is {pdf_dir}")
os.makedirs(pdf_dir, exist_ok=True)

ALLOWED_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
}


app = FastAPI(
    title="rag system",
    description="Upload documents and query them using RAG"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://genai1-pdf-chatbot.streamlit.app/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryStructure(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def check():
    return {"Status": "Successful"}

@app.post("/upload", tags=["upload"])
async def upload_file(files: List[UploadFile] = File(...)):
    
    """Handles multi-file document ingestion for the RAG vector store.
    This endpoint acts as the entry point for expanding the system's knowledge 
    base. It restricts uploads to text and PDF formats to prevent chunking errors 
    downstream, streams the files safely into disk storage using temporary files, 
    and subsequently updates the embedding vectors via `build_pipeline`.
    """
    
    uploaded = []

    for file in files:
        if file.content_type not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file.content_type}' for '{file.filename}'. Only .txt and .pdf are supported."
            )

        suffix = ALLOWED_EXTENSIONS[file.content_type]
        content = await file.read()

        with tempfile.NamedTemporaryFile(dir=pdf_dir, delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            uploaded.append(tmp.name)

    backend_logger.info(f"{len(uploaded)} files uploaded successfully")

    build_pipeline(uploaded)

    return {
        "status": "Successfully stored files",
        "count": len(uploaded),
        "files": uploaded
    }

@app.post("/question")
def answer_question(query: QueryStructure):
    """Executes context-retrieval and LLM generation for a user query.

    Passes the user's question to the inference engine. Note that `top_k` 
    is exposed in the payload to allow the frontend to tune retrieval density 
    depending on the depth of the source documents.
    """
    answer = inference_llm(
        query=query.question,
        top_k=query.top_k
    )

    return {
        "Status" : "Sucessfull",
        "LLM Output" : answer
    }

