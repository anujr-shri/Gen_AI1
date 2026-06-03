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

pdf_dir = "uploaded_file"
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


@app.get("/check")
def check():
    return {"Status": "Successful"}

@app.post("/upload", tags=["upload"])
async def upload_file(files: List[UploadFile] = File(...)):
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

    answer = inference_llm(
        query=query.question,
        top_k=query.top_k
    )

    return {
        "Status" : "Sucessfull",
        "LLM Output" : answer
    }

