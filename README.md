# 📄 AI PDF Bot

An intelligent PDF question-answering chatbot powered by Google Gemma. Upload any PDF and ask questions — the bot reads, understands, and answers based on your document with full chat history support.

🔗 **Live Demo:** [genai1-pdf-chatbot.streamlit.app](https://genai1-pdf-chatbot.streamlit.app)  
⚙️ **Backend API:** [gen-ai1.onrender.com](https://gen-ai1.onrender.com)

---

## 🚀 Features

- 📁 Upload any PDF document
- 💬 Ask natural language questions about the content
- 🧠 Context-aware answers using RAG (Retrieval-Augmented Generation)
- 🗂️ Chat history maintained across the session
- ⚡ Powered by Google Gemma 4-31B via HuggingFace Inference API

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemma 4-31B (HuggingFace) |
| Embeddings & Retrieval | LangChain + HuggingFace |
| Vector Store | ChromaDB |
| Backend API | FastAPI |
| Logging | Python `logging` |

---

## 🏗️ Architecture

```
User → Streamlit Frontend
           ↓
      FastAPI Backend (Render)
           ↓
      RAG Pipeline
       ├── PDF Processing (LangChain)
       ├── Embeddings (HuggingFace)
       ├── Vector Search (ChromaDB)
       └── Query Rewriting + Generation (Gemma 4-31B)
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Gen_AI1.git
cd Gen_AI1
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

You can get your token from [HuggingFace Settings](https://huggingface.co/settings/tokens).

---

## ▶️ Running Locally

### Start the FastAPI backend

```bash
uvicorn App.backend:app --host 0.0.0.0 --port 8000 --reload
```

### Start the Streamlit frontend

```bash
streamlit run App/home_page.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧭 Usage

1. Open the app in your browser
2. Upload a PDF using the file uploader
3. Wait for the document to be processed
4. Type your question in the chat input
5. Get answers grounded in your document

---

## 📁 Project Structure

```
Gen_AI1/
├── App/
│   ├── backend.py               # FastAPI entry point
│   └── pages/
│       ├── chat_with_llm.py     # Chat interface page
│       └── home_page.py         # Landing page
├── Core/
│   ├── inference.py             # Model inference + query rewriting
│   ├── llm.py                   # LLM + embedding initialization
│   ├── pdf_processing.py        # PDF parsing & chunking
│   ├── pipeline.py              # RAG pipeline orchestration
│   └── vectordb.py              # ChromaDB vector store
├── utils/
│   └── logger.py                # Logging configuration
├── uploaded_file/               # Temp storage for uploaded PDFs
├── prompt_query.txt             # Query rewriting prompt template
├── prompt_text.txt              # RAG answer prompt template
├── .env                         # Environment variables (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Variables

| Variable | Description |
|---|---|
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace API token |

---

## 📝 Notes

- Large PDFs may take a few seconds to process on first upload
- The app uses semantic search to retrieve relevant chunks before generating an answer
- Chat history is session-scoped and resets on page refresh
- Backend is hosted on Render free tier — first request may take ~30s due to cold start

---

## 🙋 Author

**Anuj** — CSE Student at IIIT Bhopal  
[GitHub](https://github.com/anujr-shri)
