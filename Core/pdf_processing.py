from utils.logger import get_logger
from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger_inst = get_logger(__name__)

@traceable(name="document-loading")
def loading_pdf(paths):
    """Loads and extracts raw text data from multiple PDF file paths.
    Iterates through a list of local file paths, initializes a PyPDFLoader
    for each document, and compiles all extracted pages into a single list.
    """
    pdf_data = []
    for path in paths:
        loader = PyPDFLoader(path)
        document = loader.load()
        pdf_data.extend(document)
        logger_inst.info(f"Loaded {len(document)} pages from: {path}")
    logger_inst.info(f"Total pages loaded: {len(pdf_data)}")
    return pdf_data
    
@traceable(name="text-splitting")
def split_document(paths, chunk_size=1000, chunk_overlap=200):
    """Chunks loaded document text into overlapping segments for embedding.
    Uses a RecursiveCharacterTextSplitter to split the raw text into manageable
    token sizes. The chunk overlap ensures that semantic context isn't lost
    at the boundaries where sentences are broken up.
    """
    docs_data = loading_pdf(paths)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splitted_document = splitter.split_documents(docs_data)
    logger_inst.info(f"Split the document into {len(splitted_document)} chunks")
    return splitted_document
