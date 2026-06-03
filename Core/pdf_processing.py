from utils.logger import get_logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger_inst = get_logger(__name__)

def loading_pdf(paths):
    pdf_data = []
    for path in paths:
        loader = PyPDFLoader(path)
        document = loader.load()
        pdf_data.extend(document)

    logger_inst.info(f" Loaded {len(document)} pages from pdf")
    return pdf_data

def split_document(paths, chunk_size=1000, chunk_overlap=200):

    docs_data = loading_pdf(paths)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    splitted_document = splitter.split_documents(docs_data)
    logger_inst.info(f" Split THe Document into {len(splitted_document)} chunks")
    return splitted_document
    
