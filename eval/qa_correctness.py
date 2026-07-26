from Core.vectordb import add_documents_vdb
from Core.inference import inference_llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langsmith import Client
from pandas import read_csv
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()

# --- Configuration ---
knowledge_base_url = "https://scikit-learn.org/stable/user_guide.html"
eval_logger = get_logger(__name__)

# --- Loading The QA pair ---
df = read_csv("eval/sklearn_userguide_qa.csv")[["question", "answer"]]

# --- Load The URL and Store it in pgvectors Database ---
def creating_knowledge_base(url: str, max_depth: int = 2):
    # Step1 : Loading The Documents
    try:
        eval_logger.info("Loading Started")
        url_loader = RecursiveUrlLoader(
            url=url,
            max_depth=1,
            prevent_outside=True,
            timeout=10,
            continue_on_failure=True
        )

        html2text_trans = Html2TextTransformer()

        raw_document = url_loader.load()
        text_document = html2text_trans.transform_documents(raw_document)

        eval_logger.info("Loaded The documents From url: {url}")
        eval_logger.info(f"Total Number of Document Loaded {len(text_document)}")

    except Exception as e:
        eval_logger.error("Error Occur While Loading The Document check The url {url} \n [Error Message]: {e}")
        raise

    # Step 2: Document Splitting:

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splited_document = text_splitter.split_documents(text_document)

    eval_logger.info(f"Splith The Document into Chunks \n Number of Chunks: {len(splited_document)} \n chunk_size=500 \n chunk_overlap=200")

    # Step 3: Store The Documents in Vector DataBase
    try:

        add_documents_vdb(splited_document)
        eval_logger.info(f"Store The Document in vector Database")

    except Exception as e:

        eval_logger.error("Error Occur While Storing The Document in pgvectors check Connection " \
        "\n [Error Message]: {e}")
        raise


# --- Creating Golden DataSet in langsmith ---
def creating_golden_dataset():
    # Step 1 : Create Client Instant
    client = Client()

    # Step 2 : Create The DataBase From DataFrame And Store it in Langsmith
    database_name = "sklearn_userguide_qa"
    if not client.has_dataset(dataset_name=database_name):
        database = client.create_dataset(database_name)
        questions_list = df["question"].tolist()
        answer_list = df["answer"].tolist()
        client.create_examples(
            inputs=[{"question": que} for que in questions_list],
            outputs=[{"answer": ans} for ans in answer_list], 
            dataset_id=database.id
        )

    eval_logger.info("Created The Dataset from DataFrame")












