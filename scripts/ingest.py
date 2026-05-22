from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PDF_PATH= "data/Concepts.pdf"
DB_DIR = "chroma_db"

print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print("Pages:", len(pages))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(pages)

print("chunks:", len(chunks))

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=DB_DIR
)

print("Done")