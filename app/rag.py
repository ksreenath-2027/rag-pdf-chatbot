from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding  = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db =  Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

query = "what is kubernetes service"

results = db.similarity_search(
    query,
    k=3
)

for i, r in enumerate(results):
    print(f"\n Result {i+1}")
    print("page:", r.metadata)

    print(r.page_content[:500])



