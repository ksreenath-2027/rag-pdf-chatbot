from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key= "groq_api_key"
)

chat_history=[]

print("PDF RAG Chatbot Started")
print("Type exit to quit\n")

while True:

    question=input("Ask: ")

    if question.lower()=="exit":
        break

    history="\n".join(chat_history)

    docs=db.similarity_search_with_relevance_scores(
        question,
        k=5
    )

    filtered_docs=[]

    context=""

    for doc,score in docs:

        if score>0.6:

            filtered_docs.append(doc)

            context+=doc.page_content+"\n"

    prompt=f"""
You are a Kubernetes assistant.

Previous conversation:

{history}

Document Context:

{context}

Question:

{question}

Answer only from document context.
If answer not found, say:
'I could not find this in the PDF'
"""

    response=llm.invoke(prompt)

    answer=response.content

    print("\nAnswer:\n")
    print(answer)

    chat_history.append(
        f"User:{question}"
    )

    chat_history.append(
        f"Assistant:{answer}"
    )

    print("\n"+"="*60)