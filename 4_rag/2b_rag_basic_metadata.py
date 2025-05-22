import os 

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_metadata")

embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

db = Chroma(
    embedding_function=embedding,
    persist_directory=persistent_directory)

query = "How did juliet die?"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {'score_threshold': 0.4,'k': 3}
)

relevant_docs = retriever.invoke(query)

print("\n-- Releveant docs --")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source:{doc.metadata.get('source', "Unknown")}\n")
