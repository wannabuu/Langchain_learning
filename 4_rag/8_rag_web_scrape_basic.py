import os 

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_apple")

urls = ["https://www.apple.com/"]

loader = WebBaseLoader(urls)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print("\n--- Documents chunks imformation ---")
print(f"Number of documents: {len(docs)}")
print(f"Sample Chunk: \n{docs[0].page_content}\n")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

if not os.path.exists(persistent_directory):
    print(f"\n--- Creating vector store {persistent_directory} ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print(f"--- Finished creating vector store {persistent_directory} ---")
else:
    print(
        f"Vector store {persistent_directory} already exists. No need to initialize."
    )
    db = Chroma(
        persist_directory=persistent_directory, embedding_function=embeddings
    )

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

query = "What is the latest iPhone model?"
relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")