import os 

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File {file_path} does not exist. Please check the path."
        )
    
    loader = TextLoader(file_path,encoding="utf-8")
    print(loader)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Documents Chunks Information ---")
    print(f"Number of documents chunks: {len(docs)}")
    print(f"Sample document chunk: \n{docs[0].page_content}\n")

    print("\n--- Create Embedding ---")
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    print("\n--- Finished creating embedding ---")

    print("\n--- Creating Vector store ---")
    db = Chroma.from_documents(
        docs,
        embedding,
        persist_directory=persistent_directory,
    )
    print("\n--- Finished Creating Vector store ---")

else:
    print("Vector store already exists.")