import os 

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
book_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_metadata")

print(f"Book directory: {book_dir}")
print(f"Persistent directory: {persistent_directory}")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist.")

    if not os.path.exists(book_dir):
        raise FileNotFoundError(
            f"File {book_dir} does not exist. Please check the path."
        )
    
    book_files = [f for f in os.listdir(book_dir) if f.endswith('.txt')]

    documents=[]
    for book_file in book_files:
        file_path = os.path.join(book_dir, book_file)
        loader = TextLoader(file_path,encoding="utf-8")
        book_doc = loader.load()
        for doc in book_doc:
            doc.metadata = {"source": file_path}
            documents.append(doc)
        
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Documents Chunks Information ---")
    print(f"Number of documents chunks: {len(docs)}")

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