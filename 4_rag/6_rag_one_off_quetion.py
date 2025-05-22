import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_metadata")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

query = "How can i learn about langchain?"
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Create a ChatOpenAI model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
