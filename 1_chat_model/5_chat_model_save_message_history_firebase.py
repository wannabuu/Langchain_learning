from dotenv import load_dotenv
from google.auth import compute_engine
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

PROJECT_ID = "cloud-message-history"
SESSION_ID = "user_session_new"
COLLECTION_NAME = "chat-history"


print("Initializing Firestore client...")
client = firestore.Client(project=PROJECT_ID)

print("Initializing Firestore message history...")
chat_history = FirestoreChatMessageHistory(
    client=client,
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
)

print("Chat message history initialized.")
print("Current message history:",chat_history.messages)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

print("Starting chat session. Type 'exit' to end the session.")


while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")

