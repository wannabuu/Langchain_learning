import os

from dotenv import load_dotenv

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings

load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_metadata")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

db = Chroma(persist_directory=persistent_directory,embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3}
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = rag_chain.invoke({"input": query, "chat_history": chat_history})

        print(f"AI: {result['answer']}")

        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


if __name__ == "__main__":
    continual_chat()