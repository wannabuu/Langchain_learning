from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

messages = [
    SystemMessage(content="Solve the following math problems."),
    HumanMessage(content="What is 81 divided by 9?"),
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")

messages = [
    SystemMessage(content="Solve the following math problems."),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 9 times 9?"),
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")