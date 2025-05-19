from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# model = ChatOpenAI(model="gpt-4o")
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

result = model.invoke("What is the capital of France?")

print("Full result:")
print(result)
print("Content only:")
print(result.content)