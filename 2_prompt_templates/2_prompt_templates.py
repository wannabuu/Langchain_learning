from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "cats"})
result = model.invoke(prompt)
print(result.content)

#----------------------------------------

template_multiple = """
You are a helpful assistant.
Human: Tell me a {adjective} story about {animal}.
Assistant:"""

prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective":"funny","animal": "cats"})
result = model.invoke(prompt)
print(result.content)
