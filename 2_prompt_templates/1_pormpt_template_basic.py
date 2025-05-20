from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# template = "Tell me a joke about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)

# print("Prompt template initialized.")
# prompt = prompt_template.invoke({"topic": "cats"})
# print(prompt)
#----------------------------------------

# template_multiple = """
# You are a helpful assistant.
# Human: Tell me a {adjective} story about {animal}.
# Assistant:"""

# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective":"funny","animal": "cats"})

# print("Prompt template initialized.")
# print(prompt)

#----------------------------------------

message = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_multiple = ChatPromptTemplate.from_messages(message)
prompt = prompt_multiple.invoke({"topic":"lawyers","joke_count": 3})

print("Prompt template initialized.")
print(prompt)
