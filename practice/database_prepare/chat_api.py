import os
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

# OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Which football team won the World Cup 2014?"}
    ]
)
print(response.choices[0].message.content)
# print(response)
# print(type(response))


# langchain
chat = ChatOpenAI()
template = """
    Translate the text \
    that is delimited by triple backticks \
    into Chinese. \
    text: ```{text}```
"""
# the instantiation of the template
chat_template = ChatPromptTemplate.from_template(template)

text = """
    Chat models take a list of messages as input and return a model-generated message as output. \
    Although the chat format is designed to make multi-turn conversations easy, \
    it's just as useful for single-turn tasks without any conversation.
"""
message = chat_template.format_messages(text=text)
response = chat(message)
print(response.content)
