import logging
import os
import openai
from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(temperature=0.9, model="gpt-4")

prompt_1 = ChatPromptTemplate.from_template(
    "tell me the capital of this {country} in both chinese and english,\
    and the format of the output should like this: 'Chinese result  English result'"
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

prompt_2 = ChatPromptTemplate.from_template(
    "and tell me in chinese about how many people are living in this {country}'s capital."
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)
country = "india"
logger.info(chain.run(country))
