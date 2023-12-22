import os
import openai
from dotenv import load_dotenv, find_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

persist_directory = "data_base/vector_db/chroma/"
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

question = "Can you briefly introduce about the transformer?"
sim_docs = vectordb.similarity_search(question, k=3)
for i, sim_doc in enumerate(sim_docs):
    print(f"the {i} th content: \n{sim_doc.page_content[:200]}", end="\n-----------------\n")
