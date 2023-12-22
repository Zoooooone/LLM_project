import os
import openai
from dotenv import load_dotenv, find_dotenv

from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# load pdf
loaders = [
    PyMuPDFLoader("data_base/knowledge_db/A_Practical_Introduction_to_Python_Programming_Heinold.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings()

persist_directory = "data_base/vector_db/chroma"
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory
)
vectordb.persist()

print(f"the amount of data stored in database: {vectordb._collection.count()}")
