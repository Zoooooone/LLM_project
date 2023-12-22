import os
import openai
import sys

from dotenv import load_dotenv, find_dotenv

from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

folder_path = "data_base/knowledge_db/"
files = os.listdir(folder_path)
loaders, docs = [], []

for f in files:
    file_type = f.split(".")[-1]
    if file_type == "pdf":
        loaders.append(PyMuPDFLoader(folder_path + f))
    elif file_type == "md":
        loaders.append(UnstructuredMarkdownLoader(folder_path + f))

for loader in loaders:
    docs.extend(loader.load())

# text split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

# embedding
embedding = OpenAIEmbeddings()

# persist directory
persist_directory = "data_base/vector_db/chroma/"

vertordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory
)
vertordb.persist()
