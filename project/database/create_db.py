import os
import openai

from dotenv import load_dotenv, find_dotenv

from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]
DEFAULT_DB_PATH = "data_base/knowledge_db/"
DEFAULT_PERSIST_PATH = "data_base/vector_db/chroma/"


def get_files(path=DEFAULT_DB_PATH):
    files = [path + f for f in os.listdir(path)]
    return files


def load_files(files):
    loaders, docs = [], []

    for file in files:
        file_type = file.split(".")[-1]
        if file_type == "pdf":
            loaders.append(PyMuPDFLoader(file))
        elif file_type == "md":
            loaders.append(UnstructuredMarkdownLoader(file))
        elif file_type == "txt":
            loaders.append(UnstructuredFileLoader(file))

    for loader in loaders:
        docs.extend(loader.load())

    return docs


def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    return split_docs


def create_db(path=DEFAULT_DB_PATH, persist_directory=DEFAULT_PERSIST_PATH, embeddings=OpenAIEmbeddings()):
    if path is None:
        return "can't load empty file"
    
    files = get_files(path)
    docs = load_files(files)
    split_docs = split_text(docs)

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()

    return vectordb


def load_db(persist_directory=DEFAULT_PERSIST_PATH, embeddings=OpenAIEmbeddings()):
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vectordb


if __name__ == "__main__":
    create_db()
