import os
import openai

from dotenv import load_dotenv, find_dotenv

from ..embedding.call_embedding import get_embedding

from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]
DEFAULT_DB_PATH = "data_base/knowledge_db/"
DEFAULT_PERSIST_PATH = "data_base/vector_db/chroma/"


def get_files(paths=DEFAULT_DB_PATH):
    files = []
    for path in paths:
        if os.path.isdir(path):
            for root, dir, filenames in os.walk(path):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
        elif os.path.isfile(path):
            files.append(path)
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


def create_db_info(
        paths=DEFAULT_DB_PATH,
        embeddings="openai",
        persist_directory=DEFAULT_PERSIST_PATH
):
    create_db(
        paths=paths,
        persist_directory=persist_directory,
        embeddings=embeddings
    )
    return "Embedding completed!"


def create_db(
        paths: list[str] = DEFAULT_DB_PATH, 
        persist_directory: str = DEFAULT_PERSIST_PATH, 
        embeddings: str = "openai"
):
    if paths is None:
        return "can't load empty file"
    if type(paths) is not list:
        paths = [paths]
    
    files = get_files(paths)
    docs = load_files(files)
    split_docs = split_text(docs)

    if type(embeddings) is str:
        embeddings = get_embedding(embedding=embeddings)

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
    msg = create_db_info()
    print(msg)
