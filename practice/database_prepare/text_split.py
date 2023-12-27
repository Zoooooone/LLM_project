import os
import openai
from dotenv import load_dotenv, find_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

CHUNK_SIZE = 500
OVERLAP_SIZE = 50

loader = PyMuPDFLoader("data_base/knowledge_db/A_Practical_Introduction_to_Python_Programming_Heinold.pdf")
pages = loader.load()
page = pages[102]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)
split_docs = text_splitter.split_documents(pages)

print(f"total number of splitted files: {len(split_docs)}")
print(f"total number of splitted strings: {sum([len(doc.page_content) for doc in split_docs])}")
