import os
import openai
from dotenv import load_dotenv, find_dotenv

from langchain.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader


_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]


def load_pdf(path):
    loader = PyMuPDFLoader(path)
    pages = loader.load()
    print(f"\ntype of the variable: {type(pages)}, total pages: {len(pages)}")
    print(pages[len(pages) // 2].page_content[:200])


def load_markdown(path):
    loader = UnstructuredMarkdownLoader(path)
    pages = loader.load()
    print(f"\ntype of the variable: {type(pages)}, total pages: {len(pages)}")
    print(pages[len(pages) // 2].page_content[:200])


if __name__ == "__main__":
    load_pdf("data_base/knowledge_db/A_Practical_Introduction_to_Python_Programming_Heinold.pdf")
    load_markdown("data_base/knowledge_db/markdown.md")
