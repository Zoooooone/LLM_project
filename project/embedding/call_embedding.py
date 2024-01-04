import openai

from langchain.embeddings.openai import OpenAIEmbeddings
from ..llm.call_llm import parse_llm_api_key


def get_embedding(embedding: str, embedding_key: str = None, env_file: str = None):
    if embedding is None:
        embedding_key = parse_llm_api_key(embedding, env_file)
    if embedding == "openai":
        openai.api_key = embedding_key
        return OpenAIEmbeddings()
    else:
        raise ValueError(f"embedding {embedding} not support")
