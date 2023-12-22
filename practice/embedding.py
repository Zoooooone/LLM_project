import os
import openai

import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv, find_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings


_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

embedding = OpenAIEmbeddings()

query_1 = "math"
query_2 = "number"
query_3 = "cup"

emb_1 = np.array(embedding.embed_query(query_1)).reshape(1, -1)
emb_2 = np.array(embedding.embed_query(query_2)).reshape(1, -1)
emb_3 = np.array(embedding.embed_query(query_3)).reshape(1, -1)

print(f"{query_1} and {query_2}: {cosine_similarity(emb_1, emb_2)}")
print(f"{query_1} and {query_3}: {cosine_similarity(emb_1, emb_3)}")
print(f"{query_2} and {query_3}: {cosine_similarity(emb_2, emb_3)}")
