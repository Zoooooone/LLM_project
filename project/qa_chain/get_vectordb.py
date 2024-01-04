import os

from ..database.create_db import create_db, load_db
from ..embedding.call_embedding import get_embedding


def get_vectordb(
        file_path: str = None,
        persist_path: str = None,
        embedding="openai",
        embedding_key: str = None
):
    if file_path is None:
        file_path = "data_base/knowledge_db/"
    if persist_path is None:
        persist_path = "data_base/vector_db/chroma/"

    db_embedding = get_embedding(embedding=embedding, embedding_key=embedding_key)

    if os.path.exists(persist_path):
        contents = os.listdir(persist_path)
        if not contents:
            create_db(file_path, persist_path, db_embedding)
            vectordb = load_db(persist_path, db_embedding)
        else:
            vectordb = load_db(persist_path, db_embedding)
    else:
        create_db(file_path, persist_path, db_embedding)
        vectordb = load_db(persist_path, db_embedding)
    
    return vectordb


if __name__ == "__main__":
    db = get_vectordb()
    print(db)
