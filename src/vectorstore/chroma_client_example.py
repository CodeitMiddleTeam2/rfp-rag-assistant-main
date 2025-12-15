import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
import os


def get_chroma_collection():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(os.path.join(base_dir, "..", "data", "chroma_db"))

    print(">>> USING DB PATH =", db_path)

    os.makedirs(db_path, exist_ok=True)

    client = PersistentClient(path=db_path)

    collection = client.get_or_create_collection(
        name="notices",
        metadata={"hnsw:space": "cosine"},  # embedding_function 절대 넣지 말기
    )

    return collection
