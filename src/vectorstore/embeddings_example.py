# src/vectorstore/embeddings.py
from sentence_transformers import SentenceTransformer

_model = None


def get_bge_m3():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-m3")
        _model.eval()
    return _model
