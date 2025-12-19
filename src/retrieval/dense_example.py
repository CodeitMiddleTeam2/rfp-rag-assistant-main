from vectorstore.chroma_client import get_chroma_collection
from vectorstore.embeddings import get_bge_m3


def dense_search(question: str, k: int, where: dict | None = None):
    model = get_bge_m3()
    collection = get_chroma_collection()

    query_vec = model.encode(question, normalize_embeddings=True).tolist()

    res = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        hits.append(
            {
                "chunk_id": meta.get("chunk_id"),
                "text": doc,
                "meta": meta,
                # cosine distance → similarity
                "dense_score": 1.0 - float(dist),
            }
        )

    return hits
