from vectorstore.chroma_client import get_chroma_collection


def add_to_chroma(chunk, embedding, metadata):
    collection = get_chroma_collection()

    print(">>> ADD TRY", metadata["chunk_id"], "len(text)", len(chunk))

    collection.add(
        ids=[metadata["chunk_id"]],
        documents=[chunk],
        embeddings=[embedding],
        metadatas=[metadata],
    )

    # 추가: 현재 저장된 데이터 개수 출력
    all_docs = collection.get()
    print(">>> COLLECTION SIZE AFTER ADD:", len(all_docs.get("ids", [])))

    print(f"Saved chunk: {metadata['chunk_id']}")
