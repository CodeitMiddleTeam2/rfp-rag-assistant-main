from processing.chunker import chunk_text
from processing.embedder import embed_text
from vectorstore.metadata_builder import load_metadata_template, create_metadata
from vectorstore.chroma_add import add_to_chroma


def build_index(row, page, text):
    # 1) 템플릿 로드
    template = load_metadata_template("./metadata/chroma_metadata.txt")

    # 2) 청킹
    chunks = chunk_text(text)

    # 3) 각 chunk 처리
    for idx, chunk in enumerate(chunks):
        metadata = create_metadata(row, page, idx, template)
        embedding = embed_text(chunk)
        add_to_chroma(chunk, embedding, metadata)
