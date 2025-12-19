import copy
import datetime


def load_metadata_template(filepath):
    template = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            key = line.strip()
            if key:
                template[key] = ""
    return template


def create_metadata(row, page, chunk_index, template):
    meta = template.copy()

    # 기본 row값 적용
    for key in row:
        if key in meta:
            meta[key] = row[key]

    # chunk_id 생성
    notice_id = row.get("notice_id", "noid")
    meta["page"] = page
    meta["chunk_index"] = chunk_index
    meta["chunk_id"] = f"{notice_id}_p{page}_c{chunk_index}"

    return meta
