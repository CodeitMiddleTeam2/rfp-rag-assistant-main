import re


def tokenize(text: str):
    return re.findall(r"[A-Za-z0-9]+|[가-힣]+", text.lower())


def lexical_score(query: str, document: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    d = set(tokenize(document))
    return len(q & d) / len(q)
