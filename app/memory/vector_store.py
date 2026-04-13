"""Vector store helper with FAISS when available and in-memory fallback.

Provides `init_vector_store`, `load_vector_store`, and `search`.
"""
import os
from typing import Any, List

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    import numpy as np
except Exception:
    np = None

# In-memory fallback
_index = None
_documents: List[Any] = []
_embeddings: List[List[float]] = []

INDEX_PATH = "app/data/vector.index"
DOCS_PATH = "app/data/documents.npy"
EMBS_PATH = "app/data/embeddings.npy"


def init_vector_store(embeddings: List[List[float]], docs: List[Any]):
    global _index, _documents, _embeddings

    if HAS_FAISS and np is not None:
        dim = len(embeddings[0])
        idx = faiss.IndexFlatL2(dim)
        idx.add(np.array(embeddings))
        faiss.write_index(idx, INDEX_PATH)
        np.save(DOCS_PATH, docs)
        _index = idx
        _documents = docs
    else:
        _embeddings = embeddings
        _documents = docs
        if np is not None:
            np.save(EMBS_PATH, np.array(embeddings))
            np.save(DOCS_PATH, np.array(docs, dtype=object))


def load_vector_store():
    global _index, _documents, _embeddings

    if HAS_FAISS and np is not None:
        if not os.path.exists(INDEX_PATH):
            raise ValueError("Vector DB not found. Run ingest_data first.")
        _index = faiss.read_index(INDEX_PATH)
        _documents = np.load(DOCS_PATH, allow_pickle=True).tolist()
    else:
        if os.path.exists(EMBS_PATH) and np is not None:
            _embeddings = np.load(EMBS_PATH, allow_pickle=True).tolist()
        if os.path.exists(DOCS_PATH):
            _documents = np.load(DOCS_PATH, allow_pickle=True).tolist()


def _cosine_similarity(a, b):
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def search(query_embedding: List[float], k: int = 3) -> List[Any]:
    if HAS_FAISS and np is not None and _index is not None:
        D, I = _index.search(np.array([query_embedding]), k)
        return [_documents[i] for i in I[0]]

    if not _documents or not _embeddings:
        raise ValueError("Vector DB not loaded; run ingest_data first")

    scores = [(i, _cosine_similarity(query_embedding, emb)) for i, emb in enumerate(_embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = [idx for idx, _ in scores[:k]]
    return [_documents[i] for i in top]