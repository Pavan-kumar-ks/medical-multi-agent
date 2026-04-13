"""Retriever that returns top passages with provenance.

Primary path: semantic search via vector store. Fallback: simple keyword search
over local KB files in `app/data/kb`.
"""
from typing import List, Dict
from app.memory.embeddings import get_embedding
from app.memory.vector_store import search
import os


def _keyword_search(query: str, k: int = 3) -> List[Dict]:
    kb_dir = os.path.join(os.path.dirname(__file__), "..", "data", "kb")
    kb_dir = os.path.abspath(kb_dir)
    results = []
    if not os.path.exists(kb_dir):
        return results
    for fname in os.listdir(kb_dir):
        path = os.path.join(kb_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            score = sum(1 for tok in query.lower().split() if tok in text.lower())
            if score > 0:
                results.append({"id": fname, "source": path, "text": text[:200], "score": score})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


def retrieve_context(query: str, k: int = 3) -> List[Dict]:
    """Return a list of evidence items: {id, source, text, score}.

    Tries semantic search first; falls back to keyword search if vector DB is unavailable.
    """
    try:
        emb = get_embedding(query)
        hits = search(emb, k=k)
        # Normalize returned documents into evidence items
        out = []
        for i, doc in enumerate(hits):
            if isinstance(doc, dict):
                text = doc.get("text") or str(doc)
                src = doc.get("source") or doc.get("id") or f"doc_{i}"
            else:
                text = str(doc)
                src = f"doc_{i}"
            out.append({"id": src, "source": src, "text": text[:500], "score": 1.0 - (i * 0.1)})
        if out:
            return out
    except Exception:
        # fallthrough to keyword search
        pass

    return _keyword_search(query, k=k)