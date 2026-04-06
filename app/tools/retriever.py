from app.memory.embeddings import get_embedding
from app.memory.vector_store import search


def retrieve_context(query: str):
    emb = get_embedding(query)
    results = search(emb, k=3)
    return results