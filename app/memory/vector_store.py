import faiss
import numpy as np
import os

index = None
documents = []

INDEX_PATH = "app/data/vector.index"
DOCS_PATH = "app/data/documents.npy"


def init_vector_store(embeddings, docs):
    global index, documents

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))
    documents = docs

    # ✅ SAVE TO DISK
    faiss.write_index(index, INDEX_PATH)
    np.save(DOCS_PATH, docs)


def load_vector_store():
    global index, documents

    if not os.path.exists(INDEX_PATH):
        raise ValueError("Vector DB not found. Run ingest_data first.")

    index = faiss.read_index(INDEX_PATH)
    documents = np.load(DOCS_PATH, allow_pickle=True).tolist()


def search(query_embedding, k=3):
    global index, documents

    if index is None:
        raise ValueError("Vector DB not loaded")

    D, I = index.search(np.array([query_embedding]), k)

    return [documents[i] for i in I[0]]