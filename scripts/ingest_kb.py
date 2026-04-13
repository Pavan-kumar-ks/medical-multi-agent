"""Ingest local KB files (app/data/kb/*.txt) into the vector store.

Usage:
    python scripts/ingest_kb.py

This computes embeddings using `app.memory.embeddings.get_embedding` and
initializes the vector store with documents containing `{'id','source','text'}`.
"""
from app.memory.embeddings import get_embedding
from app.memory.vector_store import init_vector_store
import os
import json

KB_DIR = os.path.join(os.path.dirname(__file__), '..', 'app', 'data', 'kb')
KB_DIR = os.path.abspath(KB_DIR)


def load_kb_files():
    docs = []
    texts = []
    if not os.path.exists(KB_DIR):
        print("No KB directory found at", KB_DIR)
        return docs, []
    for fname in sorted(os.listdir(KB_DIR)):
        path = os.path.join(KB_DIR, fname)
        if not os.path.isfile(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        docs.append({'id': fname, 'source': path, 'text': text})
        texts.append(text)
    return docs, texts


def main():
    docs, texts = load_kb_files()
    if not docs:
        print('No KB files to ingest.')
        return
    print(f'Ingesting {len(docs)} docs...')
    embeddings = [get_embedding(t) for t in texts]
    init_vector_store(embeddings, docs)
    print('Ingestion complete. Vector DB initialized.')


if __name__ == '__main__':
    main()
