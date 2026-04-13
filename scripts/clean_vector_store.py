import os
import numpy as np

DOCS_PATH = 'app/data/documents.npy'
EMBS_PATH = 'app/data/embeddings.npy'
INDEX_PATH = 'app/data/vector.index'
REMOVE_SUBSTR = 'rheumatoid_arthritis.txt'

if os.path.exists(DOCS_PATH) and os.path.exists(EMBS_PATH):
    docs = np.load(DOCS_PATH, allow_pickle=True).tolist()
    embs = np.load(EMBS_PATH, allow_pickle=True).tolist()
    kept_docs = []
    kept_embs = []
    removed = 0
    for d, e in zip(docs, embs):
        s = str(d)
        if REMOVE_SUBSTR in s:
            print('Removing document from index:', s)
            removed += 1
            continue
        kept_docs.append(d)
        kept_embs.append(e)

    np.save(DOCS_PATH, np.array(kept_docs, dtype=object))
    if kept_embs:
        np.save(EMBS_PATH, np.array(kept_embs))
    else:
        if os.path.exists(EMBS_PATH):
            os.remove(EMBS_PATH)

    try:
        import faiss
        if kept_embs:
            dim = len(kept_embs[0])
            idx = faiss.IndexFlatL2(dim)
            idx.add(np.array(kept_embs))
            faiss.write_index(idx, INDEX_PATH)
            print('Rebuilt FAISS index with', len(kept_embs), 'documents')
        else:
            if os.path.exists(INDEX_PATH):
                os.remove(INDEX_PATH)
                print('Removed FAISS index; no documents left')
    except Exception as ex:
        print('FAISS rebuild skipped:', ex)

    print('Removed', removed, 'documents')
else:
    print('No vector store files found to clean')
