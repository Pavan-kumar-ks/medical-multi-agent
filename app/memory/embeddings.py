"""Embeddings helper with lazy SentenceTransformer loading and a deterministic
fallback for offline or network-failing environments.

`get_embedding(text)` returns a list[float]. The module will attempt to load
`SentenceTransformer('all-MiniLM-L6-v2')` on first use; if that fails it will
return a stable hash-based vector so the system can run locally without HF access.
"""
from typing import List, Optional
import hashlib
import os
import json
import requests

# Model is loaded lazily to avoid network calls and heavy imports at module import time
_model = None
_model_failed = False
_DIM = 384


def _try_load_model():
    """Attempt to import and instantiate the SentenceTransformer model.

    Sets module-level `_model` on success, or marks `_model_failed` on error.
    """
    global _model, _model_failed
    if _model is not None or _model_failed:
        return

    # Only attempt to load heavy HF model when explicitly enabled to avoid
    # importing large native deps (sentence-transformers, sklearn, pyarrow)
    # on environments where they may be incompatible (NumPy ABI issues).
    import os
    if os.environ.get("MED_AGENT_ENABLE_HF", "0") != "1":
        _model_failed = True
        return

    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        # do not raise — mark failure and fallback deterministically
        _model = None
        _model_failed = True


def _hash_to_vector(text: str, dim: int = _DIM) -> List[float]:
    """Deterministic embedding based on repeated SHA256 hashing.

    Produces floats in approximately [-1, 1] and normalizes to unit length.
    """
    # Create a long byte stream by hashing text with appended counters
    digest = b""
    i = 0
    while len(digest) < dim * 4:
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        h.update(i.to_bytes(2, "little", signed=False))
        digest += h.digest()
        i += 1

    # Convert bytes to signed 32-bit ints and then to floats
    vals = []
    for j in range(dim):
        # take 4 bytes per value
        b = digest[j * 4 : j * 4 + 4]
        intval = int.from_bytes(b, "little", signed=False)
        # map to [-1,1]
        f = (intval / 0xFFFFFFFF) * 2.0 - 1.0
        vals.append(f)

    # normalize to unit length
    norm = sum(x * x for x in vals) ** 0.5
    if norm == 0:
        return vals
    return [x / norm for x in vals]


def get_embedding(text: str) -> List[float]:
    """Return an embedding vector for `text`.

    Tries the SentenceTransformer model first; on failure returns a deterministic
    hash-based vector so downstream components can operate without HF access.
    """
    global _model, _model_failed
    # Prefer remote HF Inference API if configured (avoids local heavy deps)
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    use_hf_api = os.environ.get("MED_AGENT_USE_HF_API", "1") == "1" and bool(hf_token)

    if use_hf_api:
        try:
            return _hf_api_embedding(text, hf_token)
        except Exception:
            # fallback to local model or hash-based embedding
            pass

    _try_load_model()
    if _model is not None:
        try:
            vec = _model.encode([text])[0]
            # ensure list[float]
            return list(map(float, vec.tolist() if hasattr(vec, "tolist") else vec))
        except Exception:
            # mark failed to avoid repeated heavy attempts
            global _model_failed
            _model = None
            _model_failed = True
            return _hash_to_vector(text)
    # fallback deterministic vector
    return _hash_to_vector(text)


def _hf_api_embedding(text: str, hf_token: str) -> List[float]:
    """Call Hugging Face Inference API feature-extraction pipeline for canonical model.

    Expects a valid `hf_token`. Returns a normalized vector list.
    """
    headers = {"Authorization": f"Bearer {hf_token}", "Accept": "application/json"}
    model = "sentence-transformers/all-MiniLM-L6-v2"
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
    resp = requests.post(url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # API may return nested list for tokens; reduce to a single vector if needed
    if isinstance(data, list) and data and isinstance(data[0], list):
        vec = data[0]
    else:
        raise RuntimeError("Unexpected HF API response for embeddings")

    # Normalize
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [float(x / norm) for x in vec]