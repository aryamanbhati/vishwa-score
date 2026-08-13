"""FAISS retriever over the government loan scheme corpus.

Loads schemes.json, encodes with MiniLM-L6-v2, builds an in-memory FAISS
index. The index is cached to disk after the first build so subsequent
launches are instant.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
SCHEMES_PATH = DATA_DIR / "schemes.json"
INDEX_PATH = DATA_DIR / "faiss.index"
EMBED_MODEL = "paraphrase-MiniLM-L6-v2"


def _load_schemes() -> list[dict]:
    with open(SCHEMES_PATH) as f:
        return json.load(f)


_model: SentenceTransformer | None = None
_index: faiss.IndexFlatIP | None = None
_schemes: list[dict] | None = None


def _ensure_loaded():
    global _model, _index, _schemes
    if _model is not None:
        return

    _schemes = _load_schemes()
    _model = SentenceTransformer(EMBED_MODEL)

    if INDEX_PATH.exists():
        _index = faiss.read_index(str(INDEX_PATH))
    else:
        texts = [s["content"] for s in _schemes]
        embeddings = _model.encode(texts, normalize_embeddings=True).astype(np.float32)
        _index = faiss.IndexFlatIP(embeddings.shape[1])
        _index.add(embeddings)
        faiss.write_index(_index, str(INDEX_PATH))


def retrieve(query: str, state: str | None = None, top_k: int = 5) -> list[dict]:
    """Return the top-k most relevant scheme chunks for a query."""
    _ensure_loaded()
    vec = _model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, idxs = _index.search(vec, min(top_k * 3, len(_schemes)))

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = _schemes[idx].copy()
        chunk["score"] = round(float(score), 3)
        if state and state.lower() in chunk.get("state", "").lower():
            chunk["score"] += 0.1
        results.append(chunk)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def list_schemes() -> list[dict]:
    """Return all schemes (for display in the UI)."""
    _ensure_loaded()
    return list(_schemes)
