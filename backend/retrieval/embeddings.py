"""Load scheme embeddings and embed queries via sentence-transformers."""
import io
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.config import SCHEMES_JSON, EMBEDDINGS_NPY, EMBEDDING_MODEL_ID

_embed_model: SentenceTransformer | None = None
_schemes: list[dict] = []
_embeddings: np.ndarray | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"[Embeddings] Loading sentence-transformer: {EMBEDDING_MODEL_ID}")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_ID)
        print("[Embeddings] sentence-transformer ready.")
    return _embed_model


def _read_file_bytes(path) -> bytes:
    """Read file using low-level os.read to avoid VirtioFS fcntl deadlock (EDEADLK)."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def load_schemes_and_embeddings():
    """Load schemes JSON and pre-computed embeddings into memory."""
    global _schemes, _embeddings

    if not SCHEMES_JSON.exists():
        raise FileNotFoundError(
            f"schemes.json not found at {SCHEMES_JSON}. "
            "Run: python scripts/download_data.py"
        )

    _schemes = json.loads(_read_file_bytes(SCHEMES_JSON).decode("utf-8"))

    if not EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(
            f"scheme_embeddings.npy not found at {EMBEDDINGS_NPY}. "
            "Run: python scripts/precompute_embeddings.py"
        )

    print(f"[Embeddings] Loading pre-computed embeddings ({len(_schemes)} schemes)...")
    _embeddings = np.load(io.BytesIO(_read_file_bytes(EMBEDDINGS_NPY)))
    print(f"[Embeddings] Ready — {len(_schemes)} schemes, shape {_embeddings.shape}")

    # Warm up the embedding model
    _get_embed_model()


def get_schemes() -> list[dict]:
    return _schemes


def get_embeddings() -> np.ndarray:
    return _embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a query string via sentence-transformers."""
    model = _get_embed_model()
    return model.encode(query, convert_to_numpy=True).astype(np.float32)
