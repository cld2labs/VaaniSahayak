"""
Pre-compute embeddings for all 723 schemes and save as backend/data/scheme_embeddings.npy.
Run once — takes ~2 minutes on CPU, ~30s on MPS/GPU.

Run from project root:
    python scripts/precompute_embeddings.py
"""
import json
import numpy as np
import sys
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Install dependencies first: pip install -r backend/requirements.txt")
    sys.exit(1)

SCHEMES_JSON = Path(__file__).parent.parent / "backend" / "data" / "schemes.json"
EMBEDDINGS_NPY = Path(__file__).parent.parent / "backend" / "data" / "scheme_embeddings.npy"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def scheme_to_text(scheme: dict) -> str:
    parts = [
        scheme.get("name", ""),
        scheme.get("description", ""),
        scheme.get("eligibility", ""),
        scheme.get("category", ""),
    ]
    return " | ".join(p for p in parts if p)


def main():
    if not SCHEMES_JSON.exists():
        print(f"ERROR: {SCHEMES_JSON} not found. Run scripts/download_data.py first.")
        sys.exit(1)

    with open(SCHEMES_JSON, encoding="utf-8") as f:
        schemes = json.load(f)
    print(f"Loaded {len(schemes)} schemes.")

    print(f"Loading embedding model: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID)

    texts = [scheme_to_text(s) for s in schemes]
    print("Computing embeddings (this takes a few minutes on first run)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    np.save(str(EMBEDDINGS_NPY), embeddings)
    print(f"Saved embeddings → {EMBEDDINGS_NPY}")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
