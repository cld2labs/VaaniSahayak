"""Query → top-k relevant schemes."""
import numpy as np
from backend.config import TOP_K_SCHEMES
from backend.retrieval.embeddings import get_schemes, get_embeddings, embed_query


def retrieve(query: str, top_k: int = TOP_K_SCHEMES) -> list[dict]:
    """
    Embed the query and return top-k schemes by cosine similarity.
    Falls back to keyword search if embeddings are not available.
    """
    schemes = get_schemes()
    embeddings = get_embeddings()

    if embeddings is None or len(schemes) == 0:
        return []

    q_emb = embed_query(query)

    # Cosine similarity: normalize then dot product
    norm_emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    norm_q = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    scores = norm_emb @ norm_q

    top_indices = np.argsort(scores)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        scheme = schemes[int(idx)].copy()
        scheme["_score"] = float(scores[idx])
        results.append(scheme)
    return results


def keyword_fallback(query: str, top_k: int = TOP_K_SCHEMES) -> list[dict]:
    """Simple keyword overlap as a fallback retrieval method."""
    query_words = set(query.lower().split())
    scored = []
    for scheme in get_schemes():
        text = f"{scheme.get('name','')} {scheme.get('description','')}".lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, scheme))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_k]]
