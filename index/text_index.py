"""Dense text search: BGE-small embeddings in Chroma.

"Dense" means every chunk becomes 384 numbers capturing its meaning, and search
compares meanings rather than words. It finds "the model learns by adjusting
weights" for the query "how does training work" — no shared keywords at all,
which is exactly what BM25 cannot do.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.evidence import Evidence  # noqa: E402
from index import store  # noqa: E402

COLLECTION = "text"
_model = None


def get_model():
    """Loaded once, on first use. Importing this module stays cheap."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.TEXT_MODEL)
    return _model


def embed_passages(texts: list[str], batch_size: int = 32):
    """Stored chunks are embedded with NO prefix — see embed_query."""
    return get_model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_query(query: str):
    """Queries get BGE's instruction prefix; passages do not.

    This asymmetry is real and easy to get wrong: BGE was trained that way, and
    dropping the prefix measurably costs retrieval quality.
    """
    return get_model().encode(
        [config.TEXT_QUERY_PREFIX + query], normalize_embeddings=True,
        show_progress_bar=False,
    )[0]


def build(evidence: list[Evidence], batch_size: int = 64) -> int:
    """Full rebuild from the text evidence. Returns how many were indexed."""
    texts = [e for e in evidence if e.modality == "text"]
    col = store.reset_collection(COLLECTION)
    if not texts:
        return 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        records = [store.to_record(e) for e in batch]
        vectors = embed_passages([e.content for e in batch])
        col.add(
            ids=[r[0] for r in records],
            documents=[r[1] for r in records],
            metadatas=[r[2] for r in records],
            embeddings=[v.tolist() for v in vectors],
        )
    return len(texts)


def search(query: str, k: int | None = None) -> list[Evidence]:
    k = k or config.TOP_K_PER_INDEX
    col = store.get_collection(COLLECTION)
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=[embed_query(query).tolist()],
        n_results=min(k, col.count()),
    )
    return store.unpack_query(res)


def count() -> int:
    try:
        return store.get_collection(COLLECTION).count()
    except Exception:
        return 0
