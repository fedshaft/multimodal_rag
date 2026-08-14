"""Keyword search: BM25 over the same text chunks.

BM25 scores a document by how many of the query's words it contains, weighting
rare words more heavily and long documents less. It has no idea what anything
*means* — but it is extremely good at exact terms, names and acronyms, which is
precisely where embedding search is weakest. That is why it is both the
baseline and a component of the hybrid system.

It has no model and no vectors, so it lives in a pickle file rather than Chroma.
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.evidence import Evidence  # noqa: E402
from index import store  # noqa: E402

_TOKEN = re.compile(r"[a-z0-9]+")
_index = None


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Deliberately plain — no stemming and no
    stopword list, so the baseline stays the textbook BM25 everyone recognises."""
    return _TOKEN.findall(text.lower())


def build(evidence: list[Evidence]) -> int:
    """Full rebuild from the text evidence. Returns how many were indexed."""
    from rank_bm25 import BM25Okapi

    texts = [e for e in evidence if e.modality == "text"]
    if not texts:
        config.BM25_PATH.unlink(missing_ok=True)
        return 0

    corpus = [tokenize(e.content) for e in texts]
    payload = {
        "bm25": BM25Okapi(corpus),
        "records": [store.to_record(e) for e in texts],
    }
    config.BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_PATH, "wb") as fh:
        pickle.dump(payload, fh)

    global _index
    _index = payload
    return len(texts)


def _load():
    global _index
    if _index is None:
        if not config.BM25_PATH.exists():
            return None
        with open(config.BM25_PATH, "rb") as fh:
            _index = pickle.load(fh)
    return _index


def search(query: str, k: int | None = None) -> list[Evidence]:
    k = k or config.TOP_K_PER_INDEX
    idx = _load()
    if idx is None:
        return []

    scores = idx["bm25"].get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    out = []
    for i in ranked:
        if scores[i] <= 0:  # no query term present at all — not a result
            continue
        id_, doc, meta = idx["records"][i]
        out.append(store.from_record(id_, doc, meta, score=float(scores[i])))
    return out


def count() -> int:
    idx = _load()
    return len(idx["records"]) if idx else 0
