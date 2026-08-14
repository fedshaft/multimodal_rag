"""Shared plumbing for the Chroma-backed indexes.

Chroma metadata values must be str/int/float/bool — no None, no nested dicts —
so Evidence has to be flattened on the way in and rebuilt on the way out. That
conversion lives here once rather than in each index.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.evidence import Evidence  # noqa: E402

_PAGE_NONE = -1  # Chroma cannot store None, so None page becomes this


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(name: str):
    """Cosine distance, which is what normalised embeddings want. Chroma
    defaults to squared L2 — on normalised vectors the ranking happens to be
    the same, but the numbers are much harder to read."""
    return get_client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def to_record(e: Evidence) -> tuple[str, str, dict]:
    """Evidence -> (id, document, metadata) for Chroma."""
    meta = {
        "modality": e.modality,
        "source_doc": e.source_doc,
        "page": e.page if e.page is not None else _PAGE_NONE,
    }
    for k, v in e.metadata.items():
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
    return e.id, e.content, meta


def from_record(id_: str, document: str, meta: dict, score: float) -> Evidence:
    """Chroma row -> Evidence."""
    page = meta.get("page", _PAGE_NONE)
    extra = {k: v for k, v in meta.items()
             if k not in ("modality", "source_doc", "page")}
    return Evidence(
        id=id_,
        modality=meta.get("modality", "text"),
        content=document,
        source_doc=meta.get("source_doc", "?"),
        page=None if page == _PAGE_NONE else int(page),
        score=score,
        metadata=extra,
    )


def unpack_query(res: dict) -> list[Evidence]:
    """Chroma's parallel-lists query result -> list[Evidence].

    Chroma returns *distance* (lower is better). We store similarity
    (higher is better) so every stage of the pipeline agrees on direction.
    """
    out: list[Evidence] = []
    if not res.get("ids") or not res["ids"][0]:
        return out
    for id_, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(from_record(id_, doc, meta, score=1.0 - float(dist)))
    return out


def reset_collection(name: str):
    """Drop and recreate. Ingestion is a full rebuild — re-running it must not
    leave stale evidence from documents that have since been deleted."""
    client = get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
