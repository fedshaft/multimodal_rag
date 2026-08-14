"""Image search by text query: CLIP embeddings in Chroma.

This is the module the whole project's claim rests on. CLIP was trained on
hundreds of millions of (image, caption) pairs, so it puts a picture and a
description of that picture at nearly the same point in one shared space. We
therefore embed *pixels* at build time and *words* at query time, and compare
them directly.

No captioning anywhere. We are not describing images with a model and then
searching the descriptions — the search touches the image itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.evidence import Evidence  # noqa: E402
from index import store  # noqa: E402

COLLECTION = "image"
_model = None
_preprocess = None
_tokenizer = None


def get_model():
    """CLIP model, its image preprocessing, and its tokenizer. Loaded once."""
    global _model, _preprocess, _tokenizer
    if _model is None:
        import open_clip
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
        )
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
    return _model, _preprocess, _tokenizer


def embed_images(paths: list[str], batch_size: int = 16):
    """Image files -> normalised CLIP vectors. Unreadable files are skipped,
    so the caller is told which paths actually made it."""
    model, preprocess, _ = get_model()
    tensors, kept = [], []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(preprocess(img))
            kept.append(p)
        except Exception:
            continue
    if not tensors:
        return [], []

    vectors = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i + batch_size])
            v = model.encode_image(batch)
            v = v / v.norm(dim=-1, keepdim=True)
            vectors.extend(v.cpu().numpy())
    return vectors, kept


def embed_query(query: str):
    """A text query, embedded into the *same* space as the images."""
    model, _, tokenizer = get_model()
    with torch.no_grad():
        v = model.encode_text(tokenizer([query]))
        v = v / v.norm(dim=-1, keepdim=True)
    return v[0].cpu().numpy()


def build(evidence: list[Evidence], batch_size: int = 16) -> int:
    """Full rebuild from the image evidence. Returns how many were indexed."""
    images = [e for e in evidence if e.modality == "image"]
    col = store.reset_collection(COLLECTION)
    if not images:
        return 0

    by_path = {e.content: e for e in images}
    vectors, kept = embed_images([e.content for e in images], batch_size=batch_size)
    if not kept:
        return 0

    for i in range(0, len(kept), 64):
        chunk_paths = kept[i:i + 64]
        chunk_vecs = vectors[i:i + 64]
        records = [store.to_record(by_path[p]) for p in chunk_paths]
        col.add(
            ids=[r[0] for r in records],
            documents=[r[1] for r in records],
            metadatas=[r[2] for r in records],
            embeddings=[v.tolist() for v in chunk_vecs],
        )
    return len(kept)


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
