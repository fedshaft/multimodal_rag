"""Stage 3: re-score the shortlist with a cross-encoder.

The difference from stage 1 in one line: the indexes compare two vectors that
were computed *separately* and never saw each other, while a cross-encoder
reads the question and the candidate **together in one pass** and outputs a
relevance score. Far more accurate, and far too slow to run over a whole
collection — 60 pairs is fine, 6000 is not. Hence recall first, precision after.

One honest limitation, worth stating in the report: `bge-reranker-base` is a
text model. It cannot look at an image, so image evidence keeps the rank fusion
gave it and only the text is re-scored. Selecting a final mix from those two
lanes is `search.select_final`'s job.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.evidence import Evidence  # noqa: E402

_model = None


def get_model():
    """Loaded once, on first use — importing this module stays cheap."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(config.RERANK_MODEL, max_length=512)
    return _model


def warm_up() -> float:
    """Load and run the model once so the first real query is not timed while
    a 1.1 GB cross-encoder comes off disk. Returns seconds spent."""
    t0 = time.perf_counter()
    get_model().predict([("warm up", "warm up")])
    return time.perf_counter() - t0


def rerank(query: str, evidence: list[Evidence],
           top_n: int | None = None, batch_size: int = 16) -> list[Evidence]:
    """Re-order the shortlist. Same type in, same type out.

    Only the first `top_n` candidates are re-scored — that is the point of a
    shortlist. Anything past the cut keeps its fused order and stays at the
    back, so nothing is silently dropped.
    """
    top_n = top_n or config.RERANK_TOP_N
    head, tail = evidence[:top_n], evidence[top_n:]

    texts = [e for e in head if not e.is_image]
    images = [e for e in head if e.is_image]
    if not texts:
        return head + tail

    scores = get_model().predict(
        [(query, e.content) for e in texts],
        batch_size=batch_size,
        show_progress_bar=False,
    )
    rescored = [
        replace(e, score=float(s),
                metadata={**e.metadata, "rerank_score": float(s)})
        for e, s in zip(texts, scores)
    ]
    rescored.sort(key=lambda e: -e.score)

    # Images are returned after the reranked text, still in fused order. This
    # is not a judgement that they are less relevant — they simply were not
    # scored on the same scale, so mixing them by score here would be
    # meaningless. select_final() decides the actual mix.
    return rescored + images + tail
