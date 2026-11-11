"""Stage 1 and 2 of the query pipeline: ask every enabled index, merge the lists.

Which indexes run is *not* decided here — it comes from the active `Method` in
config.py. That is what makes day 5's ablation a loop over five settings rather
than five copies of this file.

The merge is Reciprocal Rank Fusion (RRF): each result scores `1 / (60 + rank)`
in every list it appears in, and those scores are added up. It uses rank
position only, never the raw score, which matters because BM25 scores
(unbounded, ~0-30), BGE cosine similarity (0-1) and CLIP similarity (0-1, and
in practice squashed into 0.1-0.4) are on completely incompatible scales.
Averaging them directly would be meaningless; averaging ranks is not.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.evidence import Evidence  # noqa: E402
from index import bm25_index, image_index, text_index  # noqa: E402

INDEXES = {
    "bm25": bm25_index,
    "dense": text_index,
    "clip": image_index,
}


@dataclass
class Retrieved:
    """The final evidence plus everything needed to explain how it got there.

    Timings are collected on every single query, not bolted on at eval time:
    latency is one half of the results table, and a number that was never
    measured cannot be recovered afterwards.
    """

    query: str
    method: str
    evidence: list[Evidence] = field(default_factory=list)
    candidates: list[Evidence] = field(default_factory=list)
    """Everything the merge produced, before the final cut. Day 5 measures
    recall over this — a hit that reached the shortlist but lost the last slot
    is a different failure from one that was never retrieved at all."""

    timings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def images(self) -> list[Evidence]:
        return [e for e in self.evidence if e.is_image]


# ---------------------------------------------------------------- warm-up

def warm_up(method: config.Method | None = None) -> float:
    """Load the encoders before anything is timed.

    Day 2 trap, and it will silently ruin the results table if it is skipped:
    the models load lazily on first use, so the first query measures a ~9 s
    model load and every later one measures ~30 ms. Neither number is the
    query cost. Call this once at process start; it returns the seconds spent
    so a script can report it separately as one-off cost.
    """
    method = method or config.METHOD
    t0 = time.perf_counter()
    if method.use_dense:
        text_index.embed_query("warm up")
    if method.use_clip:
        image_index.embed_query("warm up")
    if method.use_rerank:
        from retrieve import rerank
        rerank.warm_up()
    return time.perf_counter() - t0


# ---------------------------------------------------------------- merging

VISUAL_KEYWORDS = {
    "diagram", "figure", "chart", "graph", "plot", "illustration",
    "flowchart", "drawing", "picture", "schematic", "architecture",
    "visual", "layout", "timeline", "table", "hardware", "slide",
    "structure", "organization", "hierarchy", "state machine",
}


def has_visual_intent(query: str | None) -> bool:
    """Returns True if the query explicitly or implicitly asks for a visual piece of evidence."""
    if not query:
        return False
    cleaned = query.lower().replace("?", " ").replace(",", " ").replace(".", " ").replace(":", " ")
    words = set(cleaned.split())
    return bool(words & VISUAL_KEYWORDS)


def rrf_merge(rankings: dict[str, list[Evidence]],
              k: int = config.RRF_K,
              weights: dict[str, float] | None = None) -> list[Evidence]:
    """Several ranked lists in, one ranked list out.

    Supports per-index weighting (Weighted RRF):
    score(d) = sum_m w_m / (k + rank_m(d))
    """
    weights = weights or getattr(config, "RRF_WEIGHTS", None)
    fused: dict[str, float] = {}
    keep: dict[str, Evidence] = {}
    ranks: dict[str, dict[str, int]] = {}
    raw: dict[str, dict[str, float]] = {}

    for index_name, results in rankings.items():
        w = weights.get(index_name, 1.0) if weights else 1.0
        for rank, e in enumerate(results, start=1):
            fused[e.id] = fused.get(e.id, 0.0) + w / (k + rank)
            keep.setdefault(e.id, e)
            ranks.setdefault(e.id, {})[index_name] = rank
            raw.setdefault(e.id, {})[index_name] = e.score

    ordered = sorted(fused, key=lambda i: -fused[i])
    return [
        replace(
            keep[i],
            score=fused[i],
            metadata={**keep[i].metadata, "ranks": ranks[i], "raw_scores": raw[i]},
        )
        for i in ordered
    ]


def clip_score(e: Evidence) -> float:
    """The image's own CLIP similarity, before fusion flattened it into a rank."""
    return float(e.metadata.get("raw_scores", {}).get("clip", 0.0))


def select_final(candidates: list[Evidence], k: int | None = None,
                 max_images: int | None = None,
                 floor: float | None = None,
                 query: str | None = None,
                 adaptive: bool | None = None) -> list[Evidence]:
    """Cut the shortlist down to what the LLM actually sees.

    When adaptive is True:
    - If query has visual intent ("diagram", "figure", "chart"), image quota is
      expanded to min(max_images + 1, k - 1) and the floor is relaxed by
      VISUAL_FLOOR_DELTA so candidates are not dropped at the final cut.
      High-scoring images are positioned at rank 1.
    - If query is purely textual, image quota is restricted to 1 and floor is
      tightened, protecting text evidence slots from irrelevant image intrusion.
    """
    k = k or config.TOP_K_FINAL
    max_images = config.MAX_IMAGES_FINAL if max_images is None else max_images
    floor = config.IMAGE_SCORE_FLOOR if floor is None else floor
    adaptive = config.ADAPTIVE_IMAGE_SLOTS if adaptive is None else adaptive

    is_visual = has_visual_intent(query) if (adaptive and query) else False

    if is_visual:
        effective_floor = max(floor - config.VISUAL_FLOOR_DELTA, 0.18)
        eff_max_images = min(max_images + 1, max(k - 1, 1))
    elif adaptive:
        effective_floor = floor + 0.02
        eff_max_images = min(max_images, 1)
    else:
        effective_floor = floor
        eff_max_images = max_images

    texts = [e for e in candidates if not e.is_image]
    images = [e for e in candidates if e.is_image and clip_score(e) >= effective_floor]

    n_img = min(eff_max_images, len(images), max(k - 1, 0))
    chosen_images = images[:n_img]
    chosen_texts = texts[:k - n_img]

    out: list[Evidence] = []
    # If the user specifically sought a visual element and we found one, lead with it:
    if is_visual and chosen_images:
        out.append(chosen_images[0])
        rem_images = chosen_images[1:]
        rem_texts = chosen_texts
        for i in range(max(len(rem_texts), len(rem_images))):
            if i < len(rem_texts):
                out.append(rem_texts[i])
            if i < len(rem_images):
                out.append(rem_images[i])
    else:
        for i in range(max(len(chosen_texts), len(chosen_images))):
            if i < len(chosen_texts):
                out.append(chosen_texts[i])
            if i < len(chosen_images):
                out.append(chosen_images[i])
    return out[:k]


# ---------------------------------------------------------------- the pipeline

def search(query: str, k: int | None = None,
           method: config.Method | None = None) -> Retrieved:
    """Query -> final evidence, with per-stage timings attached."""
    k = k or config.TOP_K_FINAL
    method = method or config.METHOD
    out = Retrieved(query=query, method=method.name)
    t_start = time.perf_counter()

    rankings: dict[str, list[Evidence]] = {}
    for name in method.enabled_indexes:
        t0 = time.perf_counter()
        rankings[name] = INDEXES[name].search(query, k=config.TOP_K_PER_INDEX)
        out.timings[name] = time.perf_counter() - t0
        out.counts[name] = len(rankings[name])

    t0 = time.perf_counter()
    merged = rrf_merge(rankings)
    out.timings["merge"] = time.perf_counter() - t0
    out.counts["merged"] = len(merged)

    if method.use_rerank and merged:
        from retrieve import rerank
        t0 = time.perf_counter()
        merged = rerank.rerank(query, merged, top_n=config.RERANK_TOP_N)
        out.timings["rerank"] = time.perf_counter() - t0

    out.candidates = merged
    out.evidence = select_final(
        merged, k=k, query=query, adaptive=getattr(method, "adaptive_slots", False)
    )
    out.timings["total"] = time.perf_counter() - t_start
    return out


def retrieve(query: str, k: int | None = None) -> list[Evidence]:
    """The contract in architecture.md §7: query in, ranked evidence out.

    `search()` is the same thing with the timings and the shortlist kept, which
    is what the eval and the interface want; this is for everything else.
    """
    return search(query, k=k).evidence
