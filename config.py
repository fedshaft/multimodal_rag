"""Every knob in one place.

Two things depend on this file being the single source of truth:

1. The Fedora PC and the MacBook differ *only* in GEN_MODEL below.
2. The five rows of the results table are five values of METHOD — retrieval
   reads which indexes are enabled from here, so producing the ablation means
   changing one setting, not writing five scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- paths

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

DOCS_DIR = Path(os.environ.get("RAG_DOCS_DIR", DATA_DIR / "own"))
"""The document collection. Overridable so the self-test can run the whole
pipeline over a throwaway corpus without touching the real one."""

INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", DATA_DIR))
"""Everything ingestion *writes* lives under here. Points somewhere else during
the self-test, so a test run can never corrupt the real indexes."""

IMAGES_DIR = INDEX_DIR / "images"    # figures extracted from PDFs, written by ingestion
CHROMA_DIR = INDEX_DIR / "chroma"    # the vector store, a plain directory on disk
BM25_PATH = INDEX_DIR / "bm25.pkl"
EVAL_SET = ROOT / "eval" / "eval_set.json"
RESULTS_DIR = ROOT / "eval" / "results"

for _d in (DATA_DIR, DOCS_DIR, IMAGES_DIR, CHROMA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- models

TEXT_MODEL = "BAAI/bge-small-en-v1.5"
"""Text encoder: a sentence -> 384 numbers. Small and CPU-fast (~130 MB)."""

TEXT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
"""BGE was trained with this exact prefix on *queries only*, never on the
stored passages. Omitting it measurably hurts retrieval — a known and easy
mistake to make."""

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
"""Image encoder: puts pictures and words in the *same* number space, which is
what lets a typed question find a diagram. ~600 MB."""

RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
"""Cross-encoder: reads the question and one candidate together and scores the
pair. Much more accurate than comparing pre-computed vectors, far too slow to
run over the whole collection — hence only over the shortlist.

Measured on this PC (12 CPU cores, no GPU), 60 candidates of ~300 words:

    BAAI/bge-reranker-base              ~14 s   default; the accurate one
    cross-encoder/ms-marco-MiniLM-L-6-v2 ~1.6 s  9x faster, smaller, English only

Both are cached locally. Override with RAG_RERANK_MODEL when iterating, and
report both in the results table — accuracy against latency is exactly the
trade-off the write-up is supposed to show."""

GEN_MODEL = os.environ.get("RAG_GEN_MODEL", "qwen2.5vl:3b")
"""The vision LLM served by Ollama. THIS is the one line that differs between
machines — override it without editing the file:

    Fedora PC (dev, no GPU):  qwen2.5vl:3b     ~3 GB, slow but works
    MacBook Pro M5 (demo):    RAG_GEN_MODEL=qwen2.5vl:7b   ~6 GB, much better
"""

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GEN_TIMEOUT = 300  # seconds; CPU generation with images is genuinely slow

GEN_NUM_CTX = int(os.environ.get("RAG_GEN_NUM_CTX", "8192"))
"""Context window passed to Ollama on every call. A question plus five evidence
items (up to three of them images) runs to ~4,300 tokens, which overflows
Ollama's 4,096 default with a 400 error. Set explicitly so every machine
behaves the same whatever its Ollama default is."""

GEN_TEMPERATURE = 0.0
"""Greedy decoding. Sampling would make the same question give a different
answer on every run, which would mean day 5's citation accuracy — and the demo
— depend on luck. Zero is not fully deterministic on all backends, but it
removes the main source of run-to-run variation."""

# ---------------------------------------------------------------- ingestion

CHUNK_WORDS = 300
CHUNK_OVERLAP_WORDS = 50
"""Chunks overlap so a sentence spanning a boundary is not lost from both sides."""

MIN_CHUNK_WORDS = 20
"""Drop fragments shorter than this — page numbers, headers, stray captions."""

MIN_IMAGE_PIXELS = 128
"""Ignore images smaller than this on either side: logos, bullet icons, rules."""

PAGE_RENDER_DPI = 150
"""Whole-page renders, used for scanned pages and slide-style layouts where the
figure is not an embedded image but part of the page itself."""

PAGE_RENDER_MODE = os.environ.get("RAG_PAGE_RENDER", "auto")
"""When to also store a picture of the whole page.

    "auto"   only when the page would otherwise be invisible to image search
             (see below) — the default, and the right one for most collections
    "all"    every page. Use for slide decks that are essentially all diagrams
    "never"  embedded figures only

"auto" renders a page when either:
  - it has almost no extractable text (a scanned page — text search is blind to
    it, so the page render is the *only* way to retrieve it), or
  - it has no embedded raster images but plenty of vector drawing commands.
    This second case matters more than it sounds: diagrams in slide decks and
    papers are very often vector art, so `get_images()` returns nothing at all
    and the figure would silently never be indexed."""

MIN_PAGE_TEXT_WORDS = 30
"""Below this, a page counts as "no real text" — probably a scan or a title slide."""

MAX_IMAGE_PAGE_REPEATS = 5
"""Drop an image that is drawn on more than this many pages of one document.

Slide templates put a cover logo and a footer graphic on every slide. Measured
on the Silberschatz decks the split is unambiguous: 371 images appear on 1-4
pages (real diagrams), and 44 appear on 25-84 pages, almost all of them the
same 391x297 template graphic. Indexing those 44 would put decoration into the
image index, where it can only ever be a false match."""

MIN_VECTOR_DRAWINGS = 8
"""Vector drawing commands needed before a page counts as holding a diagram.
A handful of commands is just a rule, a border or a table line."""

# ---------------------------------------------------------------- retrieval

TOP_K_PER_INDEX = 30   # stage 1: each index returns this many (recall)
RRF_K = 40             # the constant in reciprocal rank fusion, 60 is standard
RERANK_TOP_N = 60      # stage 3: cross-encoder rescores this many candidates
TOP_K_FINAL = 5        # what actually reaches the LLM

MAX_IMAGES_FINAL = 3
"""How many of the TOP_K_FINAL slots may be images.

The reranker is a *text* cross-encoder: it cannot read a picture, so images can
never win a slot on reranked score. Without a reserved quota the full system
would quietly become text-only — losing exactly the thing this project is
about. Two slots out of five leaves room for the text that usually carries the
actual answer."""

IMAGE_SCORE_FLOOR = 0.2
"""Minimum CLIP similarity before an image may take one of those slots."""

RRF_WEIGHTS: dict[str, float] = {
    "bm25": float(os.environ.get("RAG_WEIGHT_BM25", "1.0")),
    "dense": float(os.environ.get("RAG_WEIGHT_DENSE", "1.0")),
    "clip": float(os.environ.get("RAG_WEIGHT_CLIP", "1.0")),
}

ADAPTIVE_IMAGE_SLOTS: bool = bool(int(os.environ.get("RAG_ADAPTIVE_SLOTS", "1")))
"""When True, visual-intent queries dynamically adjust image quota and floor."""

VISUAL_FLOOR_DELTA: float = float(os.environ.get("RAG_VISUAL_FLOOR_DELTA", "0.03"))


@dataclass
class Method:
    """One row of the results table."""

    name: str
    use_bm25: bool = False
    use_dense: bool = False
    use_clip: bool = False
    use_rerank: bool = False
    adaptive_slots: bool = False
    note: str = ""

    @property
    def enabled_indexes(self) -> list[str]:
        on = []
        if self.use_bm25:
            on.append("bm25")
        if self.use_dense:
            on.append("dense")
        if self.use_clip:
            on.append("clip")
        return on


METHODS: dict[str, Method] = {
    "bm25": Method(
        "BM25 (keyword, text only)", use_bm25=True,
        note="classic keyword search — the baseline everything is compared against",
    ),
    "dense": Method(
        "Dense text (BGE-small)", use_dense=True,
        note="meaning-based text search; finds paraphrases keyword search misses",
    ),
    "hybrid": Method(
        "Hybrid (BM25 + dense)", use_bm25=True, use_dense=True,
        note="the strongest text-only system — the honest comparison point",
    ),
    "multimodal": Method(
        "Hybrid + CLIP images", use_bm25=True, use_dense=True, use_clip=True,
        note="adds image retrieval — this is where figure questions should jump",
    ),
    "full": Method(
        "Full system (+ reranker)", use_bm25=True, use_dense=True, use_clip=True,
        use_rerank=True, note="everything on; slowest and most accurate",
    ),
    "tuned": Method(
        "Tuned system (adaptive + rerank)", use_bm25=True, use_dense=True, use_clip=True,
        use_rerank=True, adaptive_slots=True,
        note="hyperparameter-tuned fusion and query-adaptive visual slotting",
    ),
}

METHOD = METHODS[os.environ.get("RAG_METHOD", "full")]
"""Which configuration is active. run_eval.py loops over all five; everything
else (the CLI, the Streamlit app) uses this default."""

# ---------------------------------------------------------------- generation

SYSTEM_PROMPT = """You are a careful research assistant answering questions about a \
document collection.

Rules:
- Answer using ONLY the numbered evidence provided. Do not use outside knowledge.
- Cite the evidence you used by its number, like [1] or [2], in the sentence it supports.
- Some evidence is an image. Describe what you actually see in it; do not guess at \
content that is not visible.
- If the evidence does not answer the question, say "Not found in the documents." \
Do not fill the gap with a plausible-sounding guess.
- Be concise: a few sentences unless the question needs more.
- Always write the answer in full sentences. A bare citation like "[2]" is not \
an answer — say what the evidence shows, then cite it."""

GEN_IMAGE_MAX_PX = 1024
"""Longest edge of an image sent to the LLM; larger ones are downscaled first.

A whole-page render at PAGE_RENDER_DPI is ~1240x1755, and a vision model turns
that into a great many image tokens. Measured on this PC with qwen2.5vl:3b: a
full-page render took **150 s** to answer against **62 s** for a small extracted
figure. Downscaling costs nothing in retrieval (the index keeps the full-size
file) and only the LLM's reading of an already-selected figure is affected."""

# ---------------------------------------------------------------- evaluation

RECALL_AT = (1, 3, 5, 10)
NDCG_AT = 10
ANSWER_TYPES = ("text", "figure", "both")
"""Every eval question is tagged with one of these. The breakdown by this tag
is the project's headline result, so it is fixed vocabulary, not free text."""
