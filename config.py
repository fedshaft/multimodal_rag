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

RERANK_MODEL = "BAAI/bge-reranker-base"
"""Cross-encoder: reads the question and one candidate together and scores the
pair. Much more accurate than comparing pre-computed vectors, far too slow to
run over the whole collection — hence only over the shortlist."""

GEN_MODEL = os.environ.get("RAG_GEN_MODEL", "qwen2.5vl:3b")
"""The vision LLM served by Ollama. THIS is the one line that differs between
machines — override it without editing the file:

    Fedora PC (dev, no GPU):  qwen2.5vl:3b     ~3 GB, slow but works
    MacBook Pro M5 (demo):    RAG_GEN_MODEL=qwen2.5vl:7b   ~6 GB, much better
"""

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GEN_TIMEOUT = 300  # seconds; CPU generation with images is genuinely slow

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

MIN_VECTOR_DRAWINGS = 8
"""Vector drawing commands needed before a page counts as holding a diagram.
A handful of commands is just a rule, a border or a table line."""

# ---------------------------------------------------------------- retrieval

TOP_K_PER_INDEX = 30   # stage 1: each index returns this many (recall)
RRF_K = 60             # the constant in reciprocal rank fusion, 60 is standard
RERANK_TOP_N = 60      # stage 3: cross-encoder rescores this many candidates
TOP_K_FINAL = 5        # what actually reaches the LLM


@dataclass
class Method:
    """One row of the results table."""

    name: str
    use_bm25: bool = False
    use_dense: bool = False
    use_clip: bool = False
    use_rerank: bool = False
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
- Be concise: a few sentences unless the question needs more."""

# ---------------------------------------------------------------- evaluation

RECALL_AT = (1, 3, 5, 10)
NDCG_AT = 10
ANSWER_TYPES = ("text", "figure", "both")
"""Every eval question is tagged with one of these. The breakdown by this tag
is the project's headline result, so it is fixed vocabulary, not free text."""
