# Multimodal RAG — System Architecture

> A question-answering system that searches across text **and** images in a document
> collection, then generates a cited answer. Runs entirely on local hardware with no
> paid API calls.

---

## 1. What the system does

You give it a collection of documents (PDFs, standalone images, short videos). You ask a
question in plain English. It:

1. Finds the most relevant pieces — **including figures and diagrams, not just text**
2. Re-ranks those pieces so the best ones come first
3. Generates a written answer using a vision-capable language model
4. Shows you the exact sources it used, images included

**Why it matters:** conventional search only reads text. Real documents are full of
diagrams, charts, screenshots, and scanned pages that keyword search cannot see at all.

---

## 2. Architecture overview

The system runs in two distinct phases. This split matters — the expensive work happens
once, ahead of time.

### Phase A — Indexing (offline, run once per collection)

```
┌───────────┐
│   PDFs    │──► PyMuPDF ──┬──► text blocks ──► chunker ──► BGE-small ──► ┌────────────┐
└───────────┘              │                                             │ TEXT INDEX │
                           │                                             └────────────┘
                           └──► page images ──┐
┌───────────┐                                 │
│  Images   │─────────────────────────────────┼──► CLIP image encoder ──► ┌─────────────┐
└───────────┘                                 │                           │ IMAGE INDEX │
┌───────────┐                                 │                           └─────────────┘
│  Videos   │──► ffmpeg keyframes ────────────┘
└───────────┘
```

### Phase B — Query (online, runs per question)

```
        "show me the diagram explaining backpropagation"
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
   BGE-small                          CLIP text encoder
        │                                       │
        ▼                                       ▼
  ┌────────────┐                        ┌─────────────┐
  │ TEXT INDEX │                        │ IMAGE INDEX │
  └────────────┘                        └─────────────┘
        │                                       │
   top-30 chunks                          top-30 images
        │                                       │
        └──────────────► MERGE (RRF) ◄──────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  CROSS-ENCODER        │   slow but accurate
                  │  RERANKER             │   re-scores the shortlist
                  └───────────────────────┘
                              │
                    top-5 (text + images mixed)
                              │
                              ▼
                  ┌───────────────────────┐
                  │  VISION LLM (Ollama)  │   local, free, offline
                  └───────────────────────┘
                              │
                              ▼
              Answer  +  cited sources  +  source images
                              │
                              ▼
                      Streamlit interface
```

---

## 3. Components

| # | Component | Tool | What it does, in plain terms |
|---|---|---|---|
| 1 | Document parser | `PyMuPDF` | Opens PDFs, pulls out the text, the embedded figures, and a picture of each page |
| 2 | Video sampler | `ffmpeg` | Grabs one frame every N seconds so video becomes a set of images |
| 3 | Chunker | custom | Splits long text into paragraph-sized pieces (~300 words) so retrieval is precise |
| 4 | Text encoder | `bge-small-en-v1.5` | Turns a text chunk into ~384 numbers. Similar meanings → similar numbers |
| 5 | Image encoder | CLIP ViT-B/32 | Turns an image into numbers **in the same space as text**, so words can find pictures |
| 6 | Vector store | ChromaDB | Holds all those numbers on disk and finds the nearest ones fast |
| 7 | Keyword baseline | `rank_bm25` | Classic keyword search. Our comparison baseline, not part of the final system |
| 8 | Merger | RRF | Combines the text ranking and the image ranking into one list |
| 9 | Reranker | `bge-reranker-base` | Reads the question against each shortlisted result and re-scores it properly |
| 10 | Generator | Ollama + vision model | Reads the top results (text and images) and writes the answer |
| 11 | Interface | Streamlit | The screen: question box, answer, source thumbnails |

### The key idea: CLIP

CLIP was trained on hundreds of millions of (image, caption) pairs. As a result, the
sentence *"a red sports car"* and an actual **photograph** of a red sports car end up with
near-identical number representations.

This is what makes the system genuinely multimodal. We are not captioning images and then
searching the captions — we search the pixels directly, using words. No manual labelling
required anywhere in the pipeline.

### Why two separate indexes instead of one

CLIP handles short phrases well but is weak on long passages. A dedicated text encoder
(BGE) is much stronger on paragraphs but cannot see images at all. Rather than compromise,
we run both and merge the two ranked lists.

**Trade-off:** two searches instead of one, so roughly 2× query latency. Justified by
substantially better text retrieval quality. This is a design decision worth defending in
the report.

---

## 4. Retrieval pipeline in detail

**Stage 1 — Recall (fast, approximate).** Both indexes return their top 30. The goal here
is not to be right, it is to not miss anything. Cheap: ~10 ms.

**Stage 2 — Merge.** Reciprocal Rank Fusion combines the two lists using rank position
rather than raw scores, which avoids the problem that the two encoders produce scores on
incompatible scales.

**Stage 3 — Precision (slow, accurate).** A cross-encoder reads the question **together
with** each of the ~60 candidates and scores relevance directly. This is far more accurate
than comparing pre-computed numbers, but far too slow to run over the whole collection —
hence the two-stage design. Cost: ~80 ms.

**Stage 4 — Generation.** The top 5 results, text and images together, go to the local
vision model with an instruction to answer using only the provided sources and to cite them.

This two-stage recall-then-precision structure is how essentially every production search
system works (web search, e-commerce, recommendation feeds).

---

## 5. Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python **3.12** | Not 3.14 — ML libraries have no 3.14 wheels yet |
| Env manager | `uv` | `uv venv --python 3.12` |
| Parsing | `pymupdf` | |
| Embeddings | `sentence-transformers`, `open_clip_torch` | CPU-only builds are fine |
| Vector store | `chromadb` | Local file, no server process |
| Keyword baseline | `rank_bm25` | Pure Python, trivial to set up |
| Generation | **Ollama** + vision model | Free, offline, no API key |
| Interface | `streamlit` | |
| Metrics | `numpy` (hand-written) | Recall@K and nDCG@10 are ~15 lines each |

**No paid APIs anywhere in this system.** Everything runs locally.

### Hardware plan

| Machine | Role | Notes |
|---|---|---|
| Fedora PC (12 CPU, 15 GB, no GPU) | Development | Small model, ~10–20 s per answer. Fine for building |
| MacBook Pro M5 | **Demo** | Apple Silicon runs these models fast. Same code, larger model |

The only thing that changes between machines is the model name in the config file.

---

## 6. Evaluation design

Evaluation runs on **our own document collection**, using a hand-built set of 30–50
questions. This is deliberate: public benchmarks such as SciFact are text-only and COCO
contains no documents, so neither measures a multimodal RAG system end to end. They would
only test individual components.

### The evaluation set

30–50 questions about the document collection. For each one, record:

| Field | Example |
|---|---|
| `question` | "What does the architecture diagram for the encoder show?" |
| `relevant_ids` | `["doc3_chunk_18", "doc3_img_p8_1"]` |
| `answer_type` | `figure` |

The `answer_type` tag is the important one. Every question is labelled as answerable from
**text**, from a **figure/image**, or from **both**. This single field is what produces the
project's headline result.

### Result table — one eval set, every method

| Method | Recall@5 | MRR | nDCG@10 | Latency |
|---|---|---|---|---|
| BM25 (keyword, text only) | | | | ~5 ms |
| Dense text (BGE-small) | | | | ~10 ms |
| Hybrid (BM25 + dense) | | | | ~15 ms |
| Hybrid + CLIP images (multimodal) | | | | ~25 ms |
| Full system (+ cross-encoder rerank) | | | | ~100 ms |

This gives the quality-versus-latency trade-off discussion: each row adds accuracy and adds
milliseconds, and the report argues about where the sensible operating point is.

### Breakdown by question type — the headline result

| Question answerable from | Text-only system | Full multimodal system |
|---|---|---|
| Text | | |
| A figure or diagram | | |
| Both | | |

**The expected finding:** the two systems perform comparably on text questions, but the
text-only system collapses on figure questions because keyword and text-embedding search
cannot see images at all. That contrast is the central argument of the project, and it is
one bar chart on a slide.

### Generation quality

Measured on the same question set: does the generated answer cite the source that actually
contains the answer? Reported as citation accuracy, with a handful of failure cases
discussed qualitatively.

### Optional extension if time allows

Run the CLIP image retriever against **MS-COCO val2017** (5,000 images) as an external
sanity check, confirming the component behaves in line with published numbers. Nice to
have, not required.

---

## 7. Code structure and extension points

The MVP is deliberately **not** written as a throwaway prototype. The interfaces below are
designed so that adding video, audio, or table retrieval later means adding a new module,
not rewriting the pipeline.

### The core abstraction: `Evidence`

Everything retrieved — a paragraph, a figure, later a video frame or an audio segment — is
returned as the same kind of object:

```python
@dataclass
class Evidence:
    id: str                 # "doc3_img_p8_1"
    modality: str           # "text" | "image"  (later: "video" | "audio" | "table")
    content: str            # the text, or a path to the image file
    source_doc: str         # which document it came from
    page: int | None        # page number, if applicable
    score: float            # relevance score from the current stage
    metadata: dict          # anything modality-specific
```

Every stage of the pipeline speaks this one type:

```python
ingest(path)            -> list[Evidence]      # documents in, evidence out
embed(evidence)         -> None                # writes vectors into the index
retrieve(query, k)      -> list[Evidence]      # searches all indexes, merges
rerank(query, evidence) -> list[Evidence]      # re-orders, same type in and out
generate(query, evidence) -> Answer            # writes the answer, cites the evidence
evaluate(eval_set)      -> dict                # metrics
```

**Why this matters for the major project:** adding video retrieval later means writing one
new ingester that returns `Evidence(modality="video", ...)`. `retrieve`, `rerank`,
`generate`, and `evaluate` need no changes at all. Without this abstraction, each new
modality means touching every stage.

### Repository layout

```
blackhole/
├── architecture.md          this file
├── PLAN.md                  day-by-day build plan and status
├── config.py                model names, paths, top-k values
├── core/
│   └── evidence.py          the Evidence dataclass — the shared contract
├── ingest/
│   ├── pdf_parser.py        PDF → text blocks + images  → list[Evidence]
│   ├── chunker.py           long text → chunks
│   └── video_sampler.py     (major project) video → keyframes → list[Evidence]
├── index/
│   ├── text_index.py        BGE embeddings → Chroma
│   ├── image_index.py       CLIP embeddings → Chroma
│   └── bm25_index.py        keyword baseline
├── retrieve/
│   ├── search.py            query all indexes, merge (RRF)
│   └── rerank.py            cross-encoder
├── generate/
│   └── answer.py            Ollama call with images + citations
├── eval/
│   ├── eval_set.json        the 30–50 labelled questions
│   ├── metrics.py           Recall@K, MRR, nDCG@10
│   └── run_eval.py          runs every method → the result table
├── app.py                   Streamlit interface
└── data/                    documents and datasets (not committed)
```

**One config switch drives the whole ablation.** `retrieve()` reads which indexes are
enabled from config, so producing every row of the result table means changing a setting
and re-running `run_eval.py` — not writing five separate scripts.

---

## 8. Scope boundaries

**In scope:** text + image retrieval, keyframe-level video search, two-stage retrieval,
local generation with citations, benchmark evaluation.

**Out of scope for this semester (candidates for the major project):**

- Training or fine-tuning any model — all encoders are used frozen
- A single unified embedding space (would replace the two-index + merge design)
- Temporal grounding in video ("*at what timestamp*", not just "*in which video*")
- Table structure understanding and numerical reasoning
- Scaling beyond ~100k items (would require approximate-nearest-neighbour indexing)

---

*Last updated: 2026-08-13*
