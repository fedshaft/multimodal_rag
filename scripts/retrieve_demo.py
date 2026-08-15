"""Day 3's gate: run the real retrieval pipeline and look hard at what it returns.

    .venv/bin/python scripts/retrieve_demo.py                      # the 10 stock queries
    .venv/bin/python scripts/retrieve_demo.py "how does X work"     # one query
    RAG_METHOD=hybrid .venv/bin/python scripts/retrieve_demo.py     # a different ablation
    .venv/bin/python scripts/retrieve_demo.py --queries my.txt      # one question per line

Unlike search_demo.py, which shows each index's raw opinion side by side, this
runs the merge, the reranker and the final selection — exactly what day 4 will
hand to the LLM. It also prints the per-stage latency table, because those
numbers are half of the day 5 results table and they are free to collect now.

The question to answer by eye: **does a text query return a relevant image?**
That is the multimodal claim, and everything after day 3 assumes it holds.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from index import bm25_index, image_index, text_index  # noqa: E402
from retrieve import search as search_mod  # noqa: E402

# Aimed at the self-test corpus (scripts/make_test_pdfs.py) so this script is
# runnable before the real PDFs exist. Roughly a third are figure questions,
# which is the same balance the day 5 eval set needs.
DEFAULT_QUERIES = [
    "how does the system update its internal state",           # text
    "what is photosynthesis and where does it happen",         # text
    "a green leaf diagram showing chloroplast structure",      # figure
    "show me the circuit diagram with a resistor",             # figure
    "bar chart of measured accuracy results",                  # figure (vector art)
    "how are errors propagated backwards through the stages",  # text
    "current is the same through every element in series",     # text
    "what does figure 1 illustrate",                           # figure
    "measured yields by configuration",                        # both
    "each stage transforms the representation",                # text
]


def show(result) -> None:
    print(f"\n{'=' * 78}\nQUERY: {result.query}\n{'=' * 78}")

    stages = " | ".join(
        f"{k} {v * 1000:.0f}ms" for k, v in result.timings.items() if k != "total"
    )
    print(f"  {stages}  ->  TOTAL {result.timings['total'] * 1000:.0f}ms")
    print(f"  candidates: {result.counts}")

    if not result.evidence:
        print("  (nothing retrieved)")
        return

    for i, e in enumerate(result.evidence, start=1):
        kind = "IMG " if e.is_image else "TEXT"
        print(f"\n  [{i}] {kind} {e.citation()}")
        print(f"      {e.preview(width=88)}")
        ranks = e.metadata.get("ranks", {})
        raw = {k: round(v, 3) for k, v in e.metadata.get("raw_scores", {}).items()}
        detail = f"      per-index rank {ranks}  raw {raw}"
        if "rerank_score" in e.metadata:
            detail += f"  rerank {e.metadata['rerank_score']:+.2f}"
        print(detail)

    # Every image the merge saw, with the score the floor is applied to. This
    # is the data for tuning IMAGE_SCORE_FLOOR on the real collection: a
    # relevant figure sitting just under the line, or junk sitting just over
    # it, is visible here and nowhere else.
    img_cands = [e for e in result.candidates if e.is_image][:5]
    if img_cands:
        print("\n      image candidates (CLIP score, floor "
              f"{config.IMAGE_SCORE_FLOOR}):")
        for e in img_cands:
            s = search_mod.clip_score(e)
            print(f"        {'keep' if s >= config.IMAGE_SCORE_FLOOR else 'drop'} "
                  f"{s:.3f}  {e.id}")

    n_img = len(result.images)
    print(f"\n  -> {n_img} image(s) in the final {len(result.evidence)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="one question; omit for the stock set")
    ap.add_argument("--queries", type=Path, help="file with one question per line")
    ap.add_argument("-k", type=int, default=config.TOP_K_FINAL)
    args = ap.parse_args()

    if args.query:
        queries = [" ".join(args.query)]
    elif args.queries:
        queries = [ln.strip() for ln in args.queries.read_text().splitlines()
                   if ln.strip() and not ln.startswith("#")]
    else:
        queries = DEFAULT_QUERIES

    method = config.METHOD
    print(f"method: {method.name}")
    print(f"        indexes {method.enabled_indexes}, rerank={method.use_rerank}")
    print(f"indexed: {bm25_index.count()} bm25, {text_index.count()} dense, "
          f"{image_index.count()} images")
    if text_index.count() == 0 and image_index.count() == 0:
        print("\nNothing indexed. Run: .venv/bin/python scripts/ingest_all.py")
        return 1

    # Everything this method needs, loaded before anything is timed.
    print("\nwarming up ...", flush=True)
    warm = search_mod.warm_up(method)
    print(f"  models loaded in {warm:.1f}s (one-off; not part of query latency)")

    results = []
    for q in queries:
        r = search_mod.search(q, k=args.k, method=method)
        results.append(r)
        show(r)

    # The latency half of the results table.
    print(f"\n{'=' * 78}\nLATENCY over {len(results)} queries "
          f"— {method.name}\n{'=' * 78}")
    stages = [s for s in ("bm25", "dense", "clip", "merge", "rerank", "total")
              if any(s in r.timings for r in results)]
    print(f"  {'stage':<10} {'median':>10} {'max':>10}")
    for s in stages:
        vals = [r.timings[s] * 1000 for r in results if s in r.timings]
        print(f"  {s:<10} {statistics.median(vals):>9.0f}ms {max(vals):>9.0f}ms")

    with_img = sum(1 for r in results if r.images)
    print(f"\n  {with_img}/{len(results)} queries returned at least one image")
    print("\nEyeball check: are those images actually about the question asked?")
    print("If not, day 4 is built on sand — fix retrieval first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
