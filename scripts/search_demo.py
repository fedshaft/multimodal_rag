"""Day 2's gate: query each index directly and look at what comes back.

    .venv/bin/python scripts/search_demo.py "how does backpropagation work"

Deliberately *not* the real retrieval pipeline — there is no merging and no
reranking here (that is day 3). This shows each index's raw opinion side by
side, which is what you want when checking whether ingestion actually worked.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from index import bm25_index, image_index, text_index  # noqa: E402

DEFAULT_QUERIES = [
    "what does the architecture diagram show",
    "how does the training process work",
]


def show(title: str, results, k: int, elapsed: float) -> None:
    print(f"\n--- {title}  ({elapsed * 1000:.0f} ms) ---")
    if not results:
        print("  (nothing — is this index built?)")
        return
    for e in results[:k]:
        print("  " + e.preview())
        print(f"  {'':>28} from {e.citation()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="the question to search for")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES

    print(f"indexed: {bm25_index.count()} bm25, {text_index.count()} dense, "
          f"{image_index.count()} images")
    if text_index.count() == 0 and image_index.count() == 0:
        print("\nNothing indexed yet. Run: .venv/bin/python scripts/ingest_all.py")
        return 1

    # Load both encoders before timing anything. Without this the first search
    # reports ~10 s and every later one ~10 ms, because the first call is
    # secretly loading a model. Those numbers go in the results table, so they
    # have to measure search rather than start-up.
    print("warming up encoders ...", flush=True)
    t0 = time.perf_counter()
    text_index.embed_query("warm up")
    image_index.embed_query("warm up")
    print(f"  models loaded in {time.perf_counter() - t0:.1f}s "
          f"(one-off; not part of query latency)")

    for q in queries:
        print("\n" + "=" * 78)
        print(f"QUERY: {q}")
        print("=" * 78)

        for title, fn in (("BM25 (keyword)", bm25_index.search),
                          ("DENSE (meaning)", text_index.search),
                          ("CLIP (images)", image_index.search)):
            t0 = time.perf_counter()
            res = fn(q, k=args.k)
            show(title, res, args.k, time.perf_counter() - t0)

    print("\nCheck by eye: do the images returned actually relate to the question?")
    print("That is the multimodal claim, and day 3 depends on it holding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
