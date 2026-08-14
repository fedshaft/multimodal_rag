"""Build every index from the document collection.

    .venv/bin/python scripts/ingest_all.py

Ingestion is a full rebuild, not an update — re-running it drops the old
collections first, so deleting a PDF and re-ingesting really removes it.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.evidence import Evidence  # noqa: E402
from index import bm25_index, image_index, text_index  # noqa: E402
from ingest.pdf_parser import collect_documents, doc_stem, ingest_path  # noqa: E402


def parse_collection(docs_dir: Path) -> list[Evidence]:
    paths = collect_documents(docs_dir)
    if not paths:
        print(f"No documents found in {docs_dir}")
        return []

    # Two files whose names differ only in punctuation would produce colliding
    # evidence ids, which would silently overwrite each other in the index.
    stems = Counter(doc_stem(p) for p in paths)
    for stem, n in stems.items():
        if n > 1:
            clashing = [p.name for p in paths if doc_stem(p) == stem]
            print(f"  !! name clash on '{stem}': {clashing}")
            print("     Rename one — their evidence ids would overwrite each other.")
            return []

    all_evidence: list[Evidence] = []
    print(f"Parsing {len(paths)} documents from {docs_dir}\n")
    for p in paths:
        t0 = time.perf_counter()
        try:
            ev = ingest_path(p)
        except Exception as exc:
            print(f"  !! {p.name}: {type(exc).__name__}: {exc}")
            continue
        n_text = sum(1 for e in ev if e.modality == "text")
        n_img = sum(1 for e in ev if e.modality == "image")
        all_evidence.extend(ev)
        print(f"  {p.name[:46]:<46} {n_text:>4} chunks {n_img:>4} images "
              f"{time.perf_counter() - t0:>6.1f}s")
    return all_evidence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=config.DOCS_DIR)
    args = ap.parse_args()

    t_start = time.perf_counter()
    evidence = parse_collection(args.docs)
    if not evidence:
        return 1

    n_text = sum(1 for e in evidence if e.modality == "text")
    n_img = sum(1 for e in evidence if e.modality == "image")
    kinds = Counter(e.metadata.get("kind") for e in evidence if e.modality == "image")
    words = sum(e.metadata.get("words", 0) for e in evidence if e.modality == "text")

    print(f"\nparsed in {time.perf_counter() - t_start:.1f}s")
    print(f"  {n_text} text chunks ({words:,} words, "
          f"{words // max(n_text, 1)} avg)")
    print(f"  {n_img} images  {dict(kinds)}")

    print("\nbuilding indexes ...")
    for name, fn in (("bm25 ", bm25_index.build),
                     ("dense", text_index.build),
                     ("clip ", image_index.build)):
        t0 = time.perf_counter()
        n = fn(evidence)
        print(f"  {name}  {n:>5} items  {time.perf_counter() - t0:>6.1f}s")

    print(f"\ntotal {time.perf_counter() - t_start:.1f}s")
    print(f"indexes written to {config.INDEX_DIR}")

    if n_img == 0:
        print("\nWARNING: no images indexed. The multimodal comparison cannot work.")
        print("Run scripts/check_corpus.py to see what the collection actually holds.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
