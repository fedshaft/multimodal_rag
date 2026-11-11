"""Check the eval set against the real index before anything is measured with it.

A relevant_id that does not exist scores zero for every configuration, silently
and identically, so a typo looks exactly like a hard question. Run this after any
edit to eval_set.json, and after any re-ingestion that could change ids.

    .venv/bin/python eval/validate_eval_set.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL = Path(__file__).resolve().parent / "eval_set.json"
CACHE = Path(__file__).resolve().parent / "results" / "cache"

VALID_TYPES = {"text", "figure", "both", "none"}


def known_ids() -> tuple[set[str], set[str]]:
    """(text ids, image ids) actually produced by ingestion."""
    chunks = pd.read_csv(CACHE / "chunks.csv")
    images = pd.read_csv(CACHE / "images.csv")
    return set(chunks["id"]), set(images["id"])


def main() -> int:
    data = json.loads(EVAL.read_text())
    qs = data["questions"]
    text_ids, image_ids = known_ids()
    all_ids = text_ids | image_ids

    problems: list[str] = []
    seen_q: set[str] = set()
    seen_id: set[str] = set()

    for q in qs:
        qid = q.get("id", "?")

        if not q.get("question", "").strip():
            problems.append(f"{qid}: empty question")
        if q["question"].strip().lower() in seen_q:
            problems.append(f"{qid}: duplicate question text")
        seen_q.add(q["question"].strip().lower())

        atype = q.get("answer_type")
        if atype not in VALID_TYPES:
            problems.append(f"{qid}: answer_type {atype!r} not in {sorted(VALID_TYPES)}")

        ids = q.get("relevant_ids", [])

        if atype == "none":
            if ids:
                problems.append(f"{qid}: answer_type 'none' must have an empty "
                                f"relevant_ids, got {ids}")
            continue

        if not ids:
            problems.append(f"{qid}: no relevant_ids")

        for i in ids:
            if i not in all_ids:
                problems.append(f"{qid}: id {i!r} does not exist in the index")
            seen_id.add(i)

        # The tag has to match what the ids actually are, or the headline
        # by-question-type breakdown is measuring the wrong thing.
        has_text = any(i in text_ids for i in ids)
        has_image = any(i in image_ids for i in ids)
        if atype == "text" and has_image:
            problems.append(f"{qid}: tagged 'text' but cites an image id")
        if atype == "figure" and has_text:
            problems.append(f"{qid}: tagged 'figure' but cites a text chunk id")
        if atype == "both" and not (has_text and has_image):
            problems.append(f"{qid}: tagged 'both' but cites only "
                            f"{'text' if has_text else 'images'}")

    print(f"eval set: {len(qs)} questions, {len(seen_id)} distinct evidence ids")
    counts = Counter(q.get("answer_type") for q in qs)
    for t in ("text", "figure", "both", "none"):
        n = counts.get(t, 0)
        print(f"  {t:<7} {n:>3}  ({100 * n / len(qs):.0f}%)")

    figureish = counts.get("figure", 0) + counts.get("both", 0)
    print(f"\n  questions needing an image: {figureish} "
          f"({100 * figureish / len(qs):.0f}%) — target is roughly a third or more")

    docs = Counter(i.rsplit("_chunk_", 1)[0].rsplit("_img_", 1)[0].rsplit("_page_", 1)[0]
                   for i in seen_id)
    print(f"  drawn from {len(docs)} distinct documents")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nAll ids exist and every tag matches the evidence it cites.")
    print("This checks consistency, NOT correctness — whether each id really")
    print("answers its question is a human judgement. See eval_set_review.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
