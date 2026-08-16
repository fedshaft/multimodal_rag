"""Day 4's gate: the whole system, from a typed question to a cited answer.

    .venv/bin/python scripts/ask.py "what is photosynthesis"
    .venv/bin/python scripts/ask.py                     # three demo questions
    .venv/bin/python scripts/ask.py -q "..." --evidence  # show the prompt's evidence
    RAG_METHOD=hybrid .venv/bin/python scripts/ask.py "..."   # a text-only ablation

Needs Ollama running in another terminal:  `ollama serve`

If this prints a sensible answer that cites the right sources, the project is
functional end to end and everything after it is measurement and polish.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from generate import answer as gen  # noqa: E402
from index import bm25_index, image_index, text_index  # noqa: E402
from retrieve import search as search_mod  # noqa: E402

# One text question, one figure question, one of each — the same balance the
# day 5 eval set needs. Aimed at the self-test corpus so this runs today.
DEMO_QUESTIONS = [
    "what is photosynthesis and where does it happen",
    "a green leaf diagram showing chloroplast structure",
    "measured yields by configuration",
]


def show(ans, show_evidence: bool = False) -> None:
    print(f"\n{'=' * 78}\nQ: {ans.question}\n{'=' * 78}\n")
    print(ans.text)

    cited = {e.id for e in gen.cited_evidence(ans)}
    print(f"\n{'-' * 78}\nSOURCES  (* = cited in the answer above)")
    for i, e in enumerate(ans.evidence, start=1):
        mark = "*" if e.id in cited else " "
        kind = "IMG " if e.is_image else "TEXT"
        print(f" {mark}[{i}] {kind} {e.citation():<34} {e.id}")
        if show_evidence:
            print(f"        {e.preview(width=88)}")

    if not cited and ans.evidence:
        # Worth noticing rather than glossing over: either the model ignored the
        # instruction, or nothing retrieved was actually relevant. Day 5 counts
        # this as a citation failure, so surface it here too.
        print("\n  !! the answer cited nothing — check the evidence is relevant")

    missing = [e for e in ans.evidence
               if e.is_image and (not e.image_path or not e.image_path.exists())]
    for e in missing:
        print(f"\n  !! image file missing, not shown to the model: {e.content}")

    stages = " | ".join(f"{k.replace('retrieve.', '')} {v:.1f}s"
                        for k, v in ans.timings.items()
                        if k not in ("total", "retrieve.total"))
    print(f"\n  {stages}")
    print(f"  retrieval {ans.timings.get('retrieve.total', 0):.1f}s + "
          f"generation {ans.timings.get('generate', 0):.1f}s "
          f"= {ans.timings.get('total', 0):.1f}s total")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="the question; omit for the demo set")
    ap.add_argument("-q", "--query", dest="query", help="the question, as a flag")
    ap.add_argument("-k", type=int, default=config.TOP_K_FINAL,
                    help="how many pieces of evidence reach the model")
    ap.add_argument("--evidence", action="store_true",
                    help="also print the evidence text that went into the prompt")
    args = ap.parse_args()

    if args.query:
        questions = [args.query]
    elif args.question:
        questions = [" ".join(args.question)]
    else:
        questions = DEMO_QUESTIONS

    method = config.METHOD
    print(f"method: {method.name}   model: {config.GEN_MODEL}")
    print(f"indexed: {bm25_index.count()} bm25, {text_index.count()} dense, "
          f"{image_index.count()} images")
    if text_index.count() == 0 and image_index.count() == 0:
        print("\nNothing indexed. Run: .venv/bin/python scripts/ingest_all.py")
        return 1

    # Both warm-ups matter, and for the same reason as on days 2 and 3: the
    # encoders load lazily in this process, and Ollama loads the LLM into RAM
    # on its first call. Timing either of those as query cost would put a
    # fabricated number in the results table.
    print("\nwarming up ...", flush=True)
    warm_retrieval = search_mod.warm_up(method)
    try:
        warm_gen = gen.warm_up()
    except gen.GenerationError as exc:
        print(f"\nFAIL: {exc}")
        return 1
    print(f"  encoders {warm_retrieval:.1f}s, {config.GEN_MODEL} {warm_gen:.1f}s "
          "(one-off; not part of answer latency)")

    if len(questions) > 1:
        print(f"\n{len(questions)} questions, roughly a minute each on this CPU.")

    for q in questions:
        try:
            ans = gen.ask(q, k=args.k, method=method)
        except gen.GenerationError as exc:
            print(f"\nFAIL: {exc}")
            return 1
        show(ans, show_evidence=args.evidence)

    print("\nEyeball check: is the answer supported by the sources it cites, and "
          "does it\nsay \"not found\" rather than guessing when the evidence is thin?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
