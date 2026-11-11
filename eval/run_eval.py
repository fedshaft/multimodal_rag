"""Run every configuration over the eval set and produce the results table.

Five methods (config.METHODS) x the labelled questions, one shared answer key,
so any difference between rows is caused by the retrieval strategy and nothing
else. That is what makes this an experiment rather than five demos.

    .venv/bin/python eval/run_eval.py                 # retrieval only, fast
    .venv/bin/python eval/run_eval.py --methods bm25,full
    .venv/bin/python eval/run_eval.py --with-generation   # adds citation
                                                          # accuracy; SLOW

Writes eval/results/results.json, results_table.csv, by_type.csv.

Retrieval over 40 questions is a few minutes. Generation is roughly a minute per
question on the development PC, so --with-generation is an hour-plus and is off
by default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from eval import metrics  # noqa: E402
from retrieve import search as search_mod  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
KS = (1, 5, 10)


def load_questions() -> list[dict]:
    data = json.loads(config.EVAL_SET.read_text())
    return data["questions"]


def run_method(key: str, questions: list[dict], k: int,
               with_generation: bool) -> dict:
    """One row of the results table, plus everything needed to break it down."""
    method = config.METHODS[key]
    print(f"\n=== {method.name} ===")

    scored: list[dict] = []       # retrieval metrics, answerable questions only
    per_q: list[dict] = []        # one record per question, for the failure analysis
    none_results: list[dict] = []
    stage_times: dict[str, list[float]] = defaultdict(list)

    for n, q in enumerate(questions, 1):
        r = search_mod.search(q["question"], k=k, method=method)
        got = [e.id for e in r.evidence]
        shortlist = [e.id for e in r.candidates]

        for stage, t in r.timings.items():
            stage_times[stage].append(t)

        rec = {
            "id": q["id"], "answer_type": q["answer_type"],
            "question": q["question"],
            "retrieved": got, "relevant": q["relevant_ids"],
        }

        if q["answer_type"] == "none":
            # No relevant ids exist, so every retrieval metric is undefined.
            # These questions test refusal, which lives in the generated answer,
            # not the ranking — scoring them here would quietly average a
            # meaningless zero into the retrieval numbers.
            rec["note"] = "out-of-collection; retrieval metrics not applicable"
            none_results.append(rec)
            per_q.append(rec)
            continue

        m = metrics.evaluate_one(got, q["relevant_ids"], ks=KS)
        # Recall over the pre-cut shortlist separates "never retrieved" from
        # "retrieved but lost the last slot" — different problems, different fixes.
        m["recall@shortlist"] = metrics.recall_at_k(
            shortlist, q["relevant_ids"], len(shortlist) or 1)
        rec.update(m)
        scored.append(m)
        per_q.append(rec)

        if n % 10 == 0 or n == len(questions):
            print(f"  {n}/{len(questions)}")

    agg = metrics.aggregate(scored)
    agg["latency_ms"] = 1000 * sum(stage_times["total"]) / max(len(stage_times["total"]), 1)

    by_type: dict[str, dict] = {}
    for t in ("text", "figure", "both"):
        sub = [r for r in per_q if r["answer_type"] == t and "mrr" in r]
        if sub:
            by_type[t] = metrics.aggregate(
                [{k: r[k] for k in sub[0] if isinstance(r[k], float)} for r in sub])
            by_type[t]["n"] = len(sub)

    stages = {s: 1000 * sum(v) / len(v) for s, v in stage_times.items()}

    print(f"  recall@5 {agg.get('recall@5', 0):.3f}   "
          f"MRR {agg.get('mrr', 0):.3f}   "
          f"nDCG@10 {agg.get('ndcg@10', 0):.3f}   "
          f"{agg['latency_ms']:.0f} ms/query")

    return {
        "key": key, "name": method.name, "note": method.note,
        "overall": agg, "by_type": by_type, "stages_ms": stages,
        "per_question": per_q, "none_questions": none_results,
        "n_scored": len(scored),
    }


def generation_pass(questions: list[dict], k: int) -> dict:
    """Citation accuracy and refusal behaviour, on the full system only.

    Retrieval finding the evidence and the model *citing* it are two different
    claims. Precision matters as much as recall here: day 4 saw the model cite
    all five retrieved items at once, which scores perfect citation recall while
    telling the reader nothing.
    """
    from generate import answer as gen

    print("\n=== generation pass (full system) ===")
    rows, refusals = [], []
    for n, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        try:
            ans = gen.ask(q["question"], k=k)
        except Exception as exc:
            print(f"  !! {q['id']}: {type(exc).__name__}: {exc}")
            continue
        dt = time.perf_counter() - t0

        text = ans.text
        cited_ids = [e.id for e in gen.cited_evidence(ans)]

        if q["answer_type"] == "none":
            ok = metrics.abstained(text)
            refusals.append({"id": q["id"], "abstained": ok, "answer": text[:300]})
            print(f"  {q['id']}: {'REFUSED (correct)' if ok else 'ANSWERED (hallucination)'}")
        else:
            rel = set(q["relevant_ids"])
            hit = len(rel & set(cited_ids))
            rows.append({
                "id": q["id"], "answer_type": q["answer_type"],
                "n_cited": len(cited_ids),
                "citation_recall": hit / len(rel) if rel else 0.0,
                "citation_precision": hit / len(cited_ids) if cited_ids else 0.0,
                "seconds": dt,
            })
        if n % 5 == 0:
            print(f"  {n}/{len(questions)}")

    out: dict = {"per_question": rows, "refusals": refusals}
    if rows:
        df = pd.DataFrame(rows)
        out["citation_recall"] = float(df["citation_recall"].mean())
        out["citation_precision"] = float(df["citation_precision"].mean())
        out["mean_cited"] = float(df["n_cited"].mean())
        out["seconds_per_answer"] = float(df["seconds"].mean())
    if refusals:
        out["refusal_rate"] = sum(r["abstained"] for r in refusals) / len(refusals)
    return out


def write_tables(results: list[dict], gen: dict | None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame([{
        "Method": r["name"],
        "Recall@5": round(r["overall"].get("recall@5", 0), 3),
        "Hit@5": round(r["overall"].get("hit@5", 0), 3),
        "MRR": round(r["overall"].get("mrr", 0), 3),
        "nDCG@10": round(r["overall"].get("ndcg@10", 0), 3),
        "Latency (ms)": round(r["overall"]["latency_ms"], 1),
    } for r in results])
    table.to_csv(RESULTS / "results_table.csv", index=False)

    rows = []
    for r in results:
        for t, m in r["by_type"].items():
            rows.append({
                "Method": r["name"], "Question type": t, "n": int(m["n"]),
                "Recall@5": round(m.get("recall@5", 0), 3),
                "MRR": round(m.get("mrr", 0), 3),
                "nDCG@10": round(m.get("ndcg@10", 0), 3),
            })
    by_type = pd.DataFrame(rows)
    by_type.to_csv(RESULTS / "by_type.csv", index=False)

    payload = {"results": results}
    if gen:
        payload["generation"] = gen
    (RESULTS / "results.json").write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 78)
    print("RESULTS TABLE")
    print("=" * 78)
    print(table.to_string(index=False))

    print("\n" + "=" * 78)
    print("BY QUESTION TYPE — the headline result")
    print("=" * 78)
    if not by_type.empty:
        pivot = by_type.pivot(index="Method", columns="Question type",
                              values="Recall@5")
        cols = [c for c in ("text", "figure", "both") if c in pivot.columns]
        print(pivot[cols].to_string())
        print("\n(Recall@5. The `figure` column is the one the project is about:")
        print(" text-only methods should collapse there and the multimodal ones")
        print(" should not.)")

    if gen:
        print("\n" + "=" * 78)
        print("GENERATION")
        print("=" * 78)
        if "citation_recall" in gen:
            print(f"  citation recall    {gen['citation_recall']:.3f}")
            print(f"  citation precision {gen['citation_precision']:.3f}"
                  f"   (mean {gen['mean_cited']:.1f} items cited)")
            print(f"  seconds per answer {gen['seconds_per_answer']:.1f}")
        if "refusal_rate" in gen:
            print(f"  refusal rate on out-of-collection questions: "
                  f"{gen['refusal_rate']:.0%} "
                  f"({sum(r['abstained'] for r in gen['refusals'])}"
                  f"/{len(gen['refusals'])})")

    print(f"\nwritten to {RESULTS}/results_table.csv, by_type.csv, results.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(config.METHODS),
                    help="comma-separated subset of config.METHODS")
    ap.add_argument("--k", type=int, default=config.TOP_K_FINAL)
    ap.add_argument("--with-generation", action="store_true",
                    help="also run the generation pass (slow: ~1 min/question)")
    args = ap.parse_args()

    questions = load_questions()
    answerable = [q for q in questions if q["answer_type"] != "none"]
    print(f"eval set: {len(questions)} questions "
          f"({len(answerable)} answerable, "
          f"{len(questions) - len(answerable)} out-of-collection)")

    if "DRAFT" in json.loads(config.EVAL_SET.read_text())["_about"].get("status", ""):
        print("\n  !! eval_set.json is still marked DRAFT — the answer keys have")
        print("     not been verified by a human. These numbers are provisional;")
        print("     see eval/eval_set_review.md before quoting any of them.\n")

    keys = [k.strip() for k in args.methods.split(",") if k.strip()]
    for k in keys:
        if k not in config.METHODS:
            print(f"unknown method {k!r}; known: {list(config.METHODS)}")
            return 1

    search_mod.warm_up()

    results = [run_method(k, questions, args.k, args.with_generation)
               for k in keys]
    gen = generation_pass(questions, args.k) if args.with_generation else None
    write_tables(results, gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
