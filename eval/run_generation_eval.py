"""Answer-quality pass over all 40 eval questions, full system.

Same metrics as eval/run_eval.py's generation_pass (citation recall/precision,
refusal on out-of-collection questions), but also keeps each answer's text and
the answer key so correctness can be graded afterwards. Writes incrementally to
eval/results/generation.json so a crash loses nothing. Does not touch the
retrieval results files.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from eval import metrics  # noqa: E402
from generate import answer as gen  # noqa: E402
from retrieve import search as search_mod  # noqa: E402

OUT = ROOT / "eval" / "results" / "generation.json"

questions = json.loads(config.EVAL_SET.read_text())["questions"]
method = config.METHODS["full"]
search_mod.warm_up(method)
gen.warm_up()

rows = []
for n, q in enumerate(questions, 1):
    t0 = time.perf_counter()
    try:
        ans = gen.ask(q["question"], method=method)
    except Exception as exc:
        rows.append({"id": q["id"], "error": f"{type(exc).__name__}: {exc}"})
        print(f"{q['id']} ERROR {exc}", flush=True)
        continue
    cited = [e.id for e in gen.cited_evidence(ans)]
    retrieved = [e.id for e in ans.evidence]
    row = {
        "id": q["id"], "answer_type": q["answer_type"], "question": q["question"],
        "key": q.get("note", ""), "answer": ans.text,
        "cited": cited, "retrieved": retrieved,
        "seconds": round(time.perf_counter() - t0, 1),
        "gen_seconds": round(ans.timings.get("generate", 0.0), 1),
    }
    if q["answer_type"] == "none":
        row["abstained"] = metrics.abstained(ans.text)
    else:
        rel = set(q["relevant_ids"])
        hit = len(rel & set(cited))
        row["citation_recall"] = hit / len(rel) if rel else 0.0
        row["citation_precision"] = hit / len(cited) if cited else 0.0
        row["gold_retrieved"] = bool(rel & set(retrieved))
    rows.append(row)
    OUT.write_text(json.dumps({"model": config.GEN_MODEL, "rows": rows}, indent=2))
    print(f"{n}/{len(questions)} {q['id']} {row['seconds']}s cited={cited}", flush=True)

ok = [r for r in rows if "citation_recall" in r]
none = [r for r in rows if "abstained" in r]
summary = {
    "model": config.GEN_MODEL,
    "answered": len(ok), "errors": sum("error" in r for r in rows),
    "citation_recall": sum(r["citation_recall"] for r in ok) / len(ok) if ok else None,
    "citation_precision": sum(r["citation_precision"] for r in ok) / len(ok) if ok else None,
    "refusal_rate": sum(r["abstained"] for r in none) / len(none) if none else None,
    "mean_seconds": sum(r["seconds"] for r in rows if "seconds" in r) / max(1, len(ok) + len(none)),
}
OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
print("SUMMARY", json.dumps(summary), flush=True)
