"""Fast hyperparameter tuning for the Multimodal RAG retrieval pipeline.

Pre-fetches candidate rankings once for the evaluation set, then performs
an exhaustive grid search over:
- RRF_K (fusion constant)
- Modality weights (BM25 vs Dense vs CLIP)
- IMAGE_SCORE_FLOOR (CLIP cutoff)
- MAX_IMAGES_FINAL (visual lane capacity)
- ADAPTIVE_SLOTS & VISUAL_FLOOR_DELTA (query-intent adaptive allocation)

Run:
    .venv/bin/python scripts/tune_retrieval.py
    .venv/bin/python scripts/tune_retrieval.py --apply
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.evidence import Evidence
from eval import metrics
from index import bm25_index, image_index, text_index
from retrieve import rerank as rerank_mod
from retrieve import search as search_mod

RESULTS_FILE = config.RESULTS_DIR / "tuning_results.json"


def load_eval_questions() -> list[dict]:
    data = json.loads(config.EVAL_SET.read_text())
    return [q for q in data["questions"] if q.get("answer_type") != "none"]


def prefetch_candidates(questions: list[dict], top_k: int = 50) -> dict[str, dict]:
    """Fetch index candidates and pre-score text with the cross-encoder once."""
    print(f"Pre-fetching candidates for {len(questions)} evaluation questions...")
    print("Warming up encoders...")
    search_mod.warm_up(config.METHODS["full"])

    cached = {}
    t0 = time.perf_counter()

    for idx, q in enumerate(questions, 1):
        query = q["question"]
        qid = q["id"]

        rankings = {
            "bm25": bm25_index.search(query, k=top_k),
            "dense": text_index.search(query, k=top_k),
            "clip": image_index.search(query, k=top_k),
        }

        # Union of text candidates for pre-scoring with cross-encoder
        all_text = {}
        for ev_list in (rankings["bm25"], rankings["dense"]):
            for e in ev_list:
                all_text[e.id] = e

        pairs = [(query, e.content) for e in all_text.values()]
        if pairs:
            scores = rerank_mod.get_model().predict(
                pairs, batch_size=16, show_progress_bar=False
            )
            text_scores = {e.id: float(s) for e, s in zip(all_text.values(), scores)}
        else:
            text_scores = {}

        cached[qid] = {
            "question": q,
            "rankings": rankings,
            "text_scores": text_scores,
        }

        if idx % 10 == 0 or idx == len(questions):
            print(f"  [{idx}/{len(questions)}] pre-fetched")

    print(f"Pre-fetching complete in {time.perf_counter() - t0:.1f}s\n")
    return cached


def evaluate_combo(cached: dict[str, dict],
                   k_rrf: int,
                   weights: dict[str, float],
                   floor: float,
                   max_images: int,
                   adaptive: bool,
                   visual_delta: float,
                   use_rerank: bool = True,
                   top_k_final: int = 5) -> dict:
    """Evaluate one parameter combination across all questions."""
    scored = []
    by_type_records = defaultdict(list)

    for qid, entry in cached.items():
        q = entry["question"]
        query = q["question"]
        rankings = entry["rankings"]
        text_scores = entry["text_scores"]
        relevant = q["relevant_ids"]
        atype = q["answer_type"]

        # 1. RRF Merge with weights
        merged = search_mod.rrf_merge(rankings, k=k_rrf, weights=weights)

        # 2. Cross-encoder rerank if enabled
        if use_rerank and text_scores:
            head = merged[:config.RERANK_TOP_N]
            tail = merged[config.RERANK_TOP_N:]
            texts = [e for e in head if not e.is_image]
            images = [e for e in head if e.is_image]

            rescored_texts = []
            for e in texts:
                s = text_scores.get(e.id, e.score)
                rescored_texts.append(
                    search_mod.replace(e, score=s, metadata={**e.metadata, "rerank_score": s})
                )
            rescored_texts.sort(key=lambda e: -e.score)
            merged = rescored_texts + images + tail

        # 3. Select final with parameters
        is_visual = search_mod.has_visual_intent(query) if adaptive else False
        eff_floor = max(floor - visual_delta, 0.15) if is_visual else (floor + 0.02 if adaptive else floor)
        eff_max_img = min(max_images + 1, max(top_k_final - 1, 1)) if is_visual else (min(max_images, 1) if adaptive else max_images)

        texts = [e for e in merged if not e.is_image]
        images = [e for e in merged if e.is_image and search_mod.clip_score(e) >= eff_floor]

        n_img = min(eff_max_img, len(images), max(top_k_final - 1, 0))
        chosen_images = images[:n_img]
        chosen_texts = texts[:top_k_final - n_img]

        out = []
        if is_visual and chosen_images:
            out.append(chosen_images[0])
            rem_img = chosen_images[1:]
            rem_txt = chosen_texts
            for i in range(max(len(rem_txt), len(rem_img))):
                if i < len(rem_txt):
                    out.append(rem_txt[i])
                if i < len(rem_img):
                    out.append(rem_img[i])
        else:
            for i in range(max(len(chosen_texts), len(chosen_images))):
                if i < len(chosen_texts):
                    out.append(chosen_texts[i])
                if i < len(chosen_images):
                    out.append(chosen_images[i])
        final_evidence = out[:top_k_final]
        got_ids = [e.id for e in final_evidence]

        m = metrics.evaluate_one(got_ids, relevant, ks=(1, 5, 10))
        scored.append(m)
        by_type_records[atype].append(m)

    agg = metrics.aggregate(scored)
    by_type = {t: metrics.aggregate(records) for t, records in by_type_records.items()}

    return {
        "params": {
            "k_rrf": k_rrf,
            "weights": weights,
            "floor": floor,
            "max_images": max_images,
            "adaptive": adaptive,
            "visual_delta": visual_delta,
        },
        "overall": agg,
        "by_type": by_type,
    }


def main():
    ap = argparse.ArgumentParser(description="Tune retrieval pipeline hyperparameters")
    ap.add_argument("--apply", action="store_true", help="Apply best settings into config.py")
    args = ap.parse_args()

    questions = load_eval_questions()
    cached = prefetch_candidates(questions)

    # 1. Baseline evaluation
    print("Evaluating Baseline Configuration...")
    baseline = evaluate_combo(
        cached=cached,
        k_rrf=60,
        weights={"bm25": 1.0, "dense": 1.0, "clip": 1.0},
        floor=0.25,
        max_images=2,
        adaptive=False,
        visual_delta=0.04,
        use_rerank=True,
    )

    print(f"  Baseline Overall Recall@5: {baseline['overall']['recall@5']:.4f} | "
          f"Hit@5: {baseline['overall']['hit@5']:.4f} | "
          f"MRR: {baseline['overall']['mrr']:.4f} | "
          f"nDCG@10: {baseline['overall']['ndcg@10']:.4f}")
    for t in ("text", "figure", "both"):
        if t in baseline["by_type"]:
            sub = baseline["by_type"][t]
            print(f"    - {t:<6} Recall@5: {sub['recall@5']:.4f} | Hit@5: {sub['hit@5']:.4f} | MRR: {sub['mrr']:.4f}")

    # 2. Grid Search
    print("\nRunning Parameter Grid Search...")
    grid = {
        "k_rrf": [40, 60, 80],
        "weight_config": [
            {"bm25": 1.0, "dense": 1.0, "clip": 1.0},
            {"bm25": 0.8, "dense": 1.2, "clip": 1.0},
            {"bm25": 0.9, "dense": 1.1, "clip": 1.2},
            {"bm25": 1.0, "dense": 1.2, "clip": 1.3},
        ],
        "floor": [0.20, 0.22, 0.24, 0.25, 0.26, 0.28],
        "max_images": [2, 3],
        "adaptive": [True, False],
        "visual_delta": [0.03, 0.04, 0.05],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Total hyperparameter trials: {len(combos)}")

    best_result = baseline
    best_score = baseline["overall"]["recall@5"] + baseline["overall"]["hit@5"] + baseline["overall"]["mrr"]
    results = []

    t_start = time.perf_counter()
    for i, c in enumerate(combos, 1):
        params = dict(zip(keys, c))
        res = evaluate_combo(
            cached=cached,
            k_rrf=params["k_rrf"],
            weights=params["weight_config"],
            floor=params["floor"],
            max_images=params["max_images"],
            adaptive=params["adaptive"],
            visual_delta=params["visual_delta"],
            use_rerank=True,
        )
        results.append(res)

        score = res["overall"]["recall@5"] + res["overall"]["hit@5"] + res["overall"]["mrr"]
        # Prioritize figure recall improvement without dropping text recall
        text_rec = res["by_type"]["text"]["recall@5"]
        base_text_rec = baseline["by_type"]["text"]["recall@5"]

        if score > best_score and text_rec >= (base_text_rec - 0.02):
            best_score = score
            best_result = res

        if i % 100 == 0 or i == len(combos):
            print(f"  [{i}/{len(combos)}] trials completed (best Recall@5 so far: {best_result['overall']['recall@5']:.4f})")

    duration = time.perf_counter() - t_start
    print(f"\nGrid search completed in {duration:.1f}s")

    # 3. Report Results
    print("\n" + "=" * 70)
    print("TUNING RESULTS SUMMARY")
    print("=" * 70)
    b_ov = baseline["overall"]
    t_ov = best_result["overall"]
    p = best_result["params"]

    print("\nOptimal Hyperparameters:")
    print(f"  RRF_K:                {p['k_rrf']}")
    print(f"  Weights:              {p['weights']}")
    print(f"  IMAGE_SCORE_FLOOR:    {p['floor']}")
    print(f"  MAX_IMAGES_FINAL:     {p['max_images']}")
    print(f"  ADAPTIVE_SLOTS:       {p['adaptive']}")
    print(f"  VISUAL_FLOOR_DELTA:   {p['visual_delta']}")

    print("\nOverall Comparison:")
    print(f"  Metric       Baseline    Tuned       Delta")
    print(f"  ------------------------------------------")
    for m in ("recall@5", "hit@5", "mrr", "ndcg@10"):
        diff = t_ov[m] - b_ov[m]
        sign = "+" if diff >= 0 else ""
        print(f"  {m:<12} {b_ov[m]:.4f}      {t_ov[m]:.4f}      {sign}{diff:.4f}")

    print("\nBy-Question-Type Recall@5:")
    print(f"  Type         Baseline    Tuned       Delta")
    print(f"  ------------------------------------------")
    for t in ("text", "figure", "both"):
        b_t = baseline["by_type"][t]["recall@5"]
        t_t = best_result["by_type"][t]["recall@5"]
        diff = t_t - b_t
        sign = "+" if diff >= 0 else ""
        print(f"  {t:<12} {b_t:.4f}      {t_t:.4f}      {sign}{diff:.4f}")

    print("\nBy-Question-Type Hit@5:")
    print(f"  Type         Baseline    Tuned       Delta")
    print(f"  ------------------------------------------")
    for t in ("text", "figure", "both"):
        b_t = baseline["by_type"][t]["hit@5"]
        t_t = best_result["by_type"][t]["hit@5"]
        diff = t_t - b_t
        sign = "+" if diff >= 0 else ""
        print(f"  {t:<12} {b_t:.4f}      {t_t:.4f}      {sign}{diff:.4f}")

    # Save to file
    out_data = {
        "baseline": baseline,
        "best": best_result,
        "grid_size": len(combos),
        "duration_seconds": round(duration, 2),
    }
    RESULTS_FILE.write_text(json.dumps(out_data, indent=2))
    print(f"\nTuning results written to {RESULTS_FILE}")

    if args.apply:
        print("\nApplying optimal parameters to config.py...")
        cfg_text = Path("config.py").read_text()
        cfg_text = cfg_text.replace(
            f"IMAGE_SCORE_FLOOR = {config.IMAGE_SCORE_FLOOR}",
            f"IMAGE_SCORE_FLOOR = {p['floor']}",
        )
        cfg_text = cfg_text.replace(
            f"RRF_K = {config.RRF_K}",
            f"RRF_K = {p['k_rrf']}",
        )
        cfg_text = cfg_text.replace(
            f"MAX_IMAGES_FINAL = {config.MAX_IMAGES_FINAL}",
            f"MAX_IMAGES_FINAL = {p['max_images']}",
        )
        cfg_text = cfg_text.replace(
            f"VISUAL_FLOOR_DELTA: float = float(os.environ.get(\"RAG_VISUAL_FLOOR_DELTA\", \"{config.VISUAL_FLOOR_DELTA}\"))",
            f"VISUAL_FLOOR_DELTA: float = float(os.environ.get(\"RAG_VISUAL_FLOOR_DELTA\", \"{p['visual_delta']}\"))",
        )
        cfg_text = cfg_text.replace(
            "ADAPTIVE_IMAGE_SLOTS: bool = bool(int(os.environ.get(\"RAG_ADAPTIVE_SLOTS\", \"0\")))",
            f"ADAPTIVE_IMAGE_SLOTS: bool = bool(int(os.environ.get(\"RAG_ADAPTIVE_SLOTS\", \"{'1' if p['adaptive'] else '0'}\")))",
        )
        Path("config.py").write_text(cfg_text)
        print("config.py updated with tuned parameters!")


if __name__ == "__main__":
    main()
