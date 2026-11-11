"""Fine-tune the Cross-Encoder reranker on Operating Systems domain passage pairs.

Trains cross-encoder/ms-marco-MiniLM-L-6-v2 using Binary Cross Entropy
on (Query, Positive Passage) [label=1.0] and (Query, Hard Negative) [label=0.0].

Usage:
    .venv/bin/python scripts/fine_tune_reranker.py --epochs 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

import config
from index import bm25_index, store
from scripts.fine_tune_embeddings import extract_synthetic_queries, load_eval_exclusion_set

MODELS_DIR = config.ROOT / "models"
DEFAULT_OUTPUT_DIR = MODELS_DIR / "reranker-os-finetuned"
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_reranker_examples(max_pairs: int = 600) -> list[InputExample]:
    col = store.get_collection("text")
    all_data = col.get(include=["documents", "metadatas"])
    excluded_ids = load_eval_exclusion_set()

    examples = []
    print(f"Building reranker training pairs (max={max_pairs})...")

    for cid, text, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
        if cid in excluded_ids:
            continue
        queries = extract_synthetic_queries(text)
        if not queries:
            continue

        src_doc = meta.get("source_doc", "")
        for q in queries:
            # Positive pair
            examples.append(InputExample(texts=[q, text], label=1.0))

            # Hard negative pair via BM25
            hits = bm25_index.search(q, k=5)
            for h in hits:
                if h.id != cid and h.source_doc != src_doc:
                    examples.append(InputExample(texts=[q, h.content], label=0.0))
                    break

            if len(examples) >= max_pairs:
                break
        if len(examples) >= max_pairs:
            break

    print(f"Generated {len(examples)} pairs for reranker training.")
    return examples


def train_reranker(epochs: int = 2, batch_size: int = 16, lr: float = 2e-5, output_dir: Path = DEFAULT_OUTPUT_DIR):
    examples = build_reranker_examples()
    if not examples:
        return

    random.seed(42)
    random.shuffle(examples)

    split = int(len(examples) * 0.9)
    train_data = examples[:split]
    val_data = examples[split:]

    print(f"\nLoading base cross-encoder: {BASE_MODEL}...")
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=512)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    warmup_steps = int(len(train_loader) * epochs * 0.1)

    print(f"Starting reranker fine-tuning: {epochs} epochs, lr={lr}, warmup_steps={warmup_steps}")
    t0 = time.perf_counter()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    model.save(str(output_dir))

    print(f"\nReranker training completed in {time.perf_counter() - t0:.1f}s!")
    print(f"Saved reranker weights to: {output_dir}")


def evaluate_reranker_comparison(finetuned_path: Path = DEFAULT_OUTPUT_DIR):
    """Compare baseline cross-encoder vs fine-tuned cross-encoder on hybrid candidate pool."""
    if not finetuned_path.exists():
        print(f"Fine-tuned model not found at {finetuned_path}. Run training first.")
        return

    print("\nEvaluating baseline vs fine-tuned reranker on eval set...")
    from retrieve.search import search
    import numpy as np

    data = json.loads(config.EVAL_SET.read_text())
    text_questions = [q for q in data["questions"] if q["answer_type"] in ("text", "both")]

    print(f"Loading baseline model: {BASE_MODEL}...")
    base_reranker = CrossEncoder(BASE_MODEL)
    print(f"Loading fine-tuned model: {finetuned_path}...")
    ft_reranker = CrossEncoder(str(finetuned_path))

    hybrid_method = config.METHODS["hybrid"]

    # Pre-retrieve hybrid candidates
    question_candidates = []
    for q in text_questions:
        res = search(q["question"], k=20, method=hybrid_method)
        candidates = res.evidence
        gold = set(q["relevant_ids"])
        question_candidates.append((q["question"], candidates, gold))

    def score_reranker(reranker, name: str):
        recalls, hits, mrrs = [], [], []
        for query_text, candidates, gold in question_candidates:
            if not candidates:
                continue
            pairs = [[query_text, c.content] for c in candidates]
            scores = reranker.predict(pairs)
            ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
            top5 = ranked[:5]
            top5_ids = [c.id for c in top5]

            hits.append(1.0 if any(cid in gold for cid in top5_ids) else 0.0)
            retrieved_gold = len(set(top5_ids) & gold)
            recalls.append(retrieved_gold / max(len(gold), 1))

            rank = next((idx + 1 for idx, cid in enumerate(top5_ids) if cid in gold), None)
            mrrs.append(1.0 / rank if rank else 0.0)

        return {
            "name": name,
            "hit@5": float(np.mean(hits)),
            "recall@5": float(np.mean(recalls)),
            "mrr@5": float(np.mean(mrrs)),
        }

    base_res = score_reranker(base_reranker, "Baseline MiniLM-L-6-v2")
    ft_res = score_reranker(ft_reranker, "Fine-tuned MiniLM-L-6-v2 (OS)")

    print("\n" + "=" * 62)
    print("          RERANKER EVALUATION COMPARISON (@5)          ")
    print("=" * 62)
    print(f"{'Model':<34} {'Hit@5':>8} {'Recall@5':>10} {'MRR@5':>8}")
    print("-" * 62)
    for res in [base_res, ft_res]:
        print(f"{res['name']:<34} {res['hit@5']:>8.4f} {res['recall@5']:>10.4f} {res['mrr@5']:>8.4f}")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune cross-encoder reranker")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval-only", action="store_true", help="Only run evaluation comparison")
    args = ap.parse_args()

    if not args.eval_only:
        train_reranker(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    evaluate_reranker_comparison()


if __name__ == "__main__":
    main()
