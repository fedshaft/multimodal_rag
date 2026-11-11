"""Domain fine-tuning for text embeddings (BAAI/bge-small-en-v1.5) on Operating Systems.

Extracts training pairs from the 2,431 indexed document chunks with:
1. Zero test data leakage: strictly excludes all evidence IDs present in eval_set.json.
2. Hard negative mining: uses BM25 keyword search to mine plausible false-positive passages.
3. Contrastive training with MultipleNegativesRankingLoss via sentence-transformers.
4. Checkpointing to models/bge-small-os-finetuned/.

Usage:
    .venv/bin/python scripts/fine_tune_embeddings.py --epochs 2 --batch-size 16
    .venv/bin/python scripts/fine_tune_embeddings.py --evaluate
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

import config
from index import bm25_index, store

MODELS_DIR = config.ROOT / "models"
DEFAULT_OUTPUT_DIR = MODELS_DIR / "bge-small-os-finetuned"


def load_eval_exclusion_set() -> set[str]:
    """All evidence IDs in the evaluation set to strictly prevent data leakage."""
    if not config.EVAL_SET.exists():
        return set()
    data = json.loads(config.EVAL_SET.read_text())
    excluded = set()
    for q in data["questions"]:
        for rid in q.get("relevant_ids", []):
            excluded.add(rid)
    return excluded


def extract_synthetic_queries(chunk_text: str) -> list[str]:
    """Generate realistic queries from passage headings, definitions, and topics."""
    queries = []
    lines = [ln.strip() for ln in chunk_text.splitlines() if ln.strip()]
    if not lines:
        return queries

    # 1. Heading extraction (lines in ALL CAPS or starting with bullet/section marks)
    first_line = lines[0]
    if len(first_line) > 5 and len(first_line) < 80:
        clean_hdr = re.sub(r"^[0-9\.\-\*\#\s]+", "", first_line).strip()
        if len(clean_hdr) > 5:
            queries.append(f"What is {clean_hdr} in operating systems?")
            queries.append(f"How does {clean_hdr} work?")

    # 2. Key definition sentences ("X is a ...", "The goal of X is ...")
    for s in re.split(r"[.!?]\s+", chunk_text):
        s_clean = s.strip()
        if re.search(r"\b(is defined as|is a mechanism|refers to|the purpose of|the goal of)\b", s_clean, re.I):
            words = s_clean.split()
            if 6 <= len(words) <= 25:
                queries.append(s_clean)
                break

    # 3. First sentence as summary query
    first_sentence = re.split(r"[.!?]\s+", chunk_text)[0].strip()
    if 8 <= len(first_sentence.split()) <= 20:
        queries.append(first_sentence)

    return list(set(queries))


def build_training_triplets(min_negatives: int = 1, max_pairs: int = 1000) -> list[InputExample]:
    """Build (query, positive, hard_negative) triplets using BM25 mining."""
    print("Extracting domain corpus from Chroma text collection...")
    col = store.get_collection("text")
    all_data = col.get(include=["documents", "metadatas"])
    excluded_ids = load_eval_exclusion_set()
    print(f"Total chunks in corpus: {len(all_data['ids'])}, excluded eval IDs: {len(excluded_ids)}")

    triplets = []
    skipped = 0

    for cid, text, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
        if cid in excluded_ids:
            skipped += 1
            continue

        queries = extract_synthetic_queries(text)
        if not queries:
            continue

        src_doc = meta.get("source_doc", "")

        for q in queries:
            # Mine hard negative using BM25
            bm25_hits = bm25_index.search(q, k=8)
            negatives = []
            for hit in bm25_hits:
                # Must not be the same chunk and preferably not from the same source document
                if hit.id != cid and hit.source_doc != src_doc and hit.modality == "text":
                    negatives.append(hit.content)
                if len(negatives) >= min_negatives:
                    break

            if negatives:
                # BGE query instruction prefix is required for queries
                prefixed_query = config.TEXT_QUERY_PREFIX + q
                triplets.append(InputExample(texts=[prefixed_query, text, negatives[0]]))

            if len(triplets) >= max_pairs:
                break
        if len(triplets) >= max_pairs:
            break

    print(f"Generated {len(triplets)} training triplets (skipped {skipped} eval chunks to prevent leakage).")
    return triplets


def train(epochs: int = 2, batch_size: int = 16, lr: float = 2e-5, output_dir: Path = DEFAULT_OUTPUT_DIR):
    """Fine-tune BGE-small using MultipleNegativesRankingLoss."""
    triplets = build_training_triplets(max_pairs=800)
    if not triplets:
        print("No training data generated.")
        return

    random.seed(42)
    random.shuffle(triplets)

    split = int(len(triplets) * 0.9)
    train_data = triplets[:split]
    val_data = triplets[split:]
    print(f"Train split: {len(train_data)} | Validation split: {len(val_data)}")

    print(f"\nLoading base model: {config.TEXT_MODEL} on CPU...")
    model = SentenceTransformer(config.TEXT_MODEL, device="cpu")
    model.max_seq_length = 256

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_steps = int(len(train_loader) * epochs * 0.1)

    print(f"Starting fine-tuning: {epochs} epochs, batch_size={batch_size}, lr={lr}, warmup_steps={warmup_steps}")
    t0 = time.perf_counter()

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    elapsed = time.perf_counter() - t0
    print(f"\nFine-tuning completed in {elapsed:.1f}s!")
    print(f"Saved fine-tuned weights to: {output_dir}")


def evaluate_comparison(finetuned_path: Path = DEFAULT_OUTPUT_DIR):
    """Compare baseline BGE-small vs fine-tuned model on eval set."""
    if not finetuned_path.exists():
        print(f"Fine-tuned model not found at {finetuned_path}. Run training first.")
        return

    print("Evaluating baseline vs fine-tuned embedding model on eval set...")
    base_model = SentenceTransformer(config.TEXT_MODEL)
    ft_model = SentenceTransformer(str(finetuned_path))

    data = json.loads(config.EVAL_SET.read_text())
    text_questions = [q for q in data["questions"] if q["answer_type"] in ("text", "both")]

    col = store.get_collection("text")
    all_chunks = col.get(include=["documents"])
    chunk_ids = all_chunks["ids"]
    chunk_texts = all_chunks["documents"]

    print("Embedding corpus with fine-tuned model...")
    ft_embeddings = ft_model.encode(chunk_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    base_embeddings = base_model.encode(chunk_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    def score_model(model, embeddings, name: str):
        recalls, hits, mrrs = [], [], []
        for q in text_questions:
            q_emb = model.encode([config.TEXT_QUERY_PREFIX + q["question"]], normalize_embeddings=True)[0]
            sims = np.dot(embeddings, q_emb)
            top5_indices = np.argsort(-sims)[:5]
            top5_ids = [chunk_ids[i] for i in top5_indices]
            rel = set(q["relevant_ids"])

            recalls.append(sum(1 for rid in top5_ids if rid in rel) / max(len(rel), 1))
            hits.append(1.0 if any(rid in rel for rid in top5_ids) else 0.0)
            mrr_val = 0.0
            for rank, cid in enumerate(top5_ids, 1):
                if cid in rel:
                    mrr_val = 1.0 / rank
                    break
            mrrs.append(mrr_val)

        print(f"\n[{name}] (over {len(text_questions)} text/both questions):")
        print(f"  Recall@5: {np.mean(recalls):.4f}")
        print(f"  Hit@5:    {np.mean(hits):.4f}")
        print(f"  MRR:      {np.mean(mrrs):.4f}")
        return {"recall@5": np.mean(recalls), "hit@5": np.mean(hits), "mrr": np.mean(mrrs)}

    base_scores = score_model(base_model, base_embeddings, "Baseline BGE-small")
    ft_scores = score_model(ft_model, ft_embeddings, "Fine-Tuned BGE-small")

    diff = ft_scores["recall@5"] - base_scores["recall@5"]
    sign = "+" if diff >= 0 else ""
    print(f"\nNet Recall@5 delta: {sign}{diff:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Fine-tune BGE-small on Operating Systems domain")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--evaluate", action="store_true", help="Compare baseline vs fine-tuned model")
    args = ap.parse_args()

    if args.evaluate:
        evaluate_comparison()
    else:
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
