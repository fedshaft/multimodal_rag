"""Retrieval metrics, in numpy, with the arithmetic checked by hand.

Every number in the results table comes out of this file, so a subtle error here
is invisible and contaminates everything — the same failure mode as a wrong
answer key. `self_test()` therefore checks each metric against a worked example
computed by hand, and `python eval/metrics.py` runs it.

Relevance is binary: an evidence id either answers the question or it does not.
There is no graded relevance in the eval set, so nDCG uses gains of 1 and 0.

    .venv/bin/python eval/metrics.py
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def _first_hit_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> int | None:
    """1-based rank of the first relevant item, or None."""
    rel = set(relevant)
    for i, rid in enumerate(retrieved, start=1):
        if rid in rel:
            return i
    return None


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the relevant evidence that appears in the top k.

    Not the same as hit rate: a question with two relevant ids of which one is
    found scores 0.5 here and 1.0 on hit_at_k. Both are reported — recall is the
    honest measure when a question needs several pieces of evidence, hit rate is
    what a user actually feels ("was the answer in there at all?").
    """
    rel = set(relevant)
    if not rel:
        raise ValueError("recall is undefined with no relevant items")
    found = sum(1 for rid in retrieved[:k] if rid in rel)
    return found / len(rel)


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant item is in the top k."""
    rel = set(relevant)
    if not rel:
        raise ValueError("hit rate is undefined with no relevant items")
    return 1.0 if any(rid in rel for rid in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant item; 0 if none was retrieved.

    Averaged over questions this is MRR. It rewards putting the answer first
    rather than merely somewhere in the list, which recall cannot see.
    """
    rank = _first_hit_rank(retrieved, relevant)
    return 1.0 / rank if rank else 0.0


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain, log2 discount, 1-based positions.

        DCG = sum_i  gain_i / log2(i + 1)

    Position 1 is undiscounted (log2(2) = 1), position 2 is divided by
    log2(3) ~ 1.585, and so on.
    """
    return float(sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1)))


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """DCG of the actual ranking over DCG of the best possible ranking.

    The ideal ranking puts every relevant item first, so IDCG uses
    min(len(relevant), k) ones. Capping at k matters: without it a question with
    more relevant items than k could never score 1.0.
    """
    rel = set(relevant)
    if not rel:
        raise ValueError("nDCG is undefined with no relevant items")
    gains = [1.0 if rid in rel else 0.0 for rid in retrieved[:k]]
    ideal = [1.0] * min(len(rel), k)
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg else 0.0


def evaluate_one(retrieved: Sequence[str], relevant: Iterable[str],
                 ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """Every metric for one question."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
    out["mrr"] = reciprocal_rank(retrieved, relevant)
    out["ndcg@10"] = ndcg_at_k(retrieved, relevant, 10)
    return out


def aggregate(per_question: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across questions.

    A plain unweighted mean over questions ("macro" averaging): every question
    counts once regardless of how many relevant ids it has, so a question with
    three pieces of evidence cannot outvote one with a single piece.
    """
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {k: float(np.mean([q[k] for q in per_question])) for k in keys}


# ---------------------------------------------------------------- abstention

def abstained(answer_text: str) -> bool:
    """Did the generated answer decline to answer?

    Used only for the `none` questions, where the correct behaviour is to say
    the documents do not cover it. Deliberately generous in what it accepts as a
    refusal — the interesting failure is a model that answers confidently from
    world knowledge, and a strict matcher would score a clumsy refusal as one of
    those.
    """
    if not answer_text:
        return True
    t = answer_text.lower()
    phrases = (
        "not found in the documents", "not found in the provided",
        "do not contain", "does not contain", "doesn't contain",
        "no information", "not covered", "cannot answer", "can't answer",
        "unable to answer", "not mentioned", "not discussed",
        "not in the documents", "no relevant", "insufficient information",
    )
    return any(p in t for p in phrases)


# ---------------------------------------------------------------- self-test

def self_test() -> int:
    """Check every metric against arithmetic done by hand.

    Worked example — relevant = {A, B}, retrieved = [X, A, Y, B, Z]:

        recall@5 = 2/2                         = 1.0
        recall@1 = 0/2                         = 0.0
        hit@5                                  = 1.0
        hit@1                                  = 0.0
        MRR      = 1/2 (A is at rank 2)        = 0.5
        DCG@5    = 1/log2(3) + 1/log2(5)
                 = 0.630930 + 0.430677         = 1.061607
        IDCG@5   = 1/log2(2) + 1/log2(3)
                 = 1.0 + 0.630930              = 1.630930
        nDCG@5   = 1.061607 / 1.630930         = 0.650920
    """
    failures: list[str] = []

    def check(name: str, got: float, want: float, tol: float = 1e-6) -> None:
        if abs(got - want) > tol:
            failures.append(f"{name}: got {got!r}, expected {want!r}")

    ret = ["X", "A", "Y", "B", "Z"]
    rel = ["A", "B"]

    check("recall@5", recall_at_k(ret, rel, 5), 1.0)
    check("recall@1", recall_at_k(ret, rel, 1), 0.0)
    check("recall@2", recall_at_k(ret, rel, 2), 0.5)
    check("hit@5", hit_at_k(ret, rel, 5), 1.0)
    check("hit@1", hit_at_k(ret, rel, 1), 0.0)
    check("mrr", reciprocal_rank(ret, rel), 0.5)
    check("dcg@5", dcg([0, 1, 0, 1, 0]), 1.0616063116448706, tol=1e-9)
    check("ndcg@5", ndcg_at_k(ret, rel, 5), 0.6509209298071326, tol=1e-9)

    # Perfect ranking scores 1.0 on everything.
    check("perfect recall@5", recall_at_k(["A", "B", "X"], rel, 5), 1.0)
    check("perfect mrr", reciprocal_rank(["A", "B", "X"], rel), 1.0)
    check("perfect ndcg", ndcg_at_k(["A", "B", "X"], rel, 10), 1.0)

    # Nothing relevant retrieved scores 0.0 on everything.
    check("miss recall", recall_at_k(["X", "Y"], rel, 5), 0.0)
    check("miss mrr", reciprocal_rank(["X", "Y"], rel), 0.0)
    check("miss ndcg", ndcg_at_k(["X", "Y"], rel, 10), 0.0)

    # nDCG must be order-sensitive where recall is not: same items, worse
    # positions, same recall, lower nDCG. This is the whole reason to report it.
    early = ndcg_at_k(["A", "B", "X", "Y", "Z"], rel, 10)
    late = ndcg_at_k(["X", "Y", "Z", "A", "B"], rel, 10)
    if not late < early:
        failures.append("nDCG is not order-sensitive")
    if recall_at_k(["A", "B", "X", "Y", "Z"], rel, 10) != recall_at_k(
            ["X", "Y", "Z", "A", "B"], rel, 10):
        failures.append("recall should be order-insensitive but is not")

    # IDCG capped at k: 3 relevant items but k=2 still allows a perfect score.
    check("ndcg@2 capped", ndcg_at_k(["A", "B"], ["A", "B", "C"], 2), 1.0)

    # Undefined cases must raise rather than silently return 0, which would
    # quietly average an out-of-collection question into the retrieval scores.
    for fn, label in ((recall_at_k, "recall"), (hit_at_k, "hit")):
        try:
            fn(ret, [], 5)
            failures.append(f"{label} with no relevant ids should raise")
        except ValueError:
            pass
    try:
        ndcg_at_k(ret, [], 10)
        failures.append("ndcg with no relevant ids should raise")
    except ValueError:
        pass

    # Abstention detection.
    if not abstained("Not found in the documents."):
        failures.append("abstained() missed a refusal")
    if abstained("Paging translates a virtual address using a page table."):
        failures.append("abstained() flagged a real answer as a refusal")

    if failures:
        print(f"SELF-TEST FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("metrics self-test: all checks pass")
    print("  verified against hand-computed nDCG@5 = 0.650920 for")
    print("  relevant={A,B}, retrieved=[X,A,Y,B,Z]")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
