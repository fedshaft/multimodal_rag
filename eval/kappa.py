"""Inter-annotator agreement on the answer_type labels (report section 3 / the
synthetic-data guidelines).

The question set is ours, so the brief wants evidence that its labels are not
just one person's opinion. Two people tag the same questions independently, and
Cohen's Kappa scores the agreement while discounting the agreement you would get
by chance.

    # 1. make a blank sheet for the second annotator
    .venv/bin/python eval/kappa.py --make-sheet annotator_b.csv

    # 2. they fill in the `label` column WITHOUT seeing eval_set.json,
    #    then score it against the tags in eval_set.json:
    .venv/bin/python eval/kappa.py --score annotator_b.csv

    # or, better, score two independent annotators against each other:
    .venv/bin/python eval/kappa.py --pair annotator_a.csv annotator_b.csv

Cohen's Kappa is computed here rather than imported so the report can show the
formula; the result is cross-checked against sklearn.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL = Path(__file__).resolve().parent / "eval_set.json"
LABELS = ["text", "figure", "both", "none"]


def load() -> list[dict]:
    return json.loads(EVAL.read_text())["questions"]


def make_sheet(path: Path, seed: int = 0) -> None:
    """Questions only, in shuffled order, with an empty label column.

    Shuffled because eval_set.json is grouped by type: handing over a sheet in
    that order tells the annotator the answer and the agreement score becomes
    meaningless.
    """
    qs = load()
    rows = [{"id": q["id"], "question": q["question"], "label": ""} for q in qs]
    random.Random(seed).shuffle(rows)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "label"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {path} — {len(rows)} questions, shuffled, labels blank\n")
    print("Give this to the second annotator with these instructions, and do")
    print("NOT let them see eval_set.json:\n")
    print("  For each question, decide what a reader would need in order to")
    print("  answer it from a collection of operating-systems course material:")
    print("    text   — a paragraph of prose answers it")
    print("    figure — you would need to look at a diagram, chart or table")
    print("    both   — genuinely needs the prose AND the picture together")
    print("    none   — the OS material would not contain this at all")
    print("\n  Put one of those four words in the `label` column. Do not")
    print("  discuss the questions with anyone until the sheet is finished.")


def cohen_kappa(a: list[str], b: list[str], labels: list[str]) -> tuple[float, float, float]:
    """(kappa, observed agreement, expected agreement).

        kappa = (Po - Pe) / (1 - Pe)

    Po is how often the two annotators agreed. Pe is how often they would be
    expected to agree purely by chance, from each annotator's own label
    frequencies. Kappa is therefore the share of the *available* agreement
    (above chance) that was actually achieved.
    """
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    return kappa, po, pe


def interpret(k: float) -> str:
    """Landis & Koch (1977) bands — the conventional reading, quoted in the
    report so the number is not left bare."""
    if k < 0:
        return "worse than chance"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


def score(path: Path) -> int:
    qs = {q["id"]: q for q in load()}
    with path.open() as f:
        rows = list(csv.DictReader(f))

    mine, theirs, ids = [], [], []
    missing, unknown = [], []
    for r in rows:
        qid = r["id"].strip()
        lab = r["label"].strip().lower()
        if qid not in qs:
            unknown.append(qid)
            continue
        if not lab:
            missing.append(qid)
            continue
        if lab not in LABELS:
            unknown.append(f"{qid}:{lab}")
            continue
        ids.append(qid)
        mine.append(qs[qid]["answer_type"])
        theirs.append(lab)

    if unknown:
        print(f"!! unrecognised: {unknown}")
    if missing:
        print(f"!! {len(missing)} unlabelled and skipped: {missing}")
    if not ids:
        print("nothing to score")
        return 1

    k, po, pe = cohen_kappa(mine, theirs, LABELS)

    print(f"\nCohen's Kappa on {len(ids)} questions")
    print("=" * 58)
    print(f"  observed agreement  Po = {po:.3f}  ({sum(x == y for x, y in zip(mine, theirs))}/{len(ids)})")
    print(f"  chance agreement    Pe = {pe:.3f}")
    print(f"  kappa                  = {k:.3f}   ({interpret(k)})")

    try:
        from sklearn.metrics import cohen_kappa_score
        print(f"  sklearn cross-check    = {cohen_kappa_score(mine, theirs, labels=LABELS):.3f}")
    except ImportError:
        pass

    print("\nconfusion (rows = annotator A / eval_set.json, cols = annotator B)")
    print(f"  {'':<8}" + "".join(f"{l:>9}" for l in LABELS))
    for la in LABELS:
        row = [sum(1 for x, y in zip(mine, theirs) if x == la and y == lb)
               for lb in LABELS]
        print(f"  {la:<8}" + "".join(f"{v:>9}" for v in row))

    disagreed = [(i, x, y) for i, x, y in zip(ids, mine, theirs) if x != y]
    if disagreed:
        print(f"\n{len(disagreed)} disagreement(s) — discuss each in the report:")
        for qid, x, y in disagreed:
            print(f"  {qid}  A={x:<7} B={y:<7}  {qs[qid]['question'][:64]}")
        print("\nResolve these by discussion and update eval_set.json. Report the")
        print("kappa BEFORE resolution — that is the number that measures whether")
        print("the labelling scheme was clear, and revising it after agreeing")
        print("would inflate it.")
    else:
        print("\nNo disagreements. Note in the report that perfect agreement on")
        print("40 items with this label distribution is a weak result to lean on:")
        print("say how the annotator was briefed, since a shared briefing is")
        print("itself a source of correlated judgement.")
    return 0


def _read_labels(path: Path) -> dict[str, str]:
    with path.open() as f:
        return {r["id"].strip(): r["label"].strip().lower()
                for r in csv.DictReader(f) if r.get("label", "").strip()}


def pair(path_a: Path, path_b: Path) -> int:
    """Cohen's Kappa between two independent annotators.

    This is the measure the brief actually asks for. Scoring one annotator
    against eval_set.json instead compares a person with a machine-drafted
    label, which is a different (and weaker) claim.
    """
    A, B = _read_labels(path_a), _read_labels(path_b)
    qs = {q["id"]: q for q in load()}
    ids = sorted(set(A) & set(B), key=lambda x: (len(x), x))

    only_a, only_b = set(A) - set(B), set(B) - set(A)
    unlabelled = set(qs) - (set(A) | set(B))
    if only_a or only_b:
        print(f"!! labelled by only one annotator — excluded: "
              f"A-only {sorted(only_a)}  B-only {sorted(only_b)}")
    if unlabelled:
        print(f"!! {len(unlabelled)} question(s) labelled by neither and excluded "
              f"from kappa: {sorted(unlabelled, key=lambda x: (len(x), x))}")

    a = [A[i] for i in ids]
    b = [B[i] for i in ids]
    k, po, pe = cohen_kappa(a, b, LABELS)

    print(f"\nCohen's Kappa — {path_a.stem} vs {path_b.stem}, {len(ids)} questions")
    print("=" * 62)
    print(f"  observed agreement  Po = {po:.4f}  "
          f"({sum(x == y for x, y in zip(a, b))}/{len(ids)})")
    print(f"  chance agreement    Pe = {pe:.4f}")
    print(f"  kappa                  = {k:.4f}   ({interpret(k)})")
    try:
        from sklearn.metrics import cohen_kappa_score
        print(f"  sklearn cross-check    = "
              f"{cohen_kappa_score(a, b, labels=LABELS):.4f}")
    except ImportError:
        pass

    print("\nconfusion (rows = A, cols = B)")
    print(f"  {'':<8}" + "".join(f"{l:>9}" for l in LABELS))
    for la in LABELS:
        print(f"  {la:<8}" + "".join(
            f"{sum(1 for x, y in zip(a, b) if x == la and y == lb):>9}"
            for lb in LABELS))

    dis = [(i, A[i], B[i]) for i in ids if A[i] != B[i]]
    if dis:
        print(f"\n{len(dis)} disagreement(s):")
        for qid, x, y in dis:
            print(f"  {qid}  A={x:<7} B={y:<7}  {qs[qid]['question'][:60]}")

    # Third view: how each annotator compares with the tags in eval_set.json.
    print("\nagreement with the drafted tags in eval_set.json:")
    for name, lab in ((path_a.stem, A), (path_b.stem, B)):
        common = [i for i in ids if i in lab]
        agree = sum(1 for i in common if lab[i] == qs[i]["answer_type"])
        kk, _, _ = cohen_kappa([lab[i] for i in common],
                               [qs[i]["answer_type"] for i in common], LABELS)
        print(f"  {name:<14} {agree}/{len(common)} = {agree/len(common):.3f}   "
              f"kappa {kk:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make-sheet", metavar="OUT.csv")
    g.add_argument("--score", metavar="FILLED.csv")
    g.add_argument("--pair", nargs=2, metavar=("A.csv", "B.csv"))
    args = ap.parse_args()

    if args.make_sheet:
        make_sheet(Path(args.make_sheet))
        return 0
    if args.pair:
        return pair(Path(args.pair[0]), Path(args.pair[1]))
    return score(Path(args.score))


if __name__ == "__main__":
    raise SystemExit(main())
