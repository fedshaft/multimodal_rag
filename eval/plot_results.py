"""Charts for the results section, from eval/results/results.json.

    .venv/bin/python eval/plot_results.py

Three figures:
  07  method comparison — the results table as a chart
  08  by question type  — the headline result
  09  quality vs latency — the trade-off the discussion argues about

Palette and mark rules follow the same data-viz method as eval/eda.py; the
two-slot categorical pair was validated with the skill's checker (adjacent CVD
delta-E 24.7, normal-vision 33.6, both >= 3:1 on the light surface).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = RESULTS / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
PALE = "#9ec3ee"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdbd6"
MUTED = "#b9b8b2"

SHORT = {
    "BM25 (keyword, text only)": "BM25",
    "Dense text (BGE-small)": "Dense",
    "Hybrid (BM25 + dense)": "Hybrid",
    "Hybrid + CLIP images": "+ CLIP",
    "Full system (+ reranker)": "+ Rerank",
    "Tuned system (adaptive + rerank)": "Tuned",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.labelsize": 10, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "legend.labelcolor": INK_2, "font.size": 10, "figure.dpi": 150,
    })


def _finish(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    p = FIGURES / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_methods(results: list[dict]) -> Path:
    """Three metrics as small multiples, one panel each.

    Not a grouped bar with three series: the metrics are different quantities on
    the same 0-1 scale, and putting them side by side invites reading across
    them as if they were comparable. One hue throughout — the methods are an
    ordered progression, and identity is carried by the axis labels.
    """
    names = [SHORT.get(r["name"], r["name"]) for r in results]
    metrics = [("recall@5", "Recall@5"), ("mrr", "MRR"), ("ndcg@10", "nDCG@10")]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    x = np.arange(len(names))
    for ax, (key, label) in zip(axes, metrics):
        vals = [r["overall"].get(key, 0) for r in results]
        best = int(np.argmax(vals))
        # Emphasis: the winner in the accent hue, the rest recessive.
        colors = [BLUE if i == best else PALE for i in range(len(vals))]
        ax.bar(x, vals, width=0.66, color=colors)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom",
                    color=INK_2, fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        _finish(ax)
    axes[0].set_ylabel("score")
    fig.suptitle("Retrieval quality by method", y=1.04, fontsize=12.5,
                 fontweight="semibold", color=INK)
    return _save(fig, "07_method_comparison.png")


def fig_by_type(results: list[dict]) -> Path:
    """The headline: what each method does on each kind of question.

    Small multiples by question type rather than one grouped chart, because the
    comparison the reader must make is *within* a question type (which method
    wins on figures?), not across types.
    """
    types = [("text", "Text questions"), ("figure", "Figure questions"),
             ("both", "Needs both")]
    names = [SHORT.get(r["name"], r["name"]) for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    x = np.arange(len(names))
    for ax, (t, label) in zip(axes, types):
        vals = [r["by_type"].get(t, {}).get("recall@5", 0.0) for r in results]
        n = next((r["by_type"][t]["n"] for r in results if t in r["by_type"]), 0)
        # Colour by whether the method can see images at all — that is the
        # variable the chart is about, and it is identity, not magnitude.
        colors = [ORANGE if r["key"] in ("multimodal", "full", "tuned") else BLUE
                  for r in results]
        ax.bar(x, vals, width=0.66, color=colors)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom",
                    color=INK_2, fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(f"{label}  (n={int(n)})", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        _finish(ax)

    axes[0].set_ylabel("Recall@5")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)]
    axes[2].legend(handles, ["text-only", "sees images"], loc="upper left")
    fig.suptitle("Recall@5 by question type — where the images earn their place",
                 y=1.04, fontsize=12.5, fontweight="semibold", color=INK)
    return _save(fig, "08_by_question_type.png")


def fig_tradeoff(results: list[dict]) -> Path:
    """Quality against cost. One series, every point direct-labelled."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    xs = [r["overall"]["latency_ms"] for r in results]
    ys = [r["overall"].get("recall@5", 0) for r in results]
    names = [SHORT.get(r["name"], r["name"]) for r in results]

    ax.plot(xs, ys, color=MUTED, linewidth=1.4, zorder=1)
    ax.scatter(xs, ys, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2)
    for xi, yi, nm in zip(xs, ys, names):
        ax.annotate(nm, (xi, yi), xytext=(0, 10), textcoords="offset points",
                    ha="center", color=INK_2, fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("latency per query (ms, log scale)")
    ax.set_ylabel("Recall@5")
    ax.set_title("Every gain in accuracy costs milliseconds")
    ax.set_ylim(0, 1.05)
    _finish(ax)
    return _save(fig, "09_quality_vs_latency.png")


def main() -> int:
    path = RESULTS / "results.json"
    if not path.exists():
        print(f"{path} not found — run eval/run_eval.py first")
        return 1
    results = json.loads(path.read_text())["results"]
    _style()
    made = [fig_methods(results), fig_by_type(results), fig_tradeoff(results)]
    print("figures written:")
    for p in made:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
