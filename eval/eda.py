"""Day 5a: look at the corpus before measuring anything on it.

Report section 3 wants descriptive statistics, distributions, missing values,
outliers, class distribution, relationships and visualisation. This produces all
of them from the *ingested* corpus — the same Evidence objects retrieval sees,
not a separate reading of the PDFs, so what is charted here is what is searched.

    .venv/bin/python eval/eda.py            # uses the cached parse if present
    .venv/bin/python eval/eda.py --refresh  # re-parse the PDFs (~4 min)

Writes eval/results/figures/*.png and eval/results/eda_summary.txt.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display on this machine

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from ingest.pdf_parser import (  # noqa: E402
    collect_documents,
    doc_stem,
    parse_pdf,
)
from ingest.chunker import chunk_pages  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = RESULTS / "figures"
CACHE = RESULTS / "cache"

# --- palette -----------------------------------------------------------------
# Categorical slots 1 and 2 from the data-viz reference palette, in fixed order.
# Validated: adjacent CVD ΔE 24.7 (protan), normal-vision ΔE 33.6, both >= 3:1
# on the light surface. Two series only, so no cap is in play.
BLUE = "#2a78d6"      # slot 1 — OSTEP (textbook prose)
ORANGE = "#eb6834"    # slot 2 — slide decks
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdbd6"

SOURCE_COLOR = {"OSTEP chapters": BLUE, "Slide decks": ORANGE}


def _style() -> None:
    """Recessive grid and axes, thin marks, text in ink tokens — never in a
    series colour."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "font.size": 10,
        "figure.dpi": 150,
    })


def _finish(ax) -> None:
    """Drop the top/right spines so the data, not the box, carries the eye."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --- data --------------------------------------------------------------------

def source_of(doc_name: str) -> str:
    """The corpus has two deliberately different halves; almost every finding
    below splits on this, so it is derived once here."""
    return "Slide decks" if doc_name.startswith("slides_") else "OSTEP chapters"


def build_frames(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(pages, chunks, images) for the whole collection.

    Parsing costs ~4 minutes, so the three frames are cached as CSV. The parse
    is the same call ingestion makes, so these frames describe exactly what was
    indexed.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {k: CACHE / f"{k}.csv" for k in ("pages", "chunks", "images")}

    if not refresh and all(p.exists() for p in paths.values()):
        frames = {k: pd.read_csv(p) for k, p in paths.items()}
        print(f"loaded cached parse from {CACHE}  (--refresh to re-parse)")
        return frames["pages"], frames["chunks"], frames["images"]

    docs = collect_documents(config.DOCS_DIR)
    print(f"parsing {len(docs)} documents ...")
    page_rows, chunk_rows, image_rows = [], [], []

    for n, path in enumerate(docs, 1):
        stem = doc_stem(path)
        pages, images = parse_pdf(path)
        src = source_of(path.name)

        for pno, text in pages:
            page_rows.append({
                "doc": path.name, "source": src, "page": pno,
                "words": len(text.split()),
            })

        for ev in chunk_pages(pages, stem, path.name):
            chunk_rows.append({
                "id": ev.id, "doc": path.name, "source": src, "page": ev.page,
                "words": len(ev.content.split()),
            })

        for ev in images:
            image_rows.append({
                "id": ev.id, "doc": path.name, "source": src, "page": ev.page,
                "kind": ev.metadata.get("kind"),
                "width": ev.metadata.get("width"),
                "height": ev.metadata.get("height"),
            })

        if n % 10 == 0 or n == len(docs):
            print(f"  {n}/{len(docs)}")

    frames = {
        "pages": pd.DataFrame(page_rows),
        "chunks": pd.DataFrame(chunk_rows),
        "images": pd.DataFrame(image_rows),
    }
    for k, df in frames.items():
        df.to_csv(paths[k], index=False)
    print(f"cached to {CACHE}")
    return frames["pages"], frames["chunks"], frames["images"]


# --- figures -----------------------------------------------------------------

def fig_chunk_lengths(chunks: pd.DataFrame) -> Path:
    """The headline distribution, and the reason it is bimodal.

    Overlaid rather than stacked: the claim is that the two halves occupy
    different ranges, which a stack would hide.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.arange(0, chunks["words"].max() + 20, 20)

    for src, grp in chunks.groupby("source"):
        ax.hist(grp["words"], bins=bins, alpha=0.75, label=src,
                color=SOURCE_COLOR[src], edgecolor=SURFACE, linewidth=0.6)

    target = config.CHUNK_WORDS
    ax.set_xlim(0, target + 70)  # room for the cap label, which would otherwise
    ax.axvline(target, color=INK_2, linestyle="--", linewidth=1.2)
    ax.annotate(f"CHUNK_WORDS = {target}\n(the cap — most OSTEP\nchunks land exactly here)",
                xy=(target, ax.get_ylim()[1] * 0.88),
                xytext=(8, 0), textcoords="offset points",
                color=INK_2, fontsize=8.5, va="top")

    ax.set_xlabel("words in chunk")
    ax.set_ylabel("number of chunks")
    ax.set_title("Chunk lengths are bimodal — the corpus has two shapes")
    ax.legend(loc="upper left")  # collide with the cap label on the right
    _finish(ax)
    return _save(fig, "01_chunk_length_distribution.png")


def fig_words_per_page(pages: pd.DataFrame) -> Path:
    """Why the chunks came out bimodal: the pages themselves are."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.arange(0, 700, 25)
    for src, grp in pages.groupby("source"):
        ax.hist(grp["words"].clip(upper=680), bins=bins, alpha=0.75, label=src,
                color=SOURCE_COLOR[src], edgecolor=SURFACE, linewidth=0.6)

    ax.axvline(config.MIN_PAGE_TEXT_WORDS, color=INK_2, linestyle=":",
               linewidth=1.2)
    ax.annotate(f"< {config.MIN_PAGE_TEXT_WORDS} words:\ntext search is blind",
                xy=(config.MIN_PAGE_TEXT_WORDS, ax.get_ylim()[1] * 0.72),
                xytext=(10, 0), textcoords="offset points",
                color=INK_2, fontsize=8.5)

    ax.set_xlabel("words on page")
    ax.set_ylabel("number of pages")
    ax.set_title("A slide is not a page — median 70 words against 386")
    ax.legend(loc="upper right")
    _finish(ax)
    return _save(fig, "02_words_per_page.png")


def fig_image_classes(images: pd.DataFrame) -> Path:
    """Class distribution of the image index: what kind of picture, from where.

    Part-to-whole across two groups -> stacked horizontal bar, with a 2px
    surface gap between segments.
    """
    counts = (images.groupby(["source", "kind"]).size()
              .unstack(fill_value=0)
              .reindex(columns=["embedded", "page"], fill_value=0))

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ypos = np.arange(len(counts))
    left = np.zeros(len(counts))
    labels = {"embedded": "extracted figure", "page": "whole-page render"}
    shades = {"embedded": BLUE, "page": "#9ec3ee"}  # one hue, two shades

    for kind in ("embedded", "page"):
        vals = counts[kind].to_numpy()
        ax.barh(ypos, vals, left=left, height=0.55, color=shades[kind],
                label=labels[kind], edgecolor=SURFACE, linewidth=2)
        for y, (v, l) in enumerate(zip(vals, left)):
            if v > 40:
                ax.text(l + v / 2, y, str(v), ha="center", va="center",
                        color=INK if kind == "page" else SURFACE, fontsize=9)
        left += vals

    ax.set_yticks(ypos)
    ax.set_yticklabels(counts.index)
    ax.set_xlabel("images in the index")
    ax.set_title("What the image index actually holds")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _finish(ax)
    return _save(fig, "03_image_class_distribution.png")


def fig_figures_per_doc(images: pd.DataFrame, pages: pd.DataFrame) -> Path:
    """Magnitude across nominal categories -> one hue for every bar.

    Colouring by source is identity, which is the one thing that IS categorical
    here, so the two-slot palette still applies.
    """
    per_doc = (images.groupby(["doc", "source"]).size()
               .reset_index(name="figures"))
    # documents with no figures never appear in `images` — add them back, since
    # a zero is the point of the chart
    all_docs = pages[["doc", "source"]].drop_duplicates()
    per_doc = all_docs.merge(per_doc, on=["doc", "source"], how="left").fillna(
        {"figures": 0})
    per_doc = per_doc.sort_values("figures", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 9))
    colors = [SOURCE_COLOR[s] for s in per_doc["source"]]
    ax.barh(np.arange(len(per_doc)), per_doc["figures"], height=0.7,
            color=colors)
    ax.set_yticks(np.arange(len(per_doc)))
    ax.set_yticklabels([d.replace(".pdf", "") for d in per_doc["doc"]],
                       fontsize=6.5)
    ax.set_xlabel("figures indexed")
    ax.set_title("Figures per document")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in (BLUE, ORANGE)]
    ax.legend(handles, ["OSTEP chapters", "Slide decks"], loc="lower right")
    ax.grid(axis="y", visible=False)
    _finish(ax)
    return _save(fig, "04_figures_per_document.png")


def fig_text_invisible(pages: pd.DataFrame) -> Path:
    """The missing-value analysis. A page with no extractable text is this
    project's missing value: BM25 and the dense index cannot see it at all, and
    only the image index can retrieve it."""
    pages = pages.copy()
    pages["band"] = pd.cut(
        pages["words"],
        bins=[-1, 0, config.MIN_PAGE_TEXT_WORDS - 1, 100, 10 ** 6],
        labels=["0 words\n(no text at all)",
                f"1–{config.MIN_PAGE_TEXT_WORDS - 1}\n(effectively blind)",
                f"{config.MIN_PAGE_TEXT_WORDS}–100", "100+"])
    counts = (pages.groupby(["band", "source"], observed=False).size()
              .unstack(fill_value=0))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(counts))
    width = 0.38
    for i, src in enumerate(["OSTEP chapters", "Slide decks"]):
        if src not in counts:
            continue
        vals = counts[src].to_numpy()
        ax.bar(x + (i - 0.5) * (width + 0.02), vals, width,
               color=SOURCE_COLOR[src], label=src)
        for xi, v in zip(x + (i - 0.5) * (width + 0.02), vals):
            if v:
                ax.text(xi, v, f" {v}", ha="center", va="bottom",
                        color=INK_2, fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, fontsize=8.5)
    ax.set_ylabel("pages")
    ax.set_title("Missing values: pages text search cannot see")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    _finish(ax)
    return _save(fig, "05_text_invisible_pages.png")


def fig_pages_vs_chunks(pages: pd.DataFrame, chunks: pd.DataFrame) -> Path:
    """Relationship analysis. Two series only — an all-pairs form (scatter)
    caps at three."""
    per_doc = pages.groupby(["doc", "source"]).size().reset_index(name="pages")
    ch = chunks.groupby("doc").size().reset_index(name="chunks")
    m = per_doc.merge(ch, on="doc", how="left").fillna({"chunks": 0})

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for src, grp in m.groupby("source"):
        ax.scatter(grp["pages"], grp["chunks"], s=42, alpha=0.85,
                   color=SOURCE_COLOR[src], label=src,
                   edgecolor=SURFACE, linewidth=2)

    ax.set_xlabel("pages in document")
    ax.set_ylabel("chunks produced")
    ax.set_title("Chunks per page differ sharply by document type")
    ax.legend(loc="upper left")
    _finish(ax)
    return _save(fig, "06_pages_vs_chunks.png")


# --- the written summary -----------------------------------------------------

def describe(pages, chunks, images) -> str:
    out: list[str] = []
    w = out.append

    w("=" * 74)
    w("CORPUS ANALYSIS — report section 3")
    w("=" * 74)

    w("\n1. DESCRIPTIVE STATISTICS\n")
    per_src = pages.groupby("source")
    tbl = pd.DataFrame({
        "documents": per_src["doc"].nunique(),
        "pages": per_src.size(),
        "words": per_src["words"].sum(),
        "median words/page": per_src["words"].median(),
    })
    tbl.loc["TOTAL"] = [pages["doc"].nunique(), len(pages),
                        pages["words"].sum(), pages["words"].median()]
    w(tbl.to_string())

    w("\n  chunk word counts:")
    w(textwrap.indent(chunks["words"].describe().to_string(), "    "))

    w("\n2. CLASS DISTRIBUTION\n")
    w("  by corpus half (text chunks):")
    w(textwrap.indent(chunks["source"].value_counts().to_string(), "    "))
    w("\n  by image kind:")
    w(textwrap.indent(images["kind"].value_counts().to_string(), "    "))
    w("\n  NOTE: the question-type balance (text / figure / both) is the class")
    w("  distribution that matters for the results, and it does not exist yet.")
    w("  Chart it here once eval/eval_set.json is written.")

    w("\n3. MISSING VALUES\n")
    blind = pages[pages["words"] < config.MIN_PAGE_TEXT_WORDS]
    w(f"  pages with < {config.MIN_PAGE_TEXT_WORDS} words: {len(blind)} "
      f"({100 * len(blind) / len(pages):.1f}% of {len(pages)})")
    w("  These are this project's missing values — BM25 and the dense index")
    w("  hold nothing for them, so only the image index can retrieve them.")
    if len(blind):
        w(textwrap.indent(blind["source"].value_counts().to_string(), "    "))
    empty = pages[pages["words"] == 0]
    w(f"\n  pages with no extractable text at all: {len(empty)}")

    docs_no_fig = (set(pages["doc"]) - set(images["doc"]))
    w(f"  documents with zero indexed figures: {len(docs_no_fig)}")
    if docs_no_fig:
        w(textwrap.indent(", ".join(sorted(docs_no_fig)[:8]), "    "))

    w("\n4. OUTLIERS\n")

    def fence(s: pd.Series) -> tuple[float, float]:
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    lo, hi = fence(chunks["words"])
    w(f"  Chunk length: IQR fence [{lo:.0f}, {hi:.0f}], "
      f"{((chunks['words'] < lo) | (chunks['words'] > hi)).sum()} outliers.")
    w(f"  That zero is an artefact, not a clean result. Chunking bounds every")
    w(f"  chunk into [{config.MIN_CHUNK_WORDS}, {config.CHUNK_WORDS}] words by")
    w("  construction, and the distribution is so wide that the fence falls")
    w("  outside those bounds — so no chunk *can* be flagged. Reporting '0")
    w("  outliers' here without that caveat would be misleading. The bounded")
    w("  quantity to watch is not spread but pile-up at the cap:")
    at_cap = (chunks["words"] == config.CHUNK_WORDS).sum()
    at_floor = (chunks["words"] <= config.MIN_CHUNK_WORDS + 5).sum()
    w(f"    chunks at the {config.CHUNK_WORDS}-word cap: {at_cap} "
      f"({100 * at_cap / len(chunks):.0f}%)")
    w(f"    chunks within 5 words of the {config.MIN_CHUNK_WORDS}-word floor: "
      f"{at_floor} ({100 * at_floor / len(chunks):.0f}%)")

    w("\n  Page length is unbounded, so IQR is meaningful there. Per source:")
    for src, grp in pages.groupby("source"):
        lo_p, hi_p = fence(grp["words"])
        n_out = ((grp["words"] < lo_p) | (grp["words"] > hi_p)).sum()
        w(f"    {src:<16} fence [{max(lo_p, 0):.0f}, {hi_p:.0f}]  "
          f"outliers: {n_out} of {len(grp)} ({100 * n_out / len(grp):.1f}%)")

    docw = pages.groupby(["doc", "source"])["words"].sum().reset_index()
    w("\n  Document length outliers:")
    for src, grp in docw.groupby("source"):
        lo_d, hi_d = fence(grp["words"])
        odd = grp[(grp["words"] < lo_d) | (grp["words"] > hi_d)]
        if len(odd):
            for _, r in odd.iterrows():
                w(f"    {r['doc']:<28} {r['words']:>7,} words  ({src})")
        else:
            w(f"    {src}: none")

    big = docw.sort_values("words", ascending=False)
    w(f"\n  largest document:  {big.iloc[0]['doc']} "
      f"({big.iloc[0]['words']:,} words)")
    w(f"  smallest document: {big.iloc[-1]['doc']} "
      f"({big.iloc[-1]['words']:,} words)")
    w("\n  A very short chunk is usually a title slide or a section heading. It")
    w("  is still indexed, and it can win on a keyword match while carrying no")
    w("  answer — worth watching in the day-5 failure analysis.")

    w("\n5. RELATIONSHIPS\n")
    per_doc = pages.groupby(["doc", "source"]).size().reset_index(name="pages")
    ch = chunks.groupby("doc").size().reset_index(name="chunks")
    im = images.groupby("doc").size().reset_index(name="figures")
    m = (per_doc.merge(ch, on="doc", how="left")
                .merge(im, on="doc", how="left")
                .fillna({"chunks": 0, "figures": 0}))
    w("  Pearson correlation (per document):")
    w(textwrap.indent(
        m[["pages", "chunks", "figures"]].corr().round(3).to_string(), "    "))
    for src, grp in m.groupby("source"):
        rate = grp["chunks"].sum() / grp["pages"].sum()
        w(f"\n  {src}: {rate:.2f} chunks per page")

    w("\n6. WHAT THIS MEANS FOR THE EXPERIMENT\n")
    med = chunks.groupby("source")["words"].median()
    shape = ", ".join(f"{s} {v:.0f}" for s, v in med.items())
    w(f"  - Chunk lengths are bimodal (median words: {shape}). BM25")
    w("    normalises by document length, so the two halves are not competing")
    w("    on equal terms. Measure it before changing CHUNK_WORDS.")
    w("  - The slide decks carry the picture-dominant pages, so the figure")
    w("    questions must come from them or the multimodal comparison has")
    w("    nothing to show.")
    w("  - Whole-page renders outnumber extracted figures, so a 'figure' hit is")
    w("    often a picture of a whole page. Expect weaker CLIP precision than a")
    w("    tightly-cropped corpus would give, and say so in the limitations.")
    w("=" * 74)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-parse the PDFs instead of using the cache")
    args = ap.parse_args()

    _style()
    pages, chunks, images = build_frames(args.refresh)

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = describe(pages, chunks, images)
    print("\n" + summary)
    (RESULTS / "eda_summary.txt").write_text(summary + "\n")

    made = [
        fig_chunk_lengths(chunks),
        fig_words_per_page(pages),
        fig_image_classes(images),
        fig_figures_per_doc(images, pages),
        fig_text_invisible(pages),
        fig_pages_vs_chunks(pages, chunks),
    ]
    print("\nfigures written:")
    for p in made:
        print(f"  {p.relative_to(Path.cwd()) if p.is_relative_to(Path.cwd()) else p}")
    print(f"\nsummary written to {RESULTS / 'eda_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
