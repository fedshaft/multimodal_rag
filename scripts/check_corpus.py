"""Day 1 gate: is this document collection actually multimodal?

The project's headline result depends on having questions that only a figure
can answer. If the PDFs turn out to be walls of text, that result cannot exist
— and it is much cheaper to discover that now than on day 5.

This does not extract anything (that is day 2's pdf_parser). It only counts,
so it is fast and safe to re-run as you add files.

    .venv/bin/python scripts/check_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from ingest.pdf_parser import _drawn_images, _template_xrefs  # noqa: E402

TARGET_DOCS = 30
TARGET_FIGURES = 60      # roughly two usable figures per document
MIN_PX = config.MIN_IMAGE_PIXELS


def survey(path: Path) -> dict:
    """Count pages, words and non-trivial embedded images in one PDF."""
    doc = pymupdf.open(path)
    template = _template_xrefs(doc)
    seen: set[int] = set()
    words = 0
    figures = 0
    vector = 0
    tiny = 0
    text_pages = 0
    for page in doc:
        text = page.get_text()
        words += len(text.split())
        if len(text.split()) > 30:
            text_pages += 1
        n_embedded = 0
        for img in _drawn_images(page):
            # Count exactly what pdf_parser will keep, so the gate and the
            # parser can never disagree again.
            if img["xref"] in template:
                tiny += 1
            elif img["xref"] in seen:
                continue
            else:
                seen.add(img["xref"])
                figures += 1
                n_embedded += 1
        # A diagram drawn as lines and shapes is invisible to get_images().
        # pdf_parser._should_render_page renders those pages instead, so the
        # gate has to count them or it under-reports a figure-rich collection
        # (OSTEP is entirely vector: 754 pages, 0 embedded rasters, ~200
        # real diagrams).
        if n_embedded == 0:
            try:
                if len(page.get_drawings()) >= config.MIN_VECTOR_DRAWINGS:
                    vector += 1
            except Exception:
                pass
    n_pages = doc.page_count
    doc.close()
    return {
        "pages": n_pages,
        "words": words,
        "figures": figures,
        "vector": vector,
        "tiny": tiny,
        "text_pages": text_pages,
    }


def main() -> int:
    pdfs = sorted(config.DOCS_DIR.glob("**/*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {config.DOCS_DIR}")
        print("Put 30-50 PDFs on one topic there, then re-run this.")
        return 1

    print(f"{'document':<40} {'pages':>6} {'words':>8} {'raster':>7} {'vector':>7}")
    print("-" * 70)

    totals = {"pages": 0, "words": 0, "figures": 0, "vector": 0, "tiny": 0,
              "text_pages": 0}
    figure_poor = []
    scanned = []

    for p in pdfs:
        try:
            s = survey(p)
        except Exception as exc:
            print(f"{p.name:<44}  !! could not read: {exc}")
            continue
        for k in totals:
            totals[k] += s[k]
        name = p.name if len(p.name) <= 39 else p.name[:36] + "..."
        print(f"{name:<40} {s['pages']:>6} {s['words']:>8} {s['figures']:>7} "
              f"{s['vector']:>7}")
        if s["figures"] + s["vector"] == 0:
            figure_poor.append(p.name)
        # Lots of pages, almost no extractable text = scanned images of text.
        if s["pages"] >= 3 and s["text_pages"] / s["pages"] < 0.25:
            scanned.append(p.name)

    n = len(pdfs)
    print("-" * 70)
    print(f"{'TOTAL ' + str(n) + ' documents':<40} {totals['pages']:>6} "
          f"{totals['words']:>8} {totals['figures']:>7} {totals['vector']:>7}")
    print(f"\n(also skipped {totals['tiny']} template images repeated on many "
          f"pages — deck logos, footers)")

    print("\n" + "=" * 70)
    ok = True

    if n < TARGET_DOCS:
        print(f"THIN: {n} documents, aiming for {TARGET_DOCS}-50. Add more.")
        ok = False
    else:
        print(f"OK: {n} documents.")

    usable = totals["figures"] + totals["vector"]
    if usable < TARGET_FIGURES:
        print(f"PROBLEM: only {usable} usable figures across the whole "
              f"collection (want {TARGET_FIGURES}+).")
        print("  The multimodal comparison needs figure-answerable questions, and")
        print("  there are not enough figures here to write ~10-15 of them.")
        print("  Fix: prefer slide decks and papers with diagrams, or drop")
        print("  standalone images into data/own/ alongside the PDFs.")
        ok = False
    else:
        print(f"OK: {usable} usable figures ({totals['figures']} embedded rasters, "
              f"{totals['vector']} vector-diagram pages) — enough for figure "
              f"questions.")

    if figure_poor:
        shown = ", ".join(figure_poor[:6])
        more = f" (+{len(figure_poor) - 6} more)" if len(figure_poor) > 6 else ""
        print(f"\nNote: {len(figure_poor)} documents have no figures of either "
              f"kind: {shown}{more}")

    if scanned:
        print(f"\nNote: these look scanned (little extractable text): "
              f"{', '.join(scanned[:6])}")
        print("  Day 2 renders whole pages as images, so CLIP can still retrieve")
        print("  them — but BM25 and dense text search will find nothing in them.")

    print("=" * 70)
    if ok:
        print("Collection looks good. While reading these, jot down any question")
        print("a FIGURE answers — those notes become the day 5 eval set.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
