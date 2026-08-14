"""Generate a small throwaway PDF corpus to exercise the whole pipeline.

This is a *self-test fixture*, not part of the project's data. It exists so
ingestion, chunking and all three indexes can be verified end to end without
waiting for the real collection — and so a later refactor has something to be
checked against.

It deliberately includes the awkward cases:
  - a page whose figure is an embedded raster image
  - a page whose figure is *vector* art (invisible to get_images())
  - a near-textless page (stands in for a scan)
  - a repeated logo, which must be de-duplicated rather than indexed 5 times

    .venv/bin/python scripts/make_test_pdfs.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "_selftest"

LOREM = (
    "The system processes each input in sequence and updates its internal state. "
    "Each stage transforms the representation before passing it onward. "
    "Errors are propagated backwards so that every parameter receives a share of "
    "the correction proportional to its contribution. "
)


def _png(draw_fn, size=(400, 300), bg="white") -> bytes:
    img = Image.new("RGB", size, bg)
    draw_fn(ImageDraw.Draw(img), *size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def leaf(d, w, h):
    d.ellipse([60, 40, w - 60, h - 40], fill="#2e8b2e", outline="black", width=3)
    d.line([w // 2, 40, w // 2, h - 40], fill="black", width=4)
    d.text((20, 10), "chloroplast cross-section", fill="black", font_size=18)


def circuit(d, w, h):
    d.rectangle([50, 100, 350, 200], outline="black", width=4)
    d.rectangle([150, 80, 250, 120], fill="#cc2222", outline="black", width=3)
    d.text((20, 20), "resistor in series", fill="black", font_size=18)


def logo(d, w, h):
    d.ellipse([10, 10, w - 10, h - 10], fill="#3355aa")


def add_text_page(doc, title: str, body: str):
    page = doc.new_page()
    page.insert_text((72, 90), title, fontsize=18)
    text = body * 6
    page.insert_textbox(pymupdf.Rect(72, 120, 520, 720), text, fontsize=11)
    return page


def add_image_page(doc, title: str, caption: str, png: bytes):
    page = doc.new_page()
    page.insert_text((72, 90), title, fontsize=18)
    page.insert_image(pymupdf.Rect(120, 120, 480, 390), stream=png)
    page.insert_textbox(pymupdf.Rect(72, 400, 520, 700),
                        caption + " " + LOREM * 3, fontsize=11)
    return page


def add_vector_page(doc, title: str):
    """A bar chart drawn with vector commands — get_images() finds nothing here."""
    page = doc.new_page()
    page.insert_text((72, 90), title, fontsize=18)
    heights = [40, 90, 150, 110, 200, 60, 130]
    for i, hgt in enumerate(heights):
        x = 90 + i * 55
        page.draw_rect(pymupdf.Rect(x, 400 - hgt, x + 40, 400),
                       color=(0, 0, 0), fill=(0.2, 0.4, 0.8), width=1)
    page.draw_line(pymupdf.Point(80, 400), pymupdf.Point(500, 400))
    page.draw_line(pymupdf.Point(80, 400), pymupdf.Point(80, 180))
    page.insert_textbox(pymupdf.Rect(72, 430, 520, 700),
                        "Figure 2: accuracy by configuration. " + LOREM * 2,
                        fontsize=11)
    return page


def add_sparse_page(doc, png: bytes):
    """Almost no text — stands in for a scanned page."""
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(100, 150, 500, 450), stream=png)
    page.insert_text((72, 90), "Plate IV", fontsize=14)
    return page


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.pdf"):
        old.unlink()

    leaf_png = _png(leaf)
    circuit_png = _png(circuit)
    logo_png = _png(logo, size=(64, 64))

    # doc 1 — biology, embedded raster figure
    doc = pymupdf.open()
    add_text_page(doc, "Photosynthesis Overview", LOREM)
    add_image_page(doc, "Chloroplast Structure",
                   "Figure 1: the green pigment sits in stacked membranes.", leaf_png)
    add_vector_page(doc, "Measured Yields")
    for _ in range(3):  # same logo on several pages — must be de-duplicated
        p = doc.new_page()
        p.insert_image(pymupdf.Rect(20, 20, 60, 60), stream=logo_png)
        p.insert_textbox(pymupdf.Rect(72, 120, 520, 700), LOREM * 4, fontsize=11)
    doc.save(OUT_DIR / "biology notes.pdf")
    doc.close()

    # doc 2 — electronics, embedded raster figure + sparse page
    doc = pymupdf.open()
    add_text_page(doc, "Introduction to Circuits", LOREM)
    add_image_page(doc, "Series Resistance",
                   "Figure 1: current is identical through every element.", circuit_png)
    add_sparse_page(doc, circuit_png)
    doc.save(OUT_DIR / "circuits.pdf")
    doc.close()

    # doc 3 — text only, no figures at all
    doc = pymupdf.open()
    for i in range(4):
        add_text_page(doc, f"Chapter {i + 1}", LOREM)
    doc.save(OUT_DIR / "theory_only.pdf")
    doc.close()

    made = sorted(p.name for p in OUT_DIR.glob("*.pdf"))
    print(f"wrote {len(made)} test PDFs to {OUT_DIR}:")
    for m in made:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
