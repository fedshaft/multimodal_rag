"""PDF -> text per page + figure images, all as Evidence objects.

Two things here are less obvious than they look:

1. **Vector diagrams are invisible to `get_images()`.** A diagram drawn with
   lines and shapes (very common in slide decks and papers) is not an embedded
   raster image, so naive extraction finds nothing and the figure never gets
   indexed. We detect those pages by their drawing-command count and render the
   whole page instead.

2. **IDs must survive re-ingestion.** The day-5 eval set refers to evidence by
   id, so an id is derived from the document name and page — never from a
   running counter over the whole collection, which would renumber everything
   the moment a document is added or removed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.evidence import Evidence  # noqa: E402
from ingest.chunker import chunk_pages  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def doc_stem(path: Path) -> str:
    """A filesystem-safe, stable short name for a document.

    "Lecture 04 - Backprop.pdf" -> "lecture_04_backprop"
    """
    stem = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    return stem or "doc"


def _page_text(page: pymupdf.Page) -> str:
    """Text of one page, in reading order, paragraphs kept apart."""
    blocks = [b for b in page.get_text("blocks") if b[6] == 0]
    blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))  # top-to-bottom, left-to-right
    parts = [b[4].strip() for b in blocks if b[4].strip()]
    return "\n\n".join(parts)


def _save_image(data: bytes, dest: Path) -> tuple[int, int] | None:
    """Write an extracted image as PNG. Returns its size, or None if it is too
    small to be a real figure (logos, bullet icons, decorative rules)."""
    import io

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return None
    if img.width < config.MIN_IMAGE_PIXELS or img.height < config.MIN_IMAGE_PIXELS:
        return None
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    return img.width, img.height


def _should_render_page(page: pymupdf.Page, text: str, n_embedded: int) -> str | None:
    """Decide whether to store a picture of the whole page. Returns the reason
    (which lands in metadata and is genuinely useful when debugging retrieval),
    or None."""
    mode = config.PAGE_RENDER_MODE
    if mode == "never":
        return None
    if mode == "all":
        return "all"
    if len(text.split()) < config.MIN_PAGE_TEXT_WORDS:
        return "scanned"  # no text to search — the render is the only handle on it
    if n_embedded == 0:
        try:
            if len(page.get_drawings()) >= config.MIN_VECTOR_DRAWINGS:
                return "vector"  # a drawn diagram, invisible to get_images()
        except Exception:
            return None
    return None


def parse_pdf(path: Path) -> tuple[list[tuple[int, str]], list[Evidence]]:
    """Returns (pages, images) for one PDF.

    pages  — [(page_number, text), ...], 1-based
    images — Evidence(modality="image"), files already written to IMAGES_DIR
    """
    stem = doc_stem(path)
    doc = pymupdf.open(path)
    pages: list[tuple[int, str]] = []
    images: list[Evidence] = []
    seen_xrefs: set[int] = set()  # a logo repeated on every slide shares one xref

    for pno, page in enumerate(doc, start=1):
        text = _page_text(page)
        pages.append((pno, text))

        kept_here = 0
        for k, info in enumerate(page.get_images(full=True)):
            xref = info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                raw = doc.extract_image(xref)
            except Exception:
                continue
            dest = config.IMAGES_DIR / f"{stem}_p{pno}_{k}.png"
            size = _save_image(raw["image"], dest)
            if size is None:
                continue
            kept_here += 1
            images.append(Evidence(
                id=f"{stem}_img_p{pno}_{k}",
                modality="image",
                content=str(dest),
                source_doc=path.name,
                page=pno,
                metadata={"kind": "embedded", "width": size[0], "height": size[1]},
            ))

        reason = _should_render_page(page, text, kept_here)
        if reason:
            dest = config.IMAGES_DIR / f"{stem}_page_p{pno}.png"
            try:
                pix = page.get_pixmap(dpi=config.PAGE_RENDER_DPI)
                dest.parent.mkdir(parents=True, exist_ok=True)
                pix.save(dest)
                images.append(Evidence(
                    id=f"{stem}_page_p{pno}",
                    modality="image",
                    content=str(dest),
                    source_doc=path.name,
                    page=pno,
                    metadata={"kind": "page", "why": reason,
                              "width": pix.width, "height": pix.height},
                ))
            except Exception:
                pass

    doc.close()
    return pages, images


def ingest_pdf(path: Path) -> list[Evidence]:
    """One PDF -> all its evidence, text and images together."""
    pages, images = parse_pdf(path)
    chunks = chunk_pages(pages, doc_stem(path), path.name)
    return chunks + images


def ingest_image(path: Path) -> list[Evidence]:
    """A standalone image dropped into the collection alongside the PDFs.

    The plan's fallback if the PDFs turn out to be figure-poor.
    """
    stem = doc_stem(path)
    dest = config.IMAGES_DIR / f"{stem}{path.suffix.lower()}"
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        return []
    if img.width < config.MIN_IMAGE_PIXELS or img.height < config.MIN_IMAGE_PIXELS:
        return []
    if dest.resolve() != path.resolve():
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(dest)
    return [Evidence(
        id=f"{stem}_img",
        modality="image",
        content=str(dest),
        source_doc=path.name,
        page=None,
        metadata={"kind": "standalone", "width": img.width, "height": img.height},
    )]


def ingest_path(path: Path) -> list[Evidence]:
    """Dispatch on file type. New modalities plug in here."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf(path)
    if suffix in IMAGE_SUFFIXES:
        return ingest_image(path)
    return []


def collect_documents(docs_dir: Path | None = None) -> list[Path]:
    """Every ingestible file in the collection, in a stable order."""
    docs_dir = docs_dir or config.DOCS_DIR
    wanted = {".pdf"} | IMAGE_SUFFIXES
    return sorted(p for p in docs_dir.glob("**/*")
                  if p.is_file() and p.suffix.lower() in wanted)
