"""Long page text -> retrieval-sized chunks.

Why chunk at all: retrieval returns whole chunks, so a chunk is the unit of
what the answer gets to see. Index a whole 600-word page and a match on one
sentence drags in 599 words of unrelated text, which both dilutes the
embedding and wastes the LLM's attention.

**Chunks never span a page boundary.** That costs a little — a paragraph
running across a page break gets split — but it buys exact page numbers in
every citation, and citations are graded. For slide decks it is free anyway,
since a slide is already about one chunk long.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.evidence import Evidence  # noqa: E402


def split_words(text: str, size: int, overlap: int, min_words: int) -> list[str]:
    """Word-window split with overlap, dropping a runt tail."""
    words = text.split()
    if len(words) < min_words:
        return []
    if len(words) <= size:
        return [" ".join(words)]

    step = max(1, size - overlap)
    out: list[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        # A short final window is already covered by the previous one's overlap.
        if len(window) < min_words and out:
            break
        out.append(" ".join(window))
        if start + size >= len(words):
            break
    return out


def chunk_pages(pages: list[tuple[int, str]], stem: str, source_doc: str) -> list[Evidence]:
    """[(page_no, text)] -> text Evidence, numbered per document.

    The chunk counter runs across the whole document (not restarted per page)
    so ids stay unique and readable: doc3_chunk_18.
    """
    out: list[Evidence] = []
    n = 0
    for page_no, text in pages:
        for body in split_words(text, config.CHUNK_WORDS,
                                config.CHUNK_OVERLAP_WORDS, config.MIN_CHUNK_WORDS):
            out.append(Evidence(
                id=f"{stem}_chunk_{n}",
                modality="text",
                content=body,
                source_doc=source_doc,
                page=page_no,
                metadata={"words": len(body.split())},
            ))
            n += 1
    return out
