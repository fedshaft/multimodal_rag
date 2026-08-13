"""The shared contract between every stage of the pipeline.

Ingestion, retrieval, reranking, generation and evaluation all speak this one
type. Adding a new modality later (video, audio, tables) means writing a new
ingester that returns Evidence objects — no other stage needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass
class Evidence:
    """One retrievable piece of a document: a chunk of text, or an image."""

    id: str
    """Stable, unique, human-readable. e.g. "doc3_chunk_18", "doc3_img_p8_1".

    This is the ID written in the eval set, so it must stay the same across
    re-ingestion runs or the labelled questions stop matching.
    """

    modality: str
    """"text" or "image". Later: "video", "audio", "table"."""

    content: str
    """For text evidence: the text itself.
    For image evidence: the path to the image file on disk."""

    source_doc: str
    """Which document this came from, e.g. "lecture04.pdf"."""

    page: int | None = None
    """1-based page number, or None if not applicable."""

    score: float = 0.0
    """Relevance score from whichever stage produced this object. Scores from
    different stages are not comparable — a retrieval score and a reranker
    score are on completely different scales."""

    metadata: dict = field(default_factory=dict)
    """Anything modality-specific: image dimensions, chunk word count,
    bounding box, the rank this came in at, etc."""

    @property
    def is_image(self) -> bool:
        return self.modality == "image"

    @property
    def image_path(self) -> Path | None:
        """The image file, or None for non-image evidence."""
        return Path(self.content) if self.is_image else None

    def with_score(self, score: float) -> "Evidence":
        """A copy with a new score. Used by retrieval and reranking so a stage
        never mutates the objects handed to it."""
        return replace(self, score=score)

    def citation(self) -> str:
        """Short human-readable source label, shown to the user and asked of
        the LLM: "lecture04.pdf p.8 (figure)"."""
        parts = [self.source_doc]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.is_image:
            parts.append("(figure)")
        return " ".join(parts)

    def preview(self, width: int = 100) -> str:
        """One-line summary for debugging and log output."""
        if self.is_image:
            body = f"[image] {self.content}"
        else:
            body = " ".join(self.content.split())
        if len(body) > width:
            body = body[: width - 1] + "…"
        return f"{self.id:<28} {self.score:>7.4f}  {body}"


@dataclass
class Answer:
    """What the generation stage returns: the written answer plus exactly the
    evidence it was given, so the interface can show the sources."""

    question: str
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    """Seconds per stage: {"retrieve": 0.02, "rerank": 0.08, "generate": 11.4}.
    Latency is half the result table, so it is collected from the start."""

    @property
    def images(self) -> list[Evidence]:
        """Just the image evidence — what the Streamlit page shows as thumbnails."""
        return [e for e in self.evidence if e.is_image]

    @property
    def sources(self) -> list[str]:
        return [e.citation() for e in self.evidence]
