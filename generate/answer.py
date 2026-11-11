"""Stage 4: turn retrieved evidence into a written, cited answer.

The contract from architecture.md §7 is one line — `generate(query, evidence)
-> Answer` — and everything else in this file exists to make that call honest:

* **The evidence is numbered, and the numbers are what the model cites.** The
  prompt shows `[1] ... [5]`, the model is told to write `[1]` in the sentence
  it supports, and `cited_evidence()` maps those numbers back to `Evidence`
  objects. That mapping is what day 5's citation-accuracy metric measures, so
  it is built in from the start rather than reconstructed from prose later.
* **Images are attached, never described.** Image evidence stores a *file path*
  in `content`. Pasting that path into the prompt would invite the model to
  answer from the filename ("chloroplast.png" is a strong hint) without ever
  looking at the picture — exactly the bluffing that `scripts/check_vision.py`
  was written to catch. So the path goes into Ollama's `images` field and the
  prompt says only which numbered item each attachment is.
* **Generation is timed separately.** It is by far the slowest stage (~30-60 s
  on this CPU) and it is a whole column of the day 5 results table.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.evidence import Answer, Evidence  # noqa: E402

_client = None


def get_client():
    """One client per process. Importing this module must not require Ollama to
    be running — only calling it does."""
    global _client
    if _client is None:
        import ollama
        _client = ollama.Client(host=config.OLLAMA_HOST, timeout=config.GEN_TIMEOUT)
    return _client


class GenerationError(RuntimeError):
    """Ollama could not be reached or the model could not answer. Raised with a
    message that says what to do about it, because the usual cause is simply
    that `ollama serve` is not running in another terminal."""


# ---------------------------------------------------------------- the prompt

def usable_images(evidence: list[Evidence]) -> list[Evidence]:
    """Image evidence whose file is actually on disk.

    Ingestion writes the figures and the index stores their paths, so the two
    can drift apart — an index built against a since-deleted `data/images/`
    would otherwise fail deep inside the Ollama call with an unhelpful error.
    Silently dropping a missing figure and answering from the text is the
    better failure here, so long as it is visible in `Answer.timings`' sibling
    diagnostics; the CLI prints a warning.
    """
    return [e for e in evidence
            if e.is_image and e.image_path and e.image_path.exists()]


def attachment(e: Evidence) -> str | bytes:
    """One image, sized for the LLM.

    Small figures are passed as a path — simplest, and Ollama reads the file
    itself. Anything larger than GEN_IMAGE_MAX_PX is downscaled in memory and
    passed as bytes, because a full-page render costs the vision model several
    times more time than the figure it contains (see config.GEN_IMAGE_MAX_PX).
    The file on disk is never modified: the index, the citations and the
    Streamlit thumbnails all still point at the original.
    """
    from PIL import Image as PILImage

    path = Path(e.content)
    with PILImage.open(path) as img:
        if max(img.size) <= config.GEN_IMAGE_MAX_PX:
            return str(path)
        # Flatten onto white, never straight to RGB: dropping an alpha channel
        # puts black behind the drawing, and a figure exported with a
        # transparent background — which is most of them — would arrive at the
        # model as black lines on black.
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            flat = PILImage.new("RGB", img.size, "white")
            flat.paste(img, mask=img.split()[-1])
            img = flat
        else:
            img = img.convert("RGB")
        img.thumbnail((config.GEN_IMAGE_MAX_PX, config.GEN_IMAGE_MAX_PX))
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def build_prompt(question: str,
                 evidence: list[Evidence]) -> tuple[str, list[str | bytes]]:
    """The user message and the list of image paths to attach.

    Returns them together because the two must agree: the Nth attached image is
    described in the prompt as belonging to a specific numbered item, and a
    vision model has no other way to know which figure is which.
    """
    present = usable_images(evidence)
    attach_order = {e.id: i for i, e in enumerate(present, start=1)}

    lines = ["EVIDENCE", ""]
    for n, e in enumerate(evidence, start=1):
        if e.is_image:
            slot = attach_order.get(e.id)
            where = (f"attached image {slot} of {len(present)}" if slot
                     else "image unavailable — ignore this item")
            # citation() already ends in "(figure)", so `where` only has to say
            # which attachment this is.
            lines.append(f"[{n}] {e.citation()} — {where}")
        else:
            body = " ".join(e.content.split())
            lines.append(f"[{n}] {e.citation()}\n{body}")
        lines.append("")

    lines += [
        "QUESTION",
        "",
        question,
        "",
        # The closing line repeats the two rules the 3B model most often drops.
        # Small models weight the *last* instruction heavily, and the system
        # prompt alone was measurably not enough: it still answered a figure
        # question with a bare "[1]" and no sentence at all.
        "Answer in complete sentences, stating what the evidence shows. Never reply "
        'with only a citation number. If the evidence does not answer the question, '
        'say "Not found in the documents."',
        "",
        # A required trailing line, not only inline [n] markers. A 3B model
        # drops inline citations the moment the prompt emphasises anything else
        # — measured: pushing it to write prose made it stop citing entirely on
        # 3 of 3 questions. A fixed final line is a format it can hold onto, and
        # parse_citations() reads the whole answer, so inline markers still
        # count when the model does produce them.
        "Then end with a final line naming only the evidence you actually used, "
        "in exactly this format:",
        "Sources: [1], [3]",
    ]
    return "\n".join(lines), [attachment(e) for e in present]


# ---------------------------------------------------------------- citations

CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_citations(text: str, n_evidence: int) -> list[int]:
    """The evidence numbers the answer actually cited, in order of first use.

    Numbers outside the evidence range are dropped: a small local model does
    occasionally invent `[7]` when it was given five items, and counting that
    as a citation would inflate day 5's citation accuracy.
    """
    seen: list[int] = []
    for match in CITATION_RE.findall(text):
        i = int(match)
        if 1 <= i <= n_evidence and i not in seen:
            seen.append(i)
    return seen


def cited_evidence(answer: Answer) -> list[Evidence]:
    """The Evidence objects the answer referred to by number."""
    return [answer.evidence[i - 1]
            for i in parse_citations(answer.text, len(answer.evidence))]


# ---------------------------------------------------------------- generation

def warm_up(model: str | None = None) -> float:
    """Make Ollama load the model into RAM before anything is timed.

    Same trap as days 2 and 3, one layer further out: the first `chat()` call
    pays for a multi-gigabyte model load inside the Ollama server, and charging
    that to the first question would put a wrong number in the results table.
    Returns the seconds spent so a caller can report it as one-off cost.
    """
    t0 = time.perf_counter()
    try:
        get_client().chat(
            model=model or config.GEN_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1, "num_ctx": config.GEN_NUM_CTX},
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with instructions
        raise GenerationError(_unreachable_message(exc)) from exc
    return time.perf_counter() - t0


def _unreachable_message(exc: Exception) -> str:
    return (
        f"could not generate with {config.GEN_MODEL} at {config.OLLAMA_HOST}\n"
        f"  {type(exc).__name__}: {exc}\n"
        "  Ollama was installed without root, so there is no service: start it\n"
        "  in its own terminal with `ollama serve`, and check the model is\n"
        f"  pulled with `ollama pull {config.GEN_MODEL}`."
    )


def generate(question: str, evidence: list[Evidence],
             model: str | None = None) -> Answer:
    """The architecture.md §7 contract: question + evidence in, `Answer` out.

    The returned `Answer` carries exactly the evidence that was put in the
    prompt — not the shortlist, not the whole retrieval — so the interface can
    show the user the same sources the model was looking at.
    """
    model = model or config.GEN_MODEL

    if not evidence:
        # Nothing retrieved is a real answer, not an error: it is what an
        # honest system says when the collection does not cover the question,
        # and the LLM should not be given a chance to fill the silence.
        return Answer(question=question,
                      text="Not found in the documents. (Nothing was retrieved "
                           "for this question.)",
                      evidence=[], timings={"generate": 0.0})

    prompt, images = build_prompt(question, evidence)
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images

    t0 = time.perf_counter()
    try:
        resp = get_client().chat(
            model=model,
            messages=[{"role": "system", "content": config.SYSTEM_PROMPT}, message],
            options={"temperature": config.GEN_TEMPERATURE,
                     "num_ctx": config.GEN_NUM_CTX},
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with instructions
        raise GenerationError(_unreachable_message(exc)) from exc
    elapsed = time.perf_counter() - t0

    return Answer(
        question=question,
        text=resp.message.content.strip(),
        evidence=evidence,
        timings={"generate": elapsed},
    )


def ask(question: str, k: int | None = None,
        method: config.Method | None = None,
        model: str | None = None) -> Answer:
    """The whole pipeline in one call: retrieve, rerank, generate.

    Retrieval's per-stage timings are merged into the answer's, so a single
    `Answer` object carries the complete latency breakdown for one question —
    which is what day 5 needs to fill in the results table.
    """
    from retrieve import search as search_mod

    result = search_mod.search(question, k=k, method=method)
    answer = generate(question, result.evidence, model=model)
    answer.timings = {
        **{f"retrieve.{stage}": secs for stage, secs in result.timings.items()},
        **answer.timings,
    }
    answer.timings["total"] = (result.timings.get("total", 0.0)
                               + answer.timings["generate"])
    return answer
