"""Day 6: the demo interface.

    .venv/bin/streamlit run app.py

Shows the thing the report argues about: a question goes in, a cited answer
comes out, the figures it used appear as thumbnails, and a toggle turns the
image index off so the difference is visible live rather than only in a table.

Generation needs `ollama serve` running in another terminal. Without it the page
still works as a retrieval demo — that is deliberate, because retrieval is the
part with the measured results and it should stay demonstrable if Ollama is
unavailable on demo day.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from retrieve import search as search_mod

st.set_page_config(page_title="Multimodal RAG over OS course material",
                   page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading encoders (one-off, ~10s) ...")
def _warm(method_key: str) -> float:
    return search_mod.warm_up(config.METHODS[method_key])


@st.cache_data(show_spinner=False)
def _corpus_stats() -> dict:
    from index import bm25_index, image_index, text_index
    try:
        return {"chunks": text_index.count(), "images": image_index.count()}
    except Exception:
        return {}


def stage_caption(timings: dict[str, float]) -> str:
    order = ("bm25", "dense", "clip", "merge", "rerank", "total")
    parts = [f"{s} {1000 * timings[s]:.0f} ms" for s in order if s in timings]
    return "  ·  ".join(parts)


def main() -> None:
    st.title("Multimodal RAG over operating-systems course material")
    st.caption(
        "69 documents · 2,431 text chunks · 843 figures. "
        "Ask a question; the system searches text and images, then a local "
        "vision model writes an answer citing what it used."
    )

    with st.sidebar:
        st.header("Configuration")
        keys = list(config.METHODS)
        labels = [config.METHODS[k].name for k in keys]
        idx = st.radio(
            "Retrieval method", range(len(keys)),
            format_func=lambda i: labels[i],
            index=keys.index("multimodal"),
            help="These are the five rows of the results table.",
        )
        method_key = keys[idx]
        st.caption(config.METHODS[method_key].note)

        k = st.slider("Evidence passed to the model", 3, 10,
                      config.TOP_K_FINAL)
        max_images = st.slider(
            "Of which, at most this many images", 0, 5,
            config.MAX_IMAGES_FINAL,
            help="The ablation in the report: more image slots help figure "
                 "questions and cost text questions more. 2 was the overall "
                 "optimum.",
        )
        do_generate = st.checkbox(
            "Generate a written answer", value=False,
            help="Needs `ollama serve` running. Slow on a CPU-only machine "
                 "(~1 min); retrieval alone is milliseconds.",
        )

        stats = _corpus_stats()
        if stats:
            st.divider()
            st.caption(f"indexed: {stats.get('chunks', '?')} chunks, "
                       f"{stats.get('images', '?')} images")

    examples = [
        "How does the operating system regain control of the CPU from a runaway process?",
        "What does the page table diagram show?",
        "Which memory section is at the highest addresses and which grows downward?",
        "How can a producer/consumer solution built with semaphores deadlock?",
        "How do I train a convolutional neural network in PyTorch?",
    ]
    picked = st.selectbox("Example questions", ["—"] + examples, index=0)
    default = "" if picked == "—" else picked
    question = st.text_input("Your question", value=default,
                             placeholder="Ask about processes, memory, "
                                         "scheduling, files, concurrency ...")

    if not question.strip():
        st.info(
            "Try a **figure question** — 'what does the page table diagram "
            "show?' — with the method set to *Hybrid (BM25 + dense)*, then "
            "switch to *Hybrid + CLIP images*. The text-only methods score "
            "**0.000** on figure questions in the evaluation, because those "
            "slides carry no indexed text at all."
        )
        return

    _warm(method_key)

    t0 = time.perf_counter()
    result = search_mod.search(question, k=k, method=config.METHODS[method_key])
    result.evidence = search_mod.select_final(
        result.candidates, k=k, max_images=max_images)
    elapsed = time.perf_counter() - t0

    answer = None
    if do_generate:
        from generate import answer as gen
        with st.spinner("Local vision model is writing an answer ..."):
            try:
                answer = gen.generate(question, result.evidence)
            except Exception as exc:
                st.error(
                    f"Generation unavailable ({type(exc).__name__}). Start it "
                    f"with `ollama serve` in another terminal. Retrieval "
                    f"results below are unaffected."
                )

    if answer is not None:
        st.subheader("Answer")
        st.write(answer.text)
        cited = gen.cited_evidence(answer)
        if cited:
            st.caption("cited: " + ", ".join(e.citation() for e in cited))
        else:
            st.caption("The model cited nothing specific.")

    texts = [e for e in result.evidence if not e.is_image]
    images = [e for e in result.evidence if e.is_image]

    st.subheader(f"Retrieved evidence — {len(texts)} text, {len(images)} image")
    st.caption(f"{stage_caption(result.timings)}  ·  wall clock "
               f"{1000 * elapsed:.0f} ms  ·  "
               f"{result.counts.get('merged', 0)} candidates before the cut")

    if images:
        cols = st.columns(min(len(images), 3))
        for col, e in zip(cols * 3, images):
            with col:
                p = Path(e.content)
                if p.exists():
                    st.image(str(p), use_container_width=True)
                st.caption(f"**{e.citation()}**  \n`{e.id}`  ·  "
                           f"score {e.score:.3f}")
    elif max_images == 0:
        st.info("Image slots set to 0 — text only.")
    else:
        st.warning(
            "No images retrieved. With a text-only method that is expected: "
            "it has no image index to search."
        )

    for e in texts:
        with st.expander(f"{e.citation()}  ·  score {e.score:.3f}"):
            st.caption(f"`{e.id}`")
            st.write(e.content)

    with st.expander("Full shortlist, before the final cut "
                     f"({len(result.candidates)} candidates)"):
        st.caption(
            "Retrieval finding the evidence and the final selection keeping it "
            "are different things. In the evaluation, figure recall over this "
            "shortlist is 0.786 against 0.357 after the cut."
        )
        for i, e in enumerate(result.candidates[:40], 1):
            kind = "image" if e.is_image else "text"
            st.text(f"{i:>3}. [{kind:>5}] {e.id:<34} {e.score:.4f}")


if __name__ == "__main__":
    main()
