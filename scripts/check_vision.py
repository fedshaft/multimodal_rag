"""Day 1 gate: prove the local vision model can actually see images.

The whole project rests on this. If the model silently ignores attached images
and answers from text alone, every later stage would still *run* — it would
just quietly produce nonsense on exactly the figure questions the project is
built to win on. So we check it now, before anything depends on it.

The test image contains facts that cannot be guessed from the prompt: a
specific shape, a specific colour, and a number. A model that is bluffing gets
them wrong.

    .venv/bin/python scripts/check_vision.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

OUT = Path(__file__).parent / "_vision_test.png"

# What the image contains, and what a model that truly sees it should say.
EXPECTED = {
    "triangle": ["triangle"],
    "green": ["green"],
    "number 47": ["47"],
}


def make_test_image(path: Path) -> None:
    """A picture whose contents cannot be inferred from the question text."""
    img = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    d.polygon([(256, 60), (450, 400), (62, 400)], fill="#22aa44", outline="black")
    d.text((236, 300), "47", fill="white", font_size=90)
    img.save(path)


def main() -> int:
    try:
        import ollama
    except ImportError:
        print("FAIL: the `ollama` python package is not installed")
        return 1

    make_test_image(OUT)
    print(f"test image written to {OUT}")

    client = ollama.Client(host=config.OLLAMA_HOST)

    try:
        models = [m.model for m in client.list().models]
    except Exception as exc:
        print(f"FAIL: cannot reach Ollama at {config.OLLAMA_HOST}\n  {exc}")
        print("\n  Start the server in another terminal:  ollama serve")
        return 1
    print(f"ollama is up at {config.OLLAMA_HOST}; models present: {models or 'none'}")

    if not any(m.startswith(config.GEN_MODEL.split(":")[0]) for m in models):
        print(f"FAIL: model {config.GEN_MODEL} not pulled.  Run: ollama pull {config.GEN_MODEL}")
        return 1

    prompt = (
        "Look at this image. In one short sentence: what shape is it, "
        "what colour is the shape, and what number is written on it?"
    )
    print(f"\nasking {config.GEN_MODEL} ...")
    t0 = time.perf_counter()
    try:
        resp = client.chat(
            model=config.GEN_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [str(OUT)]}],
        )
    except Exception as exc:
        print(f"FAIL: generation errored\n  {exc}")
        return 1
    elapsed = time.perf_counter() - t0

    answer = resp.message.content.strip()
    print(f"\n--- answer ({elapsed:.1f}s) ---\n{answer}\n---")

    low = answer.lower()
    missed = [label for label, words in EXPECTED.items()
              if not any(w in low for w in words)]

    if missed:
        print(f"FAIL: the model did not correctly report: {', '.join(missed)}")
        print("  It is probably ignoring the attached image. Try a different")
        print("  vision model (llava:7b, moondream, gemma3:4b) before building on this.")
        return 1

    print(f"PASS: vision works. {elapsed:.1f}s per image answer on this machine.")
    print("      (that latency is a development-speed number — the Mac will be far quicker)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
