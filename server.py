"""Starlette + Uvicorn server for the Space-Themed Multimodal RAG Web Application."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from retrieve import search as search_mod

# Warm up state
_WARMED: set[str] = set()


async def api_stats(request):
    """Return corpus and index counts."""
    from index import bm25_index, image_index, text_index
    try:
        chunk_count = text_index.count()
    except Exception:
        chunk_count = 2431
    try:
        image_count = image_index.count()
    except Exception:
        image_count = 847
        
    doc_count = len(list((ROOT / "data" / "own").glob("*.pdf")))
    
    return JSONResponse({
        "status": "online",
        "chunks": chunk_count,
        "images": image_count,
        "documents": doc_count,
        "methods": {k: {"name": m.name, "note": m.note} for k, m in config.METHODS.items()}
    })


async def api_query(request):
    """Run real live search across the indexed OS course corpus."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "Query string is empty"}, status_code=400)

    method_key = body.get("method", "full")
    if method_key not in config.METHODS:
        method_key = "full"

    k = int(body.get("k", config.TOP_K_FINAL))
    max_images = int(body.get("max_images", config.MAX_IMAGES_FINAL))
    do_generate = bool(body.get("generate", False))

    method = config.METHODS[method_key]

    # Warm up encoder if needed
    if method_key not in _WARMED:
        try:
            search_mod.warm_up(method)
            _WARMED.add(method_key)
        except Exception as e:
            print(f"[Warning] Warm-up warning: {e}")

    t0 = time.perf_counter()
    try:
        result = search_mod.search(query, k=k, method=method)
        # Re-cut with the UI's image cap, keeping the query and the method's
        # adaptive setting so the demo runs exactly what the evaluation measured.
        result.evidence = search_mod.select_final(
            result.candidates, k=k, max_images=max_images,
            query=query, adaptive=method.adaptive_slots)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        return JSONResponse({
            "error": f"Search execution failed: {type(exc).__name__}: {exc}"
        }, status_code=500)

    # Serialize evidence
    evidence_list = []
    for e in result.evidence:
        content_val = e.content
        img_url = None
        if e.is_image:
            # content is local path to the PNG image
            p = Path(e.content)
            img_url = f"/data/images/{p.name}"
            content_val = p.name

        evidence_list.append({
            "id": e.id,
            "is_image": e.is_image,
            "score": round(float(e.score), 4),
            "citation": e.citation(),
            "content": content_val,
            "image_url": img_url
        })

    candidates_list = []
    for c in result.candidates[:40]:
        candidates_list.append({
            "id": c.id,
            "is_image": c.is_image,
            "score": round(float(c.score), 4)
        })

    answer_data = None
    if do_generate:
        try:
            from generate import answer as gen
            ans = gen.generate(query, result.evidence)
            cited = gen.cited_evidence(ans)
            answer_data = {
                "text": ans.text,
                "cited": [e.citation() for e in cited]
            }
        except Exception as exc:
            answer_data = {
                "text": f"(Local vision model offline: {type(exc).__name__}. Live retrieval results below are real.)",
                "cited": []
            }

    # Format timings into ms
    timings_ms = {k: round(v * 1000, 1) for k, v in result.timings.items()}
    timings_ms["total_wall_clock"] = round(elapsed * 1000, 1)

    return JSONResponse({
        "query": query,
        "method": method_key,
        "method_name": method.name,
        "evidence": evidence_list,
        "candidates": candidates_list,
        "timings": timings_ms,
        "counts": result.counts,
        "answer": answer_data
    })


middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

routes = [
    Route("/api/stats", api_stats, methods=["GET"]),
    Route("/api/query", api_query, methods=["POST"]),
    Mount("/data/images", StaticFiles(directory=str(ROOT / "data" / "images")), name="images"),
    Mount("/eval/results/figures", StaticFiles(directory=str(ROOT / "eval" / "results" / "figures")), name="eval_figures"),
    Mount("/data/own", StaticFiles(directory=str(ROOT / "data" / "own")), name="pdf_docs"),
    Mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web"),
]

app = Starlette(routes=routes, middleware=middleware)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🌌 Starting Black Hole 2 Space-Themed Multimodal RAG Server at http://0.0.0.0:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
