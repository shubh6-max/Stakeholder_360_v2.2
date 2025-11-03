# s360_rag/retriever.py
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple, Any
from sqlalchemy.orm import Session
from time import perf_counter

from langchain_openai import AzureOpenAIEmbeddings

from .db import raw_cosine_topk
from .config import (
    AZURE_ENDPOINT,
    AZURE_API_KEY,
    AZURE_EMBED_DEPLOYMENT,
    AZURE_API_VERSION,
    SIM_THRESHOLD,
    TOP_K,
)

# ---------------------------
# Embedding client (lazy, cached)
# ---------------------------
@lru_cache(maxsize=1)
def _emb_client() -> AzureOpenAIEmbeddings:
    """
    Initialize once and reuse. langchain_openai's AzureOpenAIEmbeddings expects:
      - model: deployment name (e.g., 'text-embedding-3-small' or your custom deployment)
      - azure_endpoint, api_key, api_version
    """
    if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_EMBED_DEPLOYMENT and AZURE_API_VERSION):
        raise RuntimeError(
            "Azure embeddings config missing. Ensure AZURE_ENDPOINT, AZURE_API_KEY, "
            "AZURE_EMBED_DEPLOYMENT, and AZURE_API_VERSION are set."
        )
    return AzureOpenAIEmbeddings(
        model=AZURE_EMBED_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )


def embed_text(text: str) -> List[float]:
    """
    Embed a single query string to a vector[List[float]].
    """
    return _emb_client().embed_query(text or "")


def _distance_to_similarity(distance: float) -> float:
    """
    pgvector cosine distance d = 1 - cos_sim  (cos_sim ∈ [-1, 1] for unnormalized).
    A simple similarity proxy is sim = 1 - d.
    We clamp to [0, 1] to keep UI friendly.
    """
    try:
        sim = 1.0 - float(distance)
        if sim < 0.0:
            return 0.0
        if sim > 1.0:
            return 1.0
        return sim
    except Exception:
        return 0.0


def top_matches(
    session: Session,
    question: str,
    *,
    top_k: int | None = None,
    sim_threshold: float | None = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Retrieve and score top matches for a query.

    Returns:
      scored: List of dicts:
        {
          "case_id": str,
          "case_title": str,
          "source_file": str,
          "max_sim": float,
          "mean_top2": float,
          "accept": bool,
          "chunks": [
            {"page_no": int, "chunk": str, "dist": float, "sim": float},
            ...
          ]
        }
      latency_ms: int
    """
    k = top_k or TOP_K
    thr = sim_threshold if sim_threshold is not None else SIM_THRESHOLD
    # Convert similarity threshold (>= thr) into a cosine-distance threshold (<= 1 - thr)
    dist_threshold = 1.0 - float(thr)

    t0 = perf_counter()
    qvec = embed_text(question)
    rows = raw_cosine_topk(session, qvec, k)
    ms = int((perf_counter() - t0) * 1000)

    # rows schema: (case_id, case_title, source_file, page_no, chunk, distance)
    grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"title": None, "file": None, "chunks": []})
    for (case_id, title, src, page_no, chunk, dist) in rows:
        d = float(dist)
        grouped[case_id]["title"] = title or case_id
        grouped[case_id]["file"] = src
        grouped[case_id]["chunks"].append(
            {"page_no": int(page_no or 0), "chunk": chunk or "", "dist": d, "sim": _distance_to_similarity(d)}
        )

    scored: List[Dict[str, Any]] = []
    for cid, obj in grouped.items():
        chs = sorted(obj["chunks"], key=lambda x: x["dist"])
        if not chs:
            continue
        # best & stability metrics
        best_dist = chs[0]["dist"]
        best_sim = chs[0]["sim"]
        mean_top2 = (
            (chs[0]["sim"] + chs[1]["sim"]) / 2.0 if len(chs) > 1 else chs[0]["sim"]
        )

        accept = best_dist <= dist_threshold
        scored.append(
            {
                "case_id": cid,
                "case_title": obj["title"],
                "source_file": obj["file"],
                "max_sim": round(best_sim, 4),
                "mean_top2": round(mean_top2, 4),
                "accept": accept,
                "chunks": chs[:3],
            }
        )

    # Order: accepted first → highest max_sim → then highest mean_top2 → finally by title as stable tiebreaker
    scored.sort(key=lambda x: (not x["accept"], -x["max_sim"], -x["mean_top2"], (x["case_title"] or "")))
    return scored, ms


def build_context(chosen: Dict[str, Any]) -> str:
    """
    Concatenate top chunks for prompting.
    """
    if not chosen:
        return ""
    return "\n\n".join([f"[p.{c['page_no']}] {c['chunk']}" for c in chosen.get("chunks", [])])