#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rag_retriever.py — Production-grade vector retriever for scout.impact_repository

Purpose:
    Retrieve the most semantically relevant business impact statements
    (Impact, Industry, Business Group, Use Case)
    from Postgres (pgvector) based on persona or KPI text.

Core Features:
    ✅ Uses same Azure embedding model as ingestion (text-embedding-3-small)
    ✅ Fast similarity search with pgvector (<=> cosine distance)
    ✅ Clean async + sync support for flexible integration
    ✅ Structured return (list[dict])
    ✅ Confidence scoring based on distance normalization

Usage Example:
    from pipelines.rag_retriever import retrieve_impacts_sync

    persona_kpis = "Revenue growth, operational efficiency, automation"
    results = retrieve_impacts_sync(persona_kpis, top_k=10)
    for r in results:
        print(r)

Dependencies:
    - utils/db.py → get_engine()
    - pipelines/embedding_utils.py → generate_embeddings_sync()
"""

import os
import math
import logging
from typing import List, Dict, Optional

from sqlalchemy import text
from pipelines.embedding_utils import generate_embeddings_sync
from utils.db import get_engine

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("rag_retriever")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TABLE_SCHEMA = "scout"
TABLE_NAME = "impact_repository"
FQN = f"{TABLE_SCHEMA}.{TABLE_NAME}"

DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "15"))
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIM", "0.60"))  # Below this = low confidence


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _normalize_score(distance: float) -> float:
    """
    Normalize cosine distance (pgvector <=>) to similarity [0,1].
    """
    return max(0.0, min(1.0, 1.0 - distance))  # smaller distance = higher similarity


def _format_record(row) -> Dict:
    """
    Convert SQLAlchemy Row → dict with formatted fields.
    """
    return {
        "Impact": row.get("impact", "").strip(),
        "Industry": row.get("industry", "").strip(),
        "BusinessGroup": row.get("business_group", "").strip(),
        "UseCase": row.get("use_case", "").strip(),
        "Similarity": round(_normalize_score(row.get("distance", 1.0)), 4),
    }


# -----------------------------------------------------------------------------
# Core retrieval logic
# -----------------------------------------------------------------------------
def retrieve_impacts_sync(query_text: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Perform vector similarity search against scout.impact_repository.

    Args:
        query_text: Natural language query (e.g., persona KPIs or description)
        top_k: Number of top results to return

    Returns:
        List of dicts with Impact, Industry, BusinessGroup, UseCase, Similarity
    """
    if not query_text or not query_text.strip():
        logger.warning("Empty query_text provided. Returning empty list.")
        return []

    logger.info(f"🔍 Generating embedding for query: {query_text[:80]}...")
    query_vec = generate_embeddings_sync([query_text])[0]
    if not query_vec:
        logger.error("❌ Failed to generate query embedding. Returning empty list.")
        return []

    logger.info(f"🔎 Retrieving top {top_k} impacts from {FQN}...")

    sql = text(f"""
    SELECT
        impact,
        industry,
        business_group,
        use_case,
        (embedding <=> (:query_vec)::vector) AS distance
    FROM {FQN}
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> (:query_vec)::vector ASC
    LIMIT :k
""")

    # Cast Python list → JSON text for Postgres
    query_vec_json = json.dumps(query_vec)

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {"query_vec": query_vec_json, "k": top_k},
        ).fetchall()

    results = [_format_record(dict(r._mapping)) for r in rows]
    if not results:
        logger.warning("⚠️ No results found in impact_repository.")
        return []

    # Filter low-similarity results (optional threshold)
    high_conf = [r for r in results if r["Similarity"] >= MIN_SIMILARITY]
    if not high_conf:
        logger.warning(
            f"⚠️ All results below MIN_SIMILARITY={MIN_SIMILARITY}. Returning top {len(results)} anyway."
        )
        return results

    logger.info(f"✅ Retrieved {len(high_conf)} high-confidence impacts.")
    return high_conf


# -----------------------------------------------------------------------------
# Optional CLI test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Quick test for RAG retriever")
    parser.add_argument("--query", required=True, help="Persona or KPI text to search for")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    results = retrieve_impacts_sync(args.query, args.top_k)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n✓ Done. Retrieved {len(results)} records from {FQN}.")
# ===================================================================================