# features/insights/case_retriever.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import re
import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from utils.db import get_engine
from features.insights.case_vectorstore import embed_query

# Try to ensure pgvector type adaptation for psycopg2
try:
    from pgvector.psycopg2 import register_vector  # type: ignore
except Exception:
    register_vector = None  # it's ok if unavailable; server-side ::vector cast may still work

# ------------------------------
# SQL fragments
# ------------------------------

# Persona resolution from centralize_db:
# Rule: try email_address; if absent, fallback to client_name ONLY (no account).
SQL_PERSONA_RESOLVE = """
SELECT
  sr_no                 AS persona_id,
  account,
  client_name           AS persona_name,
  client_designation,
  working_group,
  business_unit,
  service_line,
  seniority_level,
  reporting_manager_designation,
  location,
  email_address,
  linkedin_url,
  internal_research,
  external_research,
  personalization_notes
FROM scout.centralize_db
WHERE {where_clause}
LIMIT 1
"""

# Candidate chunks: keep JSONB for metadata, vector ANN by <=>, with section filter and optional function filter
SQL_CHUNKS_FILTERED = """
SELECT id, doc_id, chunk_ix, section, text, metadata, embedding
FROM insights.case_chunks
WHERE (:section_any = TRUE OR section = ANY(:sections))
  AND (:use_fn = FALSE OR (metadata->>'business_function') = ANY(:fn_labels))
ORDER BY embedding <=> (:qvec)::vector
LIMIT :k
"""

# Unfiltered candidate search (fallback)
SQL_CHUNKS_ANY = """
SELECT id, doc_id, chunk_ix, section, text, metadata, embedding
FROM insights.case_chunks
WHERE (:section_any = TRUE OR section = ANY(:sections))
ORDER BY embedding <=> (:qvec)::vector
LIMIT :k
"""

# Distinct case functions universe (for mapping)
SQL_DISTINCT_CASE_FUNCTIONS = """
SELECT DISTINCT business_function
FROM insights.case_facts
WHERE business_function IS NOT NULL AND length(business_function) > 0
ORDER BY 1
"""

def _ensure_adapter(conn) -> None:
    """Register pgvector adapter if available (safe to call repeatedly)."""
    if register_vector is not None:
        try:
            register_vector(conn.connection)  # raw psycopg2 connection
        except Exception:
            pass

def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-8)
    return v / n

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((_norm(a) @ _norm(b)).item())

def _regex_hit(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in (patterns or []))

def _as_vec(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        return np.array(list(x), dtype=np.float32)
    except Exception:
        return None

# ------------------------------
# Public helpers
# ------------------------------

def resolve_persona(engine: Optional[Engine], *, email: Optional[str], client_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolve the persona row from centralize_db with rule: email first; else client_name only."""
    eng = engine or get_engine()
    with eng.begin() as conn:
        _ensure_adapter(conn)
        if email:
            r = conn.execute(
                text(SQL_PERSONA_RESOLVE.format(where_clause="email_address = :e")),
                {"e": email},
            ).mappings().first()
            if r:
                return dict(r)
        if client_name:
            r = conn.execute(
                text(SQL_PERSONA_RESOLVE.format(where_clause="client_name = :n")),
                {"n": client_name},
            ).mappings().first()
            if r:
                return dict(r)
    return None

def get_distinct_case_functions(engine: Optional[Engine]) -> List[str]:
    eng = engine or get_engine()
    with eng.begin() as conn:
        rows = conn.execute(text(SQL_DISTINCT_CASE_FUNCTIONS)).fetchall()
    return [r[0] for r in rows]

def embed_strings(strings: List[str]) -> List[List[float]]:
    from utils.rag_env import get_embeddings
    emb = get_embeddings()
    return emb.embed_documents(strings)

def retrieve_by_functions(
    engine: Optional[Engine],
    *,
    query: str,
    function_labels: List[str],
    sections: Optional[List[str]] = None,
    strict_first_k: int = 24,
    fallback_k: int = 24,
) -> List[Dict[str, Any]]:
    """Two-stage retrieval: function-filtered first, then fallback to unfiltered if too few."""
    eng = engine or get_engine()
    sections = sections or ["impact", "kpi"]
    qvec = embed_query(query)

    rows: List[Dict[str, Any]] = []
    with eng.begin() as conn:
        _ensure_adapter(conn)

        # Stage 1: function-biased
        r1 = conn.execute(
            text(SQL_CHUNKS_FILTERED),
            {
                "sections": sections,
                "section_any": False,
                "use_fn": bool(function_labels),
                "fn_labels": function_labels if function_labels else [],
                "qvec": qvec,  # adapted to vector by pgvector if registered
                "k": strict_first_k,
            },
        ).mappings().all()
        rows.extend(r1)

        # Stage 2: fallback if too few
        if len(rows) < 12:
            r2 = conn.execute(
                text(SQL_CHUNKS_ANY),
                {
                    "sections": sections,
                    "section_any": False,
                    "qvec": qvec,
                    "k": fallback_k,
                },
            ).mappings().all()
            rows = r2  # replace with wider set

    # Convert to plain dicts
    return [dict(r) for r in rows]

# ================================================