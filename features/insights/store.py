# features/insights/store.py
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import requests
from sqlalchemy import text
from sqlalchemy.engine import Engine


# ---------- Dataclass for cached insights ----------

@dataclass
class InsightRecord:
    id: int
    company_name: str
    pdf_url: str
    source_type: str         # "annual_pdf" | "quarterly_pdf" | "news"
    risk_factors_md: str
    table_md: str
    run_seconds: float
    source_hash: str
    by_user_email: str
    created_at: str          # ISO-ish string from DB


# ---------- DDL for the insights cache (idempotent) ----------

DDL_INSIGHTS = """
CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.company_insights (
    id               BIGSERIAL PRIMARY KEY,
    company_name     TEXT NOT NULL,
    pdf_url          TEXT,
    source_type      TEXT DEFAULT 'annual_pdf',
    risk_factors_md  TEXT,
    table_md         TEXT,
    run_seconds      DOUBLE PRECISION DEFAULT 0,
    source_hash      TEXT NOT NULL,
    by_user_email    TEXT,
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ci_company ON rag.company_insights (company_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ci_source_hash ON rag.company_insights (source_hash);
"""


def ensure_company_insights_table(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(DDL_INSIGHTS))


# ---------- Small helpers ----------

def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def compute_source_hash(company: str, pdf_url: str) -> str:
    """
    Deterministic hash used to dedupe cache entries.
    For news (no URL), pdf_url can be empty string.
    """
    h = hashlib.sha256()
    h.update(_norm(company).lower().encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update(_norm(pdf_url).encode("utf-8", "ignore"))
    return h.hexdigest()


# ---------- CRUD for cached insights ----------

def insert_insight(
    engine: Engine,
    *,
    company_name: str,
    pdf_url: str,
    source_type: str,
    risk_factors_md: str,
    table_md: str,
    run_seconds: float,
    by_user_email: str = "",
) -> InsightRecord:
    ensure_company_insights_table(engine)
    s_hash = compute_source_hash(company_name, pdf_url)

    sql = text("""
        INSERT INTO rag.company_insights
            (company_name, pdf_url, source_type, risk_factors_md, table_md, run_seconds, source_hash, by_user_email)
        VALUES
            (:company, :url, :stype, :risk, :tbl, :secs, :shash, :who)
        RETURNING id, company_name, pdf_url, source_type, risk_factors_md, table_md,
                  run_seconds, source_hash, by_user_email, created_at
    """)
    with engine.begin() as conn:
        row = conn.execute(sql, {
            "company": _norm(company_name),
            "url": _norm(pdf_url),
            "stype": _norm(source_type) or "annual_pdf",
            "risk": risk_factors_md or "",
            "tbl": table_md or "",
            "secs": float(run_seconds or 0.0),
            "shash": s_hash,
            "who": _norm(by_user_email),
        }).mappings().first()

    return InsightRecord(**row)  # type: ignore[arg-type]


def upsert_insight(
    engine: Engine,
    *,
    company_name: str,
    pdf_url: str,
    source_type: str,
    risk_factors_md: str,
    table_md: str,
    run_seconds: float,
    by_user_email: str = "",
) -> InsightRecord:
    ensure_company_insights_table(engine)
    s_hash = compute_source_hash(company_name, pdf_url)

    sql = text("""
        INSERT INTO rag.company_insights
            (company_name, pdf_url, source_type, risk_factors_md, table_md, run_seconds, source_hash, by_user_email)
        VALUES
            (:company, :url, :stype, :risk, :tbl, :secs, :shash, :who)
        ON CONFLICT (source_hash) DO UPDATE SET
            company_name    = EXCLUDED.company_name,
            pdf_url         = EXCLUDED.pdf_url,
            source_type     = EXCLUDED.source_type,
            risk_factors_md = EXCLUDED.risk_factors_md,
            table_md        = EXCLUDED.table_md,
            run_seconds     = EXCLUDED.run_seconds,
            by_user_email   = EXCLUDED.by_user_email
        RETURNING id, company_name, pdf_url, source_type, risk_factors_md, table_md,
                  run_seconds, source_hash, by_user_email, created_at
    """)
    with engine.begin() as conn:
        row = conn.execute(sql, {
            "company": _norm(company_name),
            "url": _norm(pdf_url),
            "stype": _norm(source_type) or "annual_pdf",
            "risk": risk_factors_md or "",
            "tbl": table_md or "",
            "secs": float(run_seconds or 0.0),
            "shash": s_hash,
            "who": _norm(by_user_email),
        }).mappings().first()

    return InsightRecord(**row)  # type: ignore[arg-type]


def get_latest_by_company(engine: Engine, company_name: str) -> Optional[InsightRecord]:
    ensure_company_insights_table(engine)
    sql = text("""
        SELECT id, company_name, pdf_url, source_type, risk_factors_md, table_md,
               run_seconds, source_hash, by_user_email, created_at
        FROM rag.company_insights
        WHERE company_name = :company
        ORDER BY created_at DESC
        LIMIT 1
    """)
    with engine.begin() as conn:
        row = conn.execute(sql, {"company": _norm(company_name)}).mappings().first()
        return InsightRecord(**row) if row else None  # type: ignore[arg-type]


def get_by_source_hash(engine: Engine, company_name: str, pdf_url: str) -> Optional[InsightRecord]:
    ensure_company_insights_table(engine)
    s_hash = compute_source_hash(company_name, pdf_url)
    sql = text("""
        SELECT id, company_name, pdf_url, source_type, risk_factors_md, table_md,
               run_seconds, source_hash, by_user_email, created_at
        FROM rag.company_insights
        WHERE source_hash = :shash
        LIMIT 1
    """)
    with engine.begin() as conn:
        row = conn.execute(sql, {"shash": s_hash}).mappings().first()
        return InsightRecord(**row) if row else None  # type: ignore[arg-type]


# ---------- Query-time retrieval over pgvector chunks ----------

def _embed_query_azure(query_text: str) -> List[float]:
    """
    Minimal REST call to Azure OpenAI embeddings.
    Requires env:
      - AZURE_ENDPOINT
      - AZURE_API_KEY
      - AZURE_EMBED_DEPLOYMENT
      - AZURE_EMBED_VERSION
    """
    endpoint = os.getenv("AZURE_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_API_KEY", "")
    deployment = os.getenv("AZURE_EMBED_DEPLOYMENT", "")
    api_version = os.getenv("AZURE_EMBED_VERSION", "2023-05-15")

    if not (endpoint and api_key and deployment):
        raise RuntimeError("Azure embedding environment is not fully configured.")

    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    payload = {"input": query_text}

    r = requests.post(url, headers=headers, json=payload, timeout=40)
    r.raise_for_status()
    data = r.json()
    return data["data"][0]["embedding"]  # type: ignore[index]


def _vector_literal(v: List[float]) -> str:
    """
    Render a pgvector literal like: [0.12, 0.34, ...]
    """
    # Shorten floats a bit to keep SQL short
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def retrieve_top_k_chunks(
    engine: Engine,
    *,
    document_id: int,
    query_text: Optional[str] = None,
    query: Optional[str] = None,          # <- accept the alias used by caller
    top_k: int = 12,
    distance_metric: str = "cosine",      # <- new: choose pgvector operator
) -> List[Dict[str, Any]]:
    """
    Embed the query via Azure, then run a pgvector similarity search over rag.chunks
    for a given document_id. Returns a list of dicts:
      { "chunk_id", "chunk_index", "text", "distance" }
    """

    # normalize query input
    qtext = (query_text or query or "").strip()
    if not qtext:
        return []

    # choose operator based on distance metric
    dm = (distance_metric or "cosine").lower()
    if dm in ("cosine", "cos"):
        op = "<=>"
    elif dm in ("l2", "euclidean", "euclid"):
        op = "<->"
    elif dm in ("inner", "ip", "inner_product", "max_inner_product"):
        op = "<#>"
    else:
        # fallback safely to cosine
        op = "<=>"

    # 1) Embed the query
    q_emb = _embed_query_azure(qtext)
    vec_sql = _vector_literal(q_emb)  # e.g. "[0.12,0.34,...]"

    # use bind for the vector literal; operator must be inlined token (not a bind)
    dim = int(os.getenv("RAG_EMBED_DIM", "1536"))
    sql = text(f"""
        SELECT id AS chunk_id,
               chunk_idx AS chunk_index,
               text,
               (embedding {op} CAST(:vec AS vector({dim}))) AS distance
        FROM rag.chunks
        WHERE document_id = :doc
        ORDER BY embedding {op} CAST(:vec AS vector({dim})) ASC
        LIMIT :k
    """)

    with engine.begin() as conn:
        rows = conn.execute(sql, {
            "doc": int(document_id),
            "k": int(top_k),
            "vec": vec_sql,
        }).mappings().all()

    return [dict(r) for r in rows]
