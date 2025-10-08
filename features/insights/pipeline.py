# features/insights/pipeline.py
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st
from sqlalchemy import text

from utils.db import get_engine
from .config import get_settings
from .fetchers import (
    company_publishes_annual_report,
    jina_find_recent_annual_pdf,
    pdf_to_text,
)
from .chunk_embed import chunk_text, embed_chunks, retrieve
from .prompts import llm_extract_risk_factors, llm_extract_painpoints_table


# =========================
# Data model for results
# =========================

@dataclass
class InsightResult:
    company: str
    pdf_url: str
    risk_factors_markdown: str
    table_markdown: str
    elapsed_seconds: float
    cached: bool = False  # True when served from DB cache

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================
# Helpers
# =========================

def _source_hash(company: str, pdf_url: str) -> str:
    """
    Stable hash used for caching/deduping: depends on normalized company & URL.
    """
    h = hashlib.sha256()
    h.update((company or "").strip().lower().encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update((pdf_url or "").strip().encode("utf-8", "ignore"))
    return h.hexdigest()


def _select_cached_by_hash(h: str) -> Optional[InsightResult]:
    """
    Read most recent cached result by source_hash from scout.company_insights.
    Returns InsightResult or None.
    """
    stg = get_settings()
    if not stg.enable_db_cache:
        return None

    eng = get_engine()
    sql = text("""
        SELECT company_name, pdf_url, risk_factors_md, table_md, run_seconds, created_at
        FROM scout.company_insights
        WHERE source_hash = :h
        ORDER BY created_at DESC
        LIMIT 1
    """)
    with eng.begin() as conn:
        row = conn.execute(sql, {"h": h}).mappings().first()
        if not row:
            return None
        return InsightResult(
            company=row["company_name"],
            pdf_url=row["pdf_url"],
            risk_factors_markdown=row["risk_factors_md"] or "",
            table_markdown=row["table_md"] or "",
            elapsed_seconds=float(row["run_seconds"] or 0.0),
            cached=True,
        )


def _upsert_cache(res: InsightResult, by_user_email: str = "") -> None:
    """
    Insert a new cache row into scout.company_insights.
    If table doesn't exist, this will raise — create it once:

    CREATE TABLE IF NOT EXISTS scout.company_insights (
        id BIGSERIAL PRIMARY KEY,
        company_name TEXT NOT NULL,
        pdf_url TEXT,
        risk_factors_md TEXT,
        table_md TEXT,
        run_seconds NUMERIC,
        created_at TIMESTAMPTZ DEFAULT now(),
        source_hash TEXT NOT NULL,
        by_user_email TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_company_insights_hash ON scout.company_insights(source_hash);
    """

    stg = get_settings()
    if not stg.enable_db_cache:
        return

    h = _source_hash(res.company, res.pdf_url)
    eng = get_engine()
    sql = text("""
        INSERT INTO scout.company_insights
            (company_name, pdf_url, risk_factors_md, table_md, run_seconds, source_hash, by_user_email)
        VALUES
            (:company, :url, :risk, :tbl, :secs, :h, :who)
    """)
    with eng.begin() as conn:
        conn.execute(sql, {
            "company": res.company,
            "url": res.pdf_url,
            "risk": res.risk_factors_markdown,
            "tbl": res.table_markdown,
            "secs": res.elapsed_seconds,
            "h": h,
            "who": by_user_email or "",
        })


# =========================
# Public pipeline
# =========================

def run_end_to_end(
    company_name: str,
    *,
    force_rerun: bool = False,
    top_k: Optional[int] = None,            # override retrieval depth if you like
    chunk_tokens: Optional[int] = None,     # override chunk size
    by_user_email: str = "",                # for cache lineage
) -> Optional[InsightResult]:
    """
    Orchestrates the full flow:
      Tavily YES/NO -> Jina URL -> PDF->text -> chunk/embed/retrieve -> LLM x2.
    Returns an InsightResult or None on failure.
    """
    start = time.time()
    stg = get_settings()

    company = (company_name or "").strip()
    if not company:
        st.error("Please provide a company name.")
        return None

    # 0) Cache check (by company + resolved URL). We need the URL to compute hash;
    #    but if we have a previous URL cached for this company, you could optionally
    #    look it up here by company. We'll keep it simple and compute hash after discovery.

    # 1) Publish check (skip if configured)
    if not stg.skip_tavily_check:
        with st.spinner("Checking if the company publishes annual reports…"):
            publishes = company_publishes_annual_report(company)
        if not publishes:
            st.warning(f"Could not confirm that **{company}** publishes annual reports.")
            return None

    # 2) Discover recent annual report URL via Jina
    with st.spinner("Finding the most recent annual report URL…"):
        pdf_url = jina_find_recent_annual_pdf(company)
    if not pdf_url:
        st.error("No annual report URL found.")
        return None

    # 2.5) Cache check by hash (now that we have the URL)
    h = _source_hash(company, pdf_url)
    if not force_rerun:
        cached = _select_cached_by_hash(h)
        if cached:
            return cached

    # 3) Download & extract text
    with st.spinner("Downloading and parsing the report…"):
        full_text = pdf_to_text(pdf_url)
    if not full_text.strip():
        st.error("Could not extract any text from the report URL.")
        return None

    # 4) Chunk + embed + retrieve
    with st.spinner("Building embeddings & retrieving the most relevant sections…"):
        chunks = chunk_text(full_text, max_tokens=chunk_tokens or stg.chunk_tokens)
        # Note: embed_chunks is cached by content-hash
        idx = embed_chunks(tuple(chunks))
        query = (
            "Identify business risks, operational challenges, pain points, functions involved, "
            "and regional performance or focus areas."
        )
        matches = retrieve(query, idx, k=top_k or stg.top_k)
        # Concatenate the top chunks (we ignore distances here)
        digest = "\n\n".join([m[0] for m in matches])

    if not digest.strip():
        st.error("Retrieval found no relevant text.")
        return None

    # 5) LLM extraction (risk bullets + insights table)
    with st.spinner("Generating risk factors…"):
        risk_md = llm_extract_risk_factors(company, digest)

    with st.spinner("Generating pain points table…"):
        table_md = llm_extract_painpoints_table(company, digest)

    elapsed = round(time.time() - start, 2)
    result = InsightResult(
        company=company,
        pdf_url=pdf_url,
        risk_factors_markdown=risk_md,
        table_markdown=table_md,
        elapsed_seconds=elapsed,
        cached=False,
    )

    # 6) Persist cache (best-effort)
    try:
        _upsert_cache(result, by_user_email=by_user_email)
    except Exception:
        # Non-fatal: rendering continues even if caching fails.
        pass

    return result
