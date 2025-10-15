# features/insights/pipeline.py
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import streamlit as st
from sqlalchemy import text

from utils.db import get_engine
from .config import get_settings
from .fetchers import (
    company_publishes_annual_report,
    jina_find_recent_annual_pdf,
    jina_find_recent_quarterly_pdf,
    tavily_company_news_context,
    pdf_to_text,
)
from .chunk_embed import chunk_text, embed_chunks, retrieve
from .prompts import llm_extract_risk_factors, llm_extract_painpoints_table


@dataclass
class InsightResult:
    company: str
    pdf_url: str
    risk_factors_markdown: str
    table_markdown: str
    elapsed_seconds: float
    cached: bool = False
    source_type: str = "annual_pdf"  # "annual_pdf" | "quarterly_pdf" | "news"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _source_hash(company: str, pdf_url: str) -> str:
    h = hashlib.sha256()
    h.update((company or "").strip().lower().encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update((pdf_url or "").strip().encode("utf-8", "ignore"))
    return h.hexdigest()


def _select_cached_by_hash(h: str) -> Optional[InsightResult]:
    stg = get_settings()
    if not stg.enable_db_cache:
        return None

    eng = get_engine()
    sql = text("""
        SELECT company_name, pdf_url, risk_factors_md, table_md, run_seconds, created_at, source_type
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
            pdf_url=row["pdf_url"] or "",
            risk_factors_markdown=row["risk_factors_md"] or "",
            table_markdown=row["table_md"] or "",
            elapsed_seconds=float(row["run_seconds"] or 0.0),
            cached=True,
            source_type=row.get("source_type") or "annual_pdf",
        )


def _upsert_cache(res: InsightResult, by_user_email: str = "") -> None:
    stg = get_settings()
    if not stg.enable_db_cache:
        return

    h = _source_hash(res.company, res.pdf_url)
    eng = get_engine()
    sql = text("""
        INSERT INTO scout.company_insights
            (company_name, pdf_url, risk_factors_md, table_md, run_seconds, source_hash, by_user_email, source_type)
        VALUES
            (:company, :url, :risk, :tbl, :secs, :h, :who, :src)
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
            "src": res.source_type,
        })


def _infer_top_k(num_chunks: int) -> int:
    """
    Dynamic retrieval depth:
      - Base around corpus size, but clamp to [6, 18].
    """
    if num_chunks <= 0:
        return 6
    k = max(6, min(18, num_chunks // 8 + 6))
    return int(k)


def run_end_to_end(
    company_name: str,
    *,
    force_rerun: bool = False,
    by_user_email: str = "",
) -> Optional[InsightResult]:
    """
    Annual -> Quarterly -> News fallback.
    Dynamic Top-K and chunk size (no user controls).
    """
    start = time.time()
    stg = get_settings()

    company = (company_name or "").strip()
    if not company:
        st.error("Please provide a company name.")
        return None

    # 1) Check publishing (optional)
    if not stg.skip_tavily_check:
        with st.spinner("Checking if the company publishes annual reports…"):
            if not company_publishes_annual_report(company):
                st.info(f"Could not confirm that **{company}** publishes annual reports. Will try alternatives.")
                # continue (don't return)

    # 2) Find source: annual -> quarterly -> news
    source_type = "annual_pdf"
    with st.spinner("Finding the most relevant public source…"):
        pdf_url = jina_find_recent_annual_pdf(company)
        if not pdf_url:
            pdf_url = jina_find_recent_quarterly_pdf(company)
            if pdf_url:
                source_type = "quarterly_pdf"

    # 2.5) Cache check only for PDF sources
    if pdf_url:
        h = _source_hash(company, pdf_url)
        if not force_rerun:
            cached = _select_cached_by_hash(h)
            if cached:
                return cached

    # 3) If we have a PDF URL → normal pipeline; else use news context
    if pdf_url:
        with st.spinner("Downloading and parsing the report…"):
            full_text = pdf_to_text(pdf_url)
        if not full_text.strip():
            st.warning("Could not extract text from the report. Falling back to recent news context.")
            pdf_url = ""   # switch to news path
        else:
            # 4) Chunk + embed + retrieve (dynamic)
            with st.spinner("Building embeddings & retrieving the most relevant sections…"):
                # Dynamic chunk size heuristic: if very large, use smaller chunks
                # (we keep chunking default inside chunk_text; this step focuses on K)
                chunks = chunk_text(full_text)   # uses default from settings
                idx = embed_chunks(tuple(chunks))  # cached by content hash
                query = (
                    "Identify business risks, operational challenges, pain points, functions involved, "
                    "and regional performance or focus areas."
                )
                top_k = _infer_top_k(len(chunks))
                matches = retrieve(query, idx, k=top_k)
                digest = "\n\n".join([m[0] for m in matches])

            if not digest.strip():
                st.warning("Retrieval found no relevant text. Falling back to recent news context.")
                pdf_url = ""  # switch to news path
            else:
                # 5) LLM extraction
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
                    source_type=source_type,
                )
                try:
                    _upsert_cache(result, by_user_email=by_user_email)
                except Exception:
                    pass
                return result

    # --- News fallback path ---
    with st.spinner("Collecting recent news context…"):
        news_ctx = tavily_company_news_context(company)
    if not news_ctx.strip():
        st.error("No reports or relevant news context found.")
        return None

    with st.spinner("Generating insights from news…"):
        risk_md = llm_extract_risk_factors(company, news_ctx)
        table_md = llm_extract_painpoints_table(company, news_ctx)

    elapsed = round(time.time() - start, 2)
    return InsightResult(
        company=company,
        pdf_url="",  # none in news path
        risk_factors_markdown=risk_md,
        table_markdown=table_md,
        elapsed_seconds=elapsed,
        cached=False,
        source_type="news",
    )
