# features/insights/pipeline.py
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

import streamlit as st

from utils.db import get_engine
from utils.rag_db import ensure_rag_schema  # safety
from .fetch_annual import fetch_or_load_annual_report, AnnualFetchResult
from .chunk_embed_langchain import chunk_and_embed_document
from .store import (
    get_by_source_hash,
    upsert_insight,
    compute_source_hash,
    retrieve_top_k_chunks,
)
from .chains import (
    generate_risk_factors,
    generate_painpoints_table,
    summarize_risk_md,
    summarize_painpoints_table_md,
)

from .store import retrieve_top_k_chunks, upsert_insight  # add upsert_insight


@dataclass
class InsightResult:
    company: str
    pdf_url: str
    risk_factors_markdown: str
    table_markdown: str
    elapsed_seconds: float
    cached: bool = False           # surfaced to UI
    source_type: str = "annual_pdf"
    risk_factors_markdown_short: str = ""
    table_markdown_short: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _infer_top_k(num_chunks: int) -> int:
    """Choose retrieval depth based on corpus size."""
    if num_chunks <= 0:
        return 12
    return int(max(6, min(18, num_chunks // 8 + 6)))


def _ensure_text(items: List[Any]) -> List[str]:
    """Normalize retrieval rows → List[str] text only."""
    out: List[str] = []
    for it in items:
        if isinstance(it, (tuple, list)) and len(it) >= 1:
            out.append(str(it[0]))
        elif isinstance(it, dict):
            out.append(str(it.get("text", "")))
        else:
            out.append(str(it))
    return out


def _has_chunks(engine, document_id: int) -> bool:
    """Fast existence check to avoid re-embedding."""
    from sqlalchemy import text
    with engine.begin() as conn:
        val = conn.execute(text("SELECT 1 FROM rag.chunks WHERE document_id = :d LIMIT 1"), {"d": int(document_id)}).scalar()
        return bool(val)


def _count_chunks(engine, document_id: int) -> int:
    from sqlalchemy import text
    with engine.begin() as conn:
        val = conn.execute(text("SELECT COUNT(*) FROM rag.chunks WHERE document_id = :d"), {"d": int(document_id)}).scalar()
        return int(val or 0)


def run_end_to_end(
    company_name: str,
    *,
    force_rerun: bool = False,
    by_user_email: str = "",
) -> Optional[InsightResult]:
    """
    DB-first RAG:

      1) Ensure annual report exists in DB (rag.company_sources + rag.documents).
      2) Ensure embeddings exist ONCE (rag.chunks); skip if already present.
      3) Try insights cache (rag.company_insights) by (company, pdf_url) hash; reuse unless force_rerun=True.
      4) If not cached (or force), retrieve top-K chunks and generate:
         - pain points table (markdown)
         - risk factors (markdown)
         Then upsert into rag.company_insights.
    """
    start = time.time()
    company = (company_name or "").strip()
    if not company:
        st.error("Please provide a company name.")
        return None

    engine = get_engine()
    ensure_rag_schema(engine)  # defensive

    # --- 1) DB-first fetch/load of annual report ---
    with st.spinner("Checking DB / fetching annual report…"):
        fetch_res: AnnualFetchResult = fetch_or_load_annual_report(engine, company)

    if fetch_res.status == "absent":
        st.info(f"No annual report found for **{company}** (marked absent).")
        return None
    if fetch_res.status in ("error",) or not fetch_res.document_id:
        st.error(fetch_res.note or "Failed to fetch or store the annual report.")
        return None

    pdf_url = fetch_res.url or ""
    source_type = "annual_pdf"

    # --- 2) Ensure embeddings exist once (skip if already present) ---
    if not _has_chunks(engine, fetch_res.document_id):
        with st.spinner("Preparing chunks & embeddings (first time may take a bit)…"):
            # Only embed if no chunks exist for this document.
            chunk_and_embed_document(
                engine=engine,
                document_id=fetch_res.document_id,
                company=company,
            )
    else:
        # Be quiet—just a tiny caption to confirm we skipped heavy work
        st.caption("Chunks already present — skipping embedding.")

    total_chunks = _count_chunks(engine, fetch_res.document_id)
    top_k = _infer_top_k(total_chunks)

    # --- 3) Try insights cache (by source hash) ---
    s_hash = compute_source_hash(company, pdf_url)
    cached = None
    if not force_rerun:
        cached = get_by_source_hash(engine, company, pdf_url)

    if cached and not force_rerun:
        elapsed = round(time.time() - start, 2)
        return InsightResult(
            company=company,
            pdf_url=pdf_url,
            risk_factors_markdown=cached.risk_factors_md or "",
            table_markdown=cached.table_md or "",
            elapsed_seconds=elapsed,
            cached=True,
            source_type=source_type,
            risk_factors_markdown_short="",  # you can compact in UI if needed
            table_markdown_short="",
        )

    # --- 4) Retrieval + LLM generation (fresh, then cache) ---
    retrieval_query = (
        "Identify business risks, operational challenges, stakeholder-relevant pain points, "
        "functions involved, and regional performance or focus areas."
    )

    with st.spinner(f"Retrieving top-{top_k} relevant sections…"):
        retrieved = retrieve_top_k_chunks(
            engine=engine,
            document_id=fetch_res.document_id,
            query=retrieval_query,          # retrieve.py accepts query|query_text
            top_k=top_k,
            distance_metric="cosine",
        )
        snippets: List[str] = _ensure_text(retrieved)
        digest = "\n\n".join(snippets).strip()

    if not digest:
        st.warning("Retrieval returned no relevant text for summarization.")
        return None

    # LLM generation (full)
    with st.spinner("Generating pain points table…"):
        table_full_md: str = generate_painpoints_table(company=company, context=digest)

    with st.spinner("Generating risk factors…"):
        risks_full_md: str = generate_risk_factors(company=company, context=digest)

    # Optional compaction (kept behind settings)
    risks_short_md = ""
    table_short_md = ""
    try:
        stg = __import__("features.insights.config", fromlist=["get_settings"]).get_settings()
        if getattr(stg, "summarize_outputs", False):
            risks_short_md = summarize_risk_md(company, risks_full_md, max_bullets=getattr(stg, "risks_max_bullets", 6))
            table_short_md = summarize_painpoints_table_md(company, table_full_md, max_rows=getattr(stg, "table_max_rows", 5))
    except Exception:
        pass  # non-fatal

    

    elapsed = round(time.time() - start, 2)

    # --- Save to cache table so future personas reuse it ---
    try:
        upsert_insight(
            engine=get_engine(),
            company_name=company,
            pdf_url=fetch_res.url or "",
            source_type="annual_pdf",
            risk_factors_md=risks_full_md or "",
            table_md=table_full_md or "",
            run_seconds=round(time.time() - start, 2),
            by_user_email=by_user_email or "",
        )
    except Exception:
        # non-fatal; continue
        pass

    return InsightResult(
        company=company,
        pdf_url=pdf_url,
        risk_factors_markdown=risks_full_md or "",
        table_markdown=table_full_md or "",
        elapsed_seconds=elapsed,
        cached=False,
        source_type=source_type,
        risk_factors_markdown_short=risks_short_md or "",
        table_markdown_short=table_short_md or "",
    )
