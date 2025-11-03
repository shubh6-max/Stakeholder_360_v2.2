# features/insights/batch_core.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import time
import os

from sqlalchemy import text
from sqlalchemy.engine import Engine

from utils.db import get_engine
from utils.rag_db import ensure_rag_schema
from features.insights.store import (
    _persona_key as persona_key,
    get_cached_kpis,
    upsert_kpis,
)
from features.insights.store import retrieve_top_k_chunks  # reuse query-time retrieval
from features.insights.fetch_annual import fetch_or_load_annual_report
from features.insights.chunk_embed_langchain import chunk_and_embed_document
from features.insights.chains import generate_top_kpis

# ---------- helpers (copied from retrieve.py, no streamlit) ----------
def _get_latest_document_id_for_company(engine, company: str) -> Optional[int]:
    sql = text("""
        SELECT id FROM rag.documents WHERE company_name = :c
        ORDER BY created_at DESC LIMIT 1
    """)
    with engine.begin() as conn:
        row = conn.execute(sql, {"c": company}).first()
        return int(row[0]) if row else None

def _get_chunk_count_for_document(engine, document_id: int) -> int:
    sql = text("SELECT COUNT(*) FROM rag.chunks WHERE document_id = :d")
    with engine.begin() as conn:
        val = conn.execute(sql, {"d": document_id}).scalar()
        return int(val or 0)

def _ensure_company_indexed(engine: Engine, company: str) -> Optional[int]:
    company = (company or "").strip()
    if not company:
        return None
    ensure_rag_schema(engine)

    doc_id = _get_latest_document_id_for_company(engine, company)
    if doc_id and _get_chunk_count_for_document(engine, doc_id) > 0:
        return doc_id

    # DB-first fetch
    fetch_res = fetch_or_load_annual_report(engine, company)
    if fetch_res.status == "absent" or not fetch_res.document_id:
        return None

    # Ensure embeddings
    chunk_and_embed_document(engine=engine, document_id=fetch_res.document_id, company=company)
    return fetch_res.document_id if _get_chunk_count_for_document(engine, fetch_res.document_id) > 0 else None

def _persona_info_for_key(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": (row.get("client_designation") or "").strip(),
        "working_group": (row.get("working_group") or "").strip(),
        "business_unit": (row.get("business_unit") or "").strip(),
        "service_line": (row.get("service_line") or "").strip(),
        "subsidiary": (row.get("subsidiary") or "").strip(),
        "region_or_location": (row.get("location") or "").strip(),
        "email": (row.get("email_address") or "").strip(),
        "manager_title": (row.get("reporting_manager_designation") or "").strip(),
    }

def _adapt_for_ui(parsed: Dict[str, Any], *, company: str, persona_name: str,
                  row: Dict[str, Any], k_used: int) -> Dict[str, Any]:
    out = {
        "company": company,
        "persona_name": persona_name,
        "persona_title": (row.get("client_designation") or "").strip(),
        "working_group": (row.get("working_group") or "").strip(),
        "business_unit": (row.get("business_unit") or "").strip(),
        "service_line": (row.get("service_line") or "").strip(),
        "k_used": k_used,
        "kpis": [],
    }
    for k in (parsed.get("kpis") or [])[:2]:
        title = (k.get("name") or "").strip()
        why = (k.get("why") or "").strip()
        owner = (k.get("suggested_owner_function") or "").strip()
        evidence = " • ".join([e for e in (k.get("backing_evidence") or []) if (e or "").strip()])[:500]
        out["kpis"].append({
            "title": title or "Untitled KPI",
            "why_it_matters": why or (f"Relevant for {owner}" if owner else "—"),
            "how_to_measure": evidence or (f"Measured by {owner}" if owner else "—"),
            "suggested_initiatives": [],
            "sources": [],
        })
    return out

# ---------- public entrypoint for automation ----------
def compute_kpis_for_persona_cached(
    engine: Optional[Engine],
    *,
    company: str,
    persona_row: Dict[str, Any],
    top_k: int = 8,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Headless: ensure index, read/update rag.persona_kpi_cache, return UI payload.
    """
    eng = engine or get_engine()
    ensure_rag_schema(eng)

    company = (company or "").strip()
    if not company:
        raise RuntimeError("company is required")

    # ensure index for company
    doc_id = _ensure_company_indexed(eng, company)
    if not doc_id:
        raise RuntimeError(f"No indexed content for company '{company}'")

    # Persona key / cache check
    pinfo = _persona_info_for_key(persona_row)
    pk = persona_key(company, pinfo)

    if use_cache:
        cached = get_cached_kpis(eng, company, pk)
        if cached and (cached.get("kpis_json") or {}).get("kpis"):
            return cached["kpis_json"]

    # Retrieval query
    retrieval_query = (
        "From the company's annual report, surface business priorities, risks, operational challenges, "
        "functions involved and regional focus areas, suitable to derive KPIs for a stakeholder."
    )

    matches = retrieve_top_k_chunks(
        engine=eng,
        document_id=doc_id,
        query=retrieval_query,
        top_k=top_k,
        distance_metric="cosine",
    )
    if not matches:
        raise RuntimeError("No retrieval matches for KPI selection")

    snippets: List[str] = [ (m["text"] if isinstance(m, dict) else (m[0] if isinstance(m, (list, tuple)) else str(m)))
                            for m in matches ]
    digest = "\n\n".join(snippets)

    # LLM → exactly 2 KPIs
    persona_name = (persona_row.get("client_name") or persona_row.get("name") or "Unknown Persona").strip()
    parsed, _raw = generate_top_kpis(
        company=company,
        persona_name=persona_name,
        persona_info=pinfo,
        context=digest,
    )

    ui_payload = _adapt_for_ui(parsed, company=company, persona_name=persona_name, row=persona_row, k_used=top_k)

    # Upsert cache
    upsert_kpis(eng, company, pk, pinfo, ui_payload, top_k)
    return ui_payload