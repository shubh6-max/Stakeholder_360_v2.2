# features/insights/retrieve.py
from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
import streamlit as st
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from utils.db import get_engine
from utils.rag_db import ensure_rag_schema
from .fetch_annual import fetch_or_load_annual_report, AnnualFetchResult
from .chunk_embed_langchain import chunk_and_embed_document
from .store import (
    retrieve_top_k_chunks,
    get_cached_kpis,
    upsert_kpis,
    _persona_key as persona_key,   # reuse the same hashing logic as store.py
)
from .chains import generate_top_kpis


# -----------------------------
# DB helpers (with auto-ensure)
# -----------------------------

def _get_latest_document_id_for_company(engine, company: str) -> Optional[int]:
    sql = text("""
        SELECT id
        FROM rag.documents
        WHERE company_name = :c
        ORDER BY created_at DESC
        LIMIT 1
    """)
    with engine.begin() as conn:
        try:
            row = conn.execute(sql, {"c": company}).first()
        except ProgrammingError:
            # If tables don't exist yet, create them and retry once
            ensure_rag_schema(engine)
            row = conn.execute(sql, {"c": company}).first()
        return int(row[0]) if row else None


def _get_chunk_count_for_document(engine, document_id: int) -> int:
    sql = text("SELECT COUNT(*) FROM rag.chunks WHERE document_id = :d")
    with engine.begin() as conn:
        try:
            val = conn.execute(sql, {"d": document_id}).scalar()
        except ProgrammingError:
            ensure_rag_schema(engine)
            val = conn.execute(sql, {"d": document_id}).scalar()
        return int(val or 0)


# -----------------------------
# Ensure company is indexed
# -----------------------------

def ensure_company_indexed(engine, company: str) -> Optional[int]:
    """
    Returns a document_id that has chunk embeddings ready.
    If doc/chunks are missing, this will:
      - ensure schema/tables exist
      - fetch/store the annual report (DB-first)
      - chunk & embed it
    """
    company = (company or "").strip()
    if not company:
        return None

    # ✅ Ensure schema & extensions exist BEFORE any SELECTs
    ensure_rag_schema(engine)

    # 1) See if we already have a doc with chunks
    doc_id = _get_latest_document_id_for_company(engine, company)
    if doc_id:
        if _get_chunk_count_for_document(engine, doc_id) > 0:
            return doc_id  # already indexed
        # doc exists but no chunks → embed now
        chunk_and_embed_document(engine=engine, document_id=doc_id, company=company)
        if _get_chunk_count_for_document(engine, doc_id) > 0:
            return doc_id

    # 2) DB-first fetch/load of annual report (creates rag.company_sources + rag.documents if missing)
    fetch_res: AnnualFetchResult = fetch_or_load_annual_report(engine, company)

    if fetch_res.status == "absent":
        return None
    if not fetch_res.document_id:
        return None

    # 3) Ensure embeddings for this document
    chunk_and_embed_document(engine=engine, document_id=fetch_res.document_id, company=company)
    if _get_chunk_count_for_document(engine, fetch_res.document_id) > 0:
        return fetch_res.document_id

    return None


# -----------------------------
# Persona helpers
# -----------------------------

def _persona_summary_from_row(row: Dict[str, Any] | Any) -> Tuple[str, Dict[str, Any]]:
    def _get(k, default=""):
        if isinstance(row, dict):
            return row.get(k, default)
        try:
            return getattr(row, k) if hasattr(row, k) else row.get(k, default)
        except Exception:
            return default

    persona_name = str(_get("client_name", "") or _get("name", "")).strip()
    info = {
        "title": _get("client_designation", ""),
        "working_group": _get("working_group", ""),
        "business_unit": _get("business_unit", ""),
        "service_line": _get("service_line", ""),
        "subsidiary": _get("subsidiary", ""),
        "region_or_location": _get("location", ""),
        "email": _get("email_address", ""),
        # NEW: include manager title to specialize KPIs
        "manager_title": _get("reporting_manager_designation", ""),
    }
    return persona_name or "Unknown Persona", info


def _adapt_for_ui(parsed: Dict[str, Any], *, company: str, persona_name: str,
                  row: Dict[str, Any], k_used: int) -> Dict[str, Any]:
    """Map KPIResponse -> UI schema expected by components.kpi_view (top 2)."""
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
        evidence = " • ".join([e for e in (k.get("backing_evidence") or []) if isinstance(e, str) and e.strip()])[:500]

        out["kpis"].append({
            "title": title or "Untitled KPI",
            "why_it_matters": why or (f"Relevant for {owner}" if owner else "—"),
            "how_to_measure": evidence or (f"Measured by {owner}" if owner else "—"),
            "suggested_initiatives": [],
            "sources": [],
        })
    return out


# -----------------------------
# Persona → retrieval → KPI generation (with cache)
# -----------------------------

def run_kpis_for_persona(company: str, persona_row: Dict[str, Any] | Any, *, top_k: int = 6) -> Dict[str, Any]:
    """
    Flow:
      0) Build persona key and check cache → return if hit
      1) Ensure company's annual report is indexed (fetch + embed only once)
      2) Retrieve top-K relevant chunks for KPI query (fast IVFFLAT)
      3) Generate exactly two KPIs tailored to the persona
      4) Adapt to UI schema and persist to persona cache → return
    """
    company = (company or "").strip()
    if not company:
        raise RuntimeError("Company is required for KPI generation.")

    engine = get_engine()

    # Persona key + cache check (FAST PATH)
    persona_name, persona_info = _persona_summary_from_row(persona_row)
    pkey = persona_key(company, persona_info)
    with st.spinner("Building KPI context from the latest annual report…"):
        cached = get_cached_kpis(engine, company, pkey)
    if cached and cached.get("kpis_json"):
        # Already adapted to UI schema in cache; return as-is
        payload = cached["kpis_json"]
        # keep k_used if present (handy to show in UI)
        if cached.get("k_used") is not None:
            payload["k_used"] = cached["k_used"]
        return payload

    # 🔑 Ensure we have indexed content (will fetch/embed if missing)
    with st.spinner(f"Ensuring index for {company}…"):
        document_id = ensure_company_indexed(engine, company)
    if not document_id:
        raise RuntimeError(
            f"No indexed content found for company '{company}'. "
            f"Could not fetch or embed an annual report."
        )

    retrieval_query = (
        "From the company's annual report, surface business priorities, risks, operational challenges, "
        "functions involved and regional focus areas, suitable to derive KPIs for a stakeholder."
    )

    # Retrieval (fast ANN; probes set inside retrieve_top_k_chunks)
    with st.spinner(f"Retrieving context for KPI selection (top-{top_k})…"):
        matches: List[Any] = retrieve_top_k_chunks(
            engine=engine,
            document_id=document_id,
            query=retrieval_query,
            top_k=top_k,
            distance_metric="cosine",
            # ivf_probes: use default from store.py unless you want to override per-call
        )

    if not matches:
        raise RuntimeError("Retrieval returned no relevant context for KPI generation.")

    snippets: List[str] = []
    for m in matches:
        if isinstance(m, (tuple, list)) and len(m) >= 1:
            snippets.append(str(m[0]))
        else:
            snippets.append(str(m))
    digest = "\n\n".join(snippets)

    # LLM (prompt will only return 2 KPIs)
    with st.spinner("Generating top KPIs…"):
        parsed, _raw = generate_top_kpis(
            company=company,
            persona_name=persona_name,
            persona_info=persona_info,
            context=digest,
        )

    # Adapt to your renderer schema (keep top-2)
    ui_payload = _adapt_for_ui(parsed, company=company, persona_name=persona_name,
                               row=persona_row, k_used=top_k)

    # Persist in cache so future requests (same persona/company) are instant
    try:
        upsert_kpis(
            engine,
            company=company,
            persona_key=pkey,
            persona_blob={"name": persona_name, **persona_info},
            kpis_json=ui_payload,
            k_used=top_k,
        )
    except Exception:
        # Non-fatal: cache miss is OK if DB perms constrain writes
        pass

    return ui_payload