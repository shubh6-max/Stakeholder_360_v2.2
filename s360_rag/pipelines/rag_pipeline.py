
# s360_rag/pipelines/rag_pipeline.py
# -------------------------------------------------
# Persona → KPIs (LLM) → Candidate impacts (pgvector KNN) → Select → Rerank (LLM)
# Returns:
# {
#   "persona_info", "persona_kpis", "persona_industries",
#   "kpi_data", "top_impacts"
# }

from __future__ import annotations
import os
import json
import re
import hashlib
from typing import List, Dict, Any

from sqlalchemy import text
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

from utils.db import get_engine
from pipelines.embedding_utils import generate_embeddings_sync
import streamlit as st

load_dotenv()

# -------------------------
# Azure config (env-driven)
# -------------------------
AZURE_ENDPOINT        = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY         = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION     = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o")

# -------------------------
# LLM client
# -------------------------
_llm = AzureChatOpenAI(
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    openai_api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version=AZURE_API_VERSION,
    temperature=0.0,
)

# -------------------------
# Prompts
# -------------------------
KPI_PROMPT = """
You are a senior B2B customer success analyst.

Step 1: Analyze the stakeholder's role and generate possible Business Functions, KPIs, and Industries.
- Business Functions
- Top 5–7 strategic KPIs for each
- Likely industry focus (use short form for industries, e.g., "Retail", "Manufacturing", "Healthcare", "Finance", "Technology", CPG etc.)

Step 2: Re-analyze your own output and select only the most relevant Business Function, at least five high-confidence strategic KPI , and one high-confidence Industry.

Respond strictly in this JSON format:
{{
    "Business Function": {{
        "strategic_kpis": ["KPI1", "KPI2", "..."],
        "Industry": ["Sector1", "Sector2"]
    }} 
}}

Input:
{persona_info}
""".strip()


IMPACT_SELECT_PROMPT = """
You are a consulting expert helping sales teams link persona KPIs to relevant business impacts.

Context (Impact Statements):
{context}

Persona KPIs:
{persona_kpis}

Task: Select and return the **top 5 most relevant impacts** that directly address the KPIs.
Output must be valid JSON in the format:

[
  {{
    "Impact": "<verbatim impact text>",
    "Industry": "<industry if present, else empty>",
    "BusinessGroup": "<business group if present, else empty>",
    "UseCase": "<use case if present, else empty>"
  }}
]
""".strip()

RERANK_PROMPT = """
You are an expert business relevance scorer.

Given the persona profile, the list of generated KPIs, and candidate impact statements:
1) Rank the impacts from most to least relevant.
2) Prefer quantitative statements (%, $, hours, accuracy, lift, savings, reduction).
3) Enforce diversity—avoid selecting near-duplicates (same numbers/phrases).
4) Output exactly 3 unique impacts.

Return only a valid JSON array (no markdown, no commentary):

[
  {{
    "Impact": "<verbatim impact text as seen in context>",
    "Industry": "<industry if present, else empty>",
    "BusinessGroup": "<business group if present, else empty>",
    "UseCase": "<use case if present, else empty>"
  }}
]

Persona Context:
{persona_info}

Persona KPIs:
{persona_kpis}

Candidate Impacts:
{candidates}
""".strip()

# -------------------------
# Helpers
# -------------------------
def _norm(s: str) -> str:
    """Normalize text for fuzzy matching."""
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _stable_key(text: str) -> str:
    """Deterministic key for caching across reruns (hash() is process-random)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def _pgvector_literal(vec: List[float]) -> str:
    """Convert list to pgvector literal."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def _llm_json(llm_text: str, fallback: Any) -> Any:
    txt = llm_text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(txt)
    except Exception:
        return fallback

def _fetch_candidates_from_db(query_vec: List[float], top_k: int = 30) -> List[Dict[str, Any]]:
    """Retrieve top-K nearest impacts from pgvector store."""
    qvec_str = _pgvector_literal(query_vec)
    engine = get_engine()
    sql = f"""
        SELECT
            impact,
            industry,
            business_group,
            use_case,
            source_file,
            (embedding <=> '{qvec_str}') AS distance
        FROM scout.impact_repository
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> '{qvec_str}' ASC
        LIMIT :k
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"k": top_k}).mappings().all()
    return [dict(r) for r in rows]

def _build_context_block(candidates: List[Dict[str, Any]], max_items: int = 40) -> str:
    """Readable block for LLM context."""
    lines = []
    for i, r in enumerate(candidates[:max_items], 1):
        parts = [
            f"{i}. {r.get('impact','').strip()}",
            f"   Industry: {r.get('industry','') or '—'} | BusinessGroup: {r.get('business_group','') or '—'} | UseCase: {r.get('use_case','') or '—'} | Source: {r.get('source_file','') or '—'}",
        ]
        lines.append("\n".join(parts))
    return "\n".join(lines)

# -------------------------
# Main pipeline
# -------------------------
def run_persona_rag_pipeline(persona_info: str, *, retrieval_k: int = 40) -> Dict[str, Any]:
    """Full RAG pipeline for persona → KPIs → top impacts (industry-aware)."""
    if not persona_info or not persona_info.strip():
        raise ValueError("persona_info is empty")

    # ---- 0) Session cache short-circuit
    cache_key = f"persona_{_stable_key(persona_info)}"
    if "rag_cache" in st.session_state and cache_key in st.session_state["rag_cache"]:
        return st.session_state["rag_cache"][cache_key]

    # ---- 1) Generate KPIs JSON
    kpi_resp = _llm.invoke(KPI_PROMPT.format(persona_info=persona_info))
    kpi_data = _llm_json(kpi_resp.content, {})
    # print("KPI data generated:")
    # print(kpi_data)
    # print("----------------------")

    # ---- 2) Extract ALL KPIs + Industries across functions
    persona_kpis: List[str] = []
    persona_industries: List[str] = []
    try:
        if isinstance(kpi_data, dict):
            for _, blob in kpi_data.items():
                if isinstance(blob, dict):
                    persona_kpis.extend([k for k in blob.get("strategic_kpis", []) if isinstance(k, str)])
                    persona_industries.extend([i for i in blob.get("Industry", []) if isinstance(i, str)])
    except Exception:
        pass

    # Dedup/clean
    persona_kpis = list(dict.fromkeys([_ for _ in (k.strip() for k in persona_kpis) if _]))
    persona_industries = list(dict.fromkeys([_ for _ in (i.strip() for i in persona_industries) if _]))

    if not persona_kpis:
        persona_kpis = ["Process Automation", "Operational Efficiency", "Revenue Growth"]

    # ---- 3) Build embedding query INCLUDING industries
    # This biases retrieval toward industry-matching impacts
    query_terms = persona_kpis + persona_industries
    query = " ; ".join(query_terms)
    vecs = generate_embeddings_sync([query])
    if not vecs or vecs[0] is None:
        raise RuntimeError("Failed to generate embedding for persona KPIs/industries.")
    qvec = vecs[0]

    # ---- 4) Retrieve candidates (pgvector)
    candidates = _fetch_candidates_from_db(qvec, top_k=retrieval_k)
    if not candidates:
        payload = {
            "persona_info": persona_info,
            "persona_kpis": persona_kpis,
            "persona_industries": persona_industries,
            "kpi_data": kpi_data,
            "top_impacts": [],
        }
        st.session_state.setdefault("rag_cache", {})[cache_key] = payload
        return payload

    # Build lookup for enrichment
    impact_to_meta = {
        _norm(c["impact"]): {
            "Industry": c.get("industry") or "",
            "BusinessGroup": c.get("business_group") or "",
            "UseCase": c.get("use_case") or "",
            "FileName": c.get("source_file") or "",
        }
        for c in candidates
    }
    raw_impacts = [c["impact"] for c in candidates]

    # ---- 5) LLM Stage-1: select top-5 from context
    context_block = _build_context_block(candidates, max_items=retrieval_k)
    imp_sel = _llm.invoke(
        IMPACT_SELECT_PROMPT.format(
            context=context_block,
            persona_kpis=json.dumps(persona_kpis, ensure_ascii=False),
        )
    )
    candidates_llm = _llm_json(imp_sel.content, [])
    if not isinstance(candidates_llm, list):
        candidates_llm = [candidates_llm] if candidates_llm else []

    # ---- 6) LLM Stage-2: rerank to top-3
    rer = _llm.invoke(
        RERANK_PROMPT.format(
            persona_info=persona_info,
            persona_kpis=json.dumps(persona_kpis, ensure_ascii=False),
            candidates=json.dumps(candidates_llm, ensure_ascii=False),
        )
    )
    top_impacts = _llm_json(rer.content, [])
    if not isinstance(top_impacts, list):
        top_impacts = candidates_llm[:3]

    # ---- 7) Enrich impacts with meta (Industry/UseCase/BusinessGroup/FileName)
    def _attach_meta(item: Dict[str, Any]) -> Dict[str, Any]:
        txt = item.get("Impact", "") or ""
        meta = impact_to_meta.get(_norm(txt))
        if not meta:
            ntxt = _norm(txt)
            for raw in raw_impacts:
                nraw = _norm(raw)
                if ntxt and (ntxt in nraw or nraw in ntxt):
                    meta = impact_to_meta.get(nraw)
                    break
        meta = meta or {"Industry": "", "BusinessGroup": "", "UseCase": "", "FileName": ""}

        return {
            "Impact": txt,
            "Industry": item.get("Industry", "") or meta["Industry"],
            "BusinessGroup": item.get("BusinessGroup", "") or meta["BusinessGroup"],
            "UseCase": item.get("UseCase", "") or meta["UseCase"],
            "FileName": meta["FileName"],
        }

    cleaned = [_attach_meta(it) for it in top_impacts[:3]]

    # ---- 8) Persist in Streamlit session so refresh doesn't recompute
    payload = {
        "persona_info": persona_info,
        "persona_kpis": persona_kpis,
        "persona_industries": persona_industries,
        "kpi_data": kpi_data,
        "top_impacts": cleaned,
    }
    st.session_state.setdefault("rag_cache", {})[cache_key] = payload
    return payload
