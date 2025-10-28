# s360_rag/kpi_builder.py
from __future__ import annotations

from typing import List, Callable, Optional, Dict, Any
import json
import re

from langchain_openai import AzureChatOpenAI

from .schemas import PersonaInput, KPIBlock
from .config import (
    AZURE_ENDPOINT,
    AZURE_API_KEY,
    AZURE_API_VERSION,
    AZURE_DEPLOYMENT,
)

JsonDict = Dict[str, Any]
DebugHook = Optional[Callable[[str, Any], None]]


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _persona_dict(p: PersonaInput) -> JsonDict:
    """Flatten persona to a compact dict the LLM can consume deterministically."""
    return {
        "client_designation": _norm(getattr(p, "client_designation", "")),
        "seniority_level": _norm(getattr(p, "seniority_level", "")),
        "business_functions": _norm(getattr(p, "business_functions", "")),
        "service_line": _norm(getattr(p, "service_line", "")),
        "industry_hint": _norm(getattr(p, "industry_hint", "")),
        "linkedin_title": _norm(getattr(p, "linkedin_title", "")),
        "linkedin_about": _norm(getattr(p, "linkedin_about", "")),
        "linkedin_desc_html": _norm(getattr(p, "linkedin_desc_html", "")),
    }


def _strip_md_fences(s: str) -> str:
    """Remove ```json fences etc."""
    return re.sub(r"```(?:json)?|```", "", s or "").strip()


def _safe_json_loads(s: str, default: Any) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return default


def build(
    persona: PersonaInput,
    *,
    max_kpis: int = 5,
    debug_hook: DebugHook = None,
) -> KPIBlock:
    """
    Fully dynamic KPI builder:
      - Derives KPIs from persona metadata + LinkedIn text using AzureChatOpenAI
      - Returns up to `max_kpis` canonical KPI names
      - If JSON malformed, falls back to a minimal generic set

    Args:
      persona: PersonaInput with fields (designation, seniority, function, industry, LinkedIn text, etc.)
      max_kpis: cap on number of KPIs to return (default 5)
      debug_hook: optional callable(label, payload) to surface internals (e.g., lambda k,v: st.write(k, v))

    Returns:
      KPIBlock with {"Business_Function": {"strategic_kpis": [...], "Industry": []}}
    """
    pdict = _persona_dict(persona)

    if debug_hook:
        debug_hook("kpi_builder.persona_input", pdict)

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        model=AZURE_DEPLOYMENT,
        temperature=0.2,
        # Force JSON object responses from Azure:
        response_format={"type": "json_object"},
    )

    # System + user prompt designed for canonical, business-grade KPI names
    # No hard-coded heuristics; the LLM infers from role, function, industry, and LinkedIn signals.
    user_prompt = (
        "You are a KPI standardization assistant. Based on the persona profile below, "
        "produce a list of **canonical business KPIs** that this persona is most likely to track.\n\n"
        "Rules:\n"
        "1) Output **only** well-known KPI names (no explanations, no formulas).\n"
        "2) Prefer function- and industry-relevant KPIs.\n"
        "3) Avoid synonyms—pick the most standard name (e.g., use 'Revenue Growth' over 'Sales Growth %' if needed).\n"
        "4) Keep it concise; do not include duplicates.\n\n"
        f"Persona JSON:\n{json.dumps(pdict, ensure_ascii=False)}\n\n"
        f"Return JSON only with this shape:\n"
        f'{{"kpis": ["<KPI 1>", "<KPI 2>", "..."]}}'
    )

    raw = llm.invoke(user_prompt).content or "{}"
    if debug_hook:
        debug_hook("kpi_builder.raw_llm_response", raw)

    clean = _strip_md_fences(raw)
    parsed = _safe_json_loads(clean, {"kpis": []})
    kpis: List[str] = [str(x).strip() for x in parsed.get("kpis", []) if str(x).strip()]

    # Final guards
    #  - de-dup while preserving order
    #  - cap to max_kpis
    seen = set()
    deduped: List[str] = []
    for k in kpis:
        kl = k.lower()
        if kl not in seen:
            deduped.append(k)
            seen.add(kl)

    if not deduped:
        # ultra-minimal fallback if LLM gives nothing usable
        deduped = ["Revenue Growth", "Cost Savings", "Customer Satisfaction"][:max_kpis]

    final = deduped[:max_kpis]

    if debug_hook:
        debug_hook("kpi_builder.final_kpis", final)

    return KPIBlock(Business_Function={"strategic_kpis": final, "Industry": []})
