# s360_rag/matcher_topn.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser

from .config import (
    AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT,
)
from .prompt_templates import STRICT_JSON_PROMPT
from .utils_text import strip_md_fences, safe_json
from .retriever import top_matches, build_context

def _llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        model=AZURE_DEPLOYMENT,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

def match_topn(
    session,
    persona_text: str,
    persona_kpis: List[str],
    *,
    top_n: int = 3,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Returns (items, latency_ms)
    items: [
      {
        "case_title": str,
        "impact_pointers": [str, ...],
        "reason": str,
        "source_file": str,
        "max_sim": float,
    }, ...]
    Applies strictness: drops items with no impact pointers.
    """
    scored, latency_ms = top_matches(session, persona_text)
    if not scored:
        return [], latency_ms

    # Filter to accepted ones first, then take top_n
    pool = [s for s in scored if s.get("accept")] or []
    pool = pool[:top_n]

    if not pool:
        return [], latency_ms

    llm = _llm()
    items: List[Dict[str, Any]] = []

    for s in pool:
        # Build per-case context
        ctx = build_context(s)
        # Prepare prompt → JSON
        chain = (
            RunnableMap({
                "context": (lambda x: ctx),
                "question": (lambda x: persona_text),
            })
            | STRICT_JSON_PROMPT
            | llm
            | StrOutputParser()
        )
        raw = chain.invoke({"question": persona_text})
        clean = strip_md_fences(raw)
        data = safe_json(clean) or {}

        pointers = data.get("Impact_Pointers") or []
        if not pointers:
            # Strict: if LLM didn’t find pointers for this case, skip it
            continue

        items.append({
            "case_title": data.get("Best_Matching_Case_Study", s.get("case_title", "")),
            "impact_pointers": pointers[:3],
            "reason": data.get("Reason_for_Match", ""),
            "source_file": s.get("source_file", ""),
            "max_sim": s.get("max_sim"),
        })

    return items, latency_ms