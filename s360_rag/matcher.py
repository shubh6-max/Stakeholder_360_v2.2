# s360_rag/matcher.py
from __future__ import annotations

import json
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser

from .config import (
    AZURE_ENDPOINT,
    AZURE_API_KEY,
    AZURE_API_VERSION,
    AZURE_DEPLOYMENT,
    SIM_THRESHOLD as DEFAULT_SIM_THRESHOLD,
)
from .prompt_templates import STRICT_JSON_PROMPT
from .utils_text import strip_md_fences, safe_json
from .retriever import top_matches, build_context
from .schemas import MatchOutput, KPIBlock


def match_strict(
    session: Session,
    persona_text: str,
    persona_kpis: KPIBlock,
    *,
    sim_threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    debug: bool = False,
) -> MatchOutput:
    """
    Strict persona→case match:
      - Retrieve top matches with cosine ANN
      - If best hit doesn't meet similarity threshold OR yields no impact pointers → "no match"
      - Otherwise, return structured JSON

    Args:
      session: SQLAlchemy session
      persona_text: free-text description of the persona
      persona_kpis: KPIBlock with canonical persona KPIs
      sim_threshold: optional override for similarity threshold (0..1)
      top_k: optional override for retrieval fan-out
      debug: include light debug information in Reason_for_Match (kept concise)

    Returns:
      MatchOutput with Persona_KPIs (<=5), Best_Matching_Case_Study, Impact_Pointers (<=3), Reason_for_Match
    """
    thr = DEFAULT_SIM_THRESHOLD if sim_threshold is None else float(sim_threshold)

    # 1) Retrieve & score
    scored, latency_ms = top_matches(
        session=session,
        question=persona_text,
        top_k=top_k,
        sim_threshold=thr,  # enforce strictness here
    )
    top = scored[0] if scored else None
    accepted = bool(top and top.get("accept"))

    # 2) Prepare LLM only if we have an accepted hit
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        model=AZURE_DEPLOYMENT,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    # Build LCEL chain – pass empty context if not accepted to avoid hallucinations
    retrieval_chain = (
        RunnableMap(
            {
                "context": (lambda x: build_context(top) if accepted else ""),
                "question": (lambda x: x["question"]),
            }
        )
        | STRICT_JSON_PROMPT
        | llm
        | StrOutputParser()
    )

    # 3) Invoke LLM (or short-circuit)
    raw = retrieval_chain.invoke({"question": persona_text}) if accepted else "{}"
    clean = strip_md_fences(raw or "")
    data = safe_json(clean) or {}

    # 4) Compose strict output
    persona_kpi_list = (persona_kpis.Business_Function or {}).get("strategic_kpis", [])[:5]

    # Strict fail conditions:
    no_pointers = not data.get("Impact_Pointers")
    if (not accepted) or no_pointers:
        reason = "No matching case study found."
        if debug and top:
            reason += f" (debug: max_sim={top.get('max_sim')}, mean_top2={top.get('mean_top2')}, thr={thr}, ms={latency_ms})"
        return MatchOutput(
            Persona_KPIs=persona_kpi_list,
            Best_Matching_Case_Study="None",
            Impact_Pointers=[],
            Reason_for_Match=reason,
        )

    # Happy path (cap impact pointers to 3, prefer LLM title else retriever title)
    best_title = data.get("Best_Matching_Case_Study") or (top.get("case_title") if top else "Unknown")
    impact = list(data.get("Impact_Pointers", []))[:3]
    reason = data.get("Reason_for_Match", "")

    if debug and top:
        tail = f" (debug: max_sim={top.get('max_sim')}, mean_top2={top.get('mean_top2')}, ms={latency_ms})"
        reason = f"{reason}{tail}" if reason else tail

    return MatchOutput(
        Persona_KPIs=persona_kpi_list,
        Best_Matching_Case_Study=best_title,
        Impact_Pointers=impact,
        Reason_for_Match=reason,
    )