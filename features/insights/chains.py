# features/insights/chains.py
from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Tuple

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator


# ------------------------------
# LLM factory (Azure OpenAI)
# ------------------------------

def make_llm(*, temperature: float = 0.2, model: Optional[str] = None) -> AzureChatOpenAI:
    """
    Build an AzureChatOpenAI instance from environment variables.
    Required ENV:
      - AZURE_ENDPOINT
      - AZURE_API_KEY
      - AZURE_DEPLOYMENT   (deployment name)
      - AZURE_API_VERSION  (e.g., 2024-02-15-preview)
    """
    endpoint = os.getenv("AZURE_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_API_KEY", "")
    deployment = model or os.getenv("AZURE_DEPLOYMENT", "")
    api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

    if not (endpoint and api_key and deployment):
        raise RuntimeError(
            "Azure OpenAI not configured. Ensure AZURE_ENDPOINT, AZURE_API_KEY, "
            "AZURE_DEPLOYMENT, AZURE_API_VERSION are set."
        )

    # Note: keeping argument names as you had them to avoid breaking your env.
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        api_version=api_version,
        temperature=temperature,
    )


# ------------------------------
# Helpers
# ------------------------------

def _clip(text: str, max_chars: int = 15000) -> str:
    """Defensive truncate for long contexts."""
    if not isinstance(text, str):
        return ""
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text


# ------------------------------
# Chain 1: Risk Factors (markdown bullets only)
# ------------------------------

def risk_factors_chain(llm: Optional[AzureChatOpenAI] = None):
    """
    Returns a chain that emits ONLY a markdown bullet list of risk factors.
    Usage:
        chain = risk_factors_chain()
        md = chain.invoke({"company": "Kellanova", "context": "... big text ..."})
    """
    llm = llm or make_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an SDR research assistant. Work ONLY with the provided context. "
                "Answer conservatively and avoid hallucinations.",
            ),
            (
                "human",
                "Company: {company}\n"
                "Task: From the context below, extract the top 8–12 Risk Factors affecting the company.\n"
                "Return ONLY a markdown bullet list (no headers, no intro, no numbering).\n\n"
                "Context:\n{context}"
            ),
        ]
    )

    return prompt | llm | StrOutputParser()


# ------------------------------
# Chain 2: Pain Points Table (markdown table only)
# ------------------------------

def painpoints_table_chain(llm: Optional[AzureChatOpenAI] = None):
    """
    Returns a chain that emits ONLY a markdown table:
    | Pain Point / Challenge | Functional Focus | Regional Focus Area | Growth Driver |
    Usage:
        chain = painpoints_table_chain()
        md_table = chain.invoke({"company":"...", "context":"..."})
    """
    llm = llm or make_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You convert context into compact, business-ready tables without extra narration.",
            ),
            (
                "human",
                "Company: {company}\n"
                "From the context below, output ONE markdown table with EXACTLY these columns:\n"
                "| Pain Point / Challenge | Functional Focus | Regional Focus Area | Growth Driver |\n\n"
                "Rules:\n"
                "- Output ONLY the table (no intro, no code fences, no text before/after).\n"
                "- Functional Focus must be one function per row (e.g., Finance, Supply Chain, Marketing, IT, Risk Management, Strategy).\n"
                "- Regional Focus Area should be concise (e.g., 'North America (NA)', 'International (LATAM, EU)', 'AMEA').\n"
                "- Growth Driver should be a short phrase pointing to an enabler/initiative.\n\n"
                "Context:\n{context}"
            ),
        ]
    )

    return prompt | llm | StrOutputParser()


# ------------------------------
# Chain 3: Top KPIs (JSON, structured)
# ------------------------------

class KPI(BaseModel):
    name: str = Field(..., description="Short KPI title.")
    why: str = Field(..., description="Why this KPI matters for the selected persona/company.")
    suggested_owner_function: str = Field(
        default="",
        description="One likely owning function (e.g., Finance, Supply Chain)."
    )
    relevance_score: int = Field(
        default=4,
        ge=1,
        le=5,
        description="1 (low) to 5 (high) relevance for this persona."
    )
    backing_evidence: List[str] = Field(
        default_factory=list,
        description="1-3 short quotes/phrases from the context that support this KPI."
    )

    @validator("name", "why", pre=True)
    def _strip(cls, v):
        return (v or "").strip()


class KPIResponse(BaseModel):
    company: str
    persona: str
    kpis: List[KPI]

    @validator("company", "persona", pre=True)
    def _norm(cls, v):
        return (v or "").strip()


def kpis_chain(llm: Optional[AzureChatOpenAI] = None):
    """
    Returns (chain, parser) where chain outputs JSON matching KPIResponse.
    Usage:
        chain, parser = kpis_chain()
        raw = chain.invoke({
            "company":"Kellanova",
            "persona_name":"Adi McIlveen",
            "persona_info":"Title: VP Supply Chain; Region: NA; ..."
            "context":"<retrieved chunks here>"
        })
        obj = parser.parse(raw)
    """
    llm = llm or make_llm(temperature=0.1)
    parser = PydanticOutputParser(pydantic_object=KPIResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise KPI planner. Propose exactly TWO KPIs tailored to the persona and company. "
                "Use ONLY the supplied context. If unsure, be conservative.",
            ),
            (
                "human",
                "Company: {company}\n"
                "Persona: {persona_name}\n"
                "Persona Info (free-form dict or bullets):\n{persona_info}\n\n"
                "Context (retrieved snippets):\n{context}\n\n"
                "Return STRICT JSON matching this schema:\n{format_instructions}\n"
                "Fill: company, persona, and kpis (length exactly 2). "
                "Each KPI must include name, why, suggested_owner_function, relevance_score (1-5), backing_evidence (1-3 short quotes)."
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | StrOutputParser()
    return chain, parser


# ------------------------------
# NEW: Summarizers
# ------------------------------

def summarize_risk_md(company: str, risk_md: str, max_bullets: int = 6) -> str:
    """
    Condense/merge an existing markdown bullet list of risk factors into a shorter list.
    Returns ONLY a markdown bullet list with at most `max_bullets` bullets.
    """
    risk_md = (risk_md or "").strip()
    if not risk_md:
        return ""
    llm = make_llm(temperature=0.1)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You compress and deduplicate risk bullets into a smaller, high-signal set."
            ),
            (
                "human",
                "Company: {company}\n"
                "Input bullets (markdown):\n<risks>\n{risks}\n</risks>\n\n"
                "Task:\n"
                f"- Merge duplicates/near-duplicates; keep the most material items.\n"
                f"- Use short, clear phrasing; preserve key qualifiers.\n"
                f"- Return ONLY a markdown bullet list with AT MOST {max_bullets} bullets.\n"
                "- No headings, no extra text, no code fences."
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"company": company, "risks": _clip(risk_md, 6000)}).strip()


def summarize_painpoints_table_md(company: str, table_md: str, max_rows: int = 5) -> str:
    """
    Reduce a markdown table to a representative subset, preserving columns:
    | Pain Point / Challenge | Functional Focus | Regional Focus Area | Growth Driver |
    Returns ONLY the reduced table (with header row).
    """
    table_md = (table_md or "").strip()
    if not table_md:
        return ""
    llm = make_llm(temperature=0.1)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You reduce markdown tables to representative subsets while preserving schema."
            ),
            (
                "human",
                "Company: {company}\n"
                "Input table (Markdown):\n<table>\n{table}\n</table>\n\n"
                "Task:\n"
                "| Pain Point / Challenge | Functional Focus | Regional Focus Area | Growth Driver |\n"
                f"- Keep the SAME 4 columns in this exact order.\n"
                f"- Remove duplicates and low-signal rows; keep a representative set of AT MOST {max_rows} rows.\n"
                "- Clean formatting and ensure a proper header row is present.\n"
                "- Return ONLY the reduced Markdown table, nothing else."
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"company": company, "table": _clip(table_md, 9000)}).strip()


# ------------------------------
# Thin convenience wrappers
# ------------------------------

def generate_risk_factors(company: str, context: str, llm: Optional[AzureChatOpenAI] = None) -> str:
    """Run the risk chain with safe truncation."""
    chain = risk_factors_chain(llm)
    return chain.invoke({"company": company, "context": _clip(context, 15000)})


def generate_painpoints_table(company: str, context: str, llm: Optional[AzureChatOpenAI] = None) -> str:
    """Run the painpoints table chain with safe truncation."""
    chain = painpoints_table_chain(llm)
    return chain.invoke({"company": company, "context": _clip(context, 18000)})


def generate_top_kpis(
    company: str,
    persona_name: str,
    persona_info: Dict[str, Any] | str,
    context: str,
    llm: Optional[AzureChatOpenAI] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Run the KPI chain and return (parsed_dict, raw_text).
    persona_info can be a dict or a string; it is passed through as-is.
    """
    chain, parser = kpis_chain(llm)
    raw = chain.invoke({
        "company": company,
        "persona_name": persona_name,
        "persona_info": persona_info if isinstance(persona_info, str) else str(persona_info),
        "context": _clip(context, 16000),
    })
    try:
        parsed = parser.parse(raw).dict()
    except Exception:
        # Fallback to a minimal JSON if parsing fails
        parsed = {
            "company": company,
            "persona": persona_name,
            "kpis": [],
        }
    return parsed, raw


def generate_and_summarize(
    company: str,
    context: str,
    *,
    max_bullets: int = 6,
    max_rows: int = 5,
) -> Tuple[str, str, str, str]:
    """
    Convenience: returns (full_risks_md, full_table_md, short_risks_md, short_table_md).
    """
    full_risks = generate_risk_factors(company, context)
    full_table = generate_painpoints_table(company, context)
    short_risks = summarize_risk_md(company, full_risks, max_bullets=max_bullets) if full_risks else ""
    short_table = summarize_painpoints_table_md(company, full_table, max_rows=max_rows) if full_table else ""
    return full_risks, full_table, short_risks, short_table
