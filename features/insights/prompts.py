# features/insights/prompts.py
from __future__ import annotations

from typing import Optional

from .clients import get_gpt_client, get_gpt_deployment_name
from .config import get_settings


# ------------- helpers -------------

def _safe_clip(text: str, max_chars: int = 15000) -> str:
    """
    Defensive clipping before sending to the model (keeps latency and cost sane).
    We clip by characters here because earlier retrieval already shrinks content.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ------------- public API -------------

def llm_extract_risk_factors(company: str, context_text: str, *, max_chars: int = 15000) -> str:
    """
    Returns a markdown bullet list (no intro, no headings) of 8–12 concise risk factors.
    """
    stg = get_settings()
    client = get_gpt_client()
    model = get_gpt_deployment_name()

    clipped = _safe_clip(context_text, max_chars=max_chars)

    prompt = f"""
You are an SDR research assistant.

From the context below (snippets of {company}'s annual/earnings report), extract the top 8–12 **Risk Factors** affecting {company}.
Rules:
- Output ONLY a markdown bullet list (no intro, no headings, no extra text).
- Each bullet MUST be one crisp sentence (<= 25 words).
- Avoid duplicates; merge similar points.
- Prefer business/operational/market/financial/cyber/regulatory risks over generic boilerplate.

Context:
{clipped}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
    )
    return (resp.choices[0].message.content or "").strip()


def llm_extract_painpoints_table(company: str, context_text: str, *, max_chars: int = 18000) -> str:
    """
    Returns a SINGLE markdown table with EXACT columns:
      | Pain Point / Challenge | Functional Focus | Regional Focus Area | Growth Driver |
    No intro, no code fences, only the table.
    """
    stg = get_settings()
    client = get_gpt_client()
    model = get_gpt_deployment_name()

    clipped = _safe_clip(context_text, max_chars=max_chars)

    prompt = f"""
From the context below, output a single Markdown table with EXACTLY these columns:
| Pain Point / Challenge | Functional Focus | Regional Focus Area | Growth Driver |

STRICT Rules:
- Output ONLY the table (no headings, no bullets, no text before/after, no code fences).
- 6–12 rows, each concise and specific.
- "Functional Focus" must be ONE function (e.g., Finance, Supply Chain, Marketing, Sales, IT, Risk Management, Strategy, HR).
- "Regional Focus Area" should be brief (e.g., "North America (NA)", "EU", "LATAM", "AMEA", or a country).
- "Growth Driver" is a short phrase: what's pushing demand/efficiency (e.g., "automation", "omnichannel", "pricing analytics", "AI-led forecasting").

Company: {company}

Context:
{clipped}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()
