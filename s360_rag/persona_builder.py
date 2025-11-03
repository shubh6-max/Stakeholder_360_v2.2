"""
s360_rag/persona_builder.py
---------------------------------
Builds a clean, LLM-ready persona description from session data
(s360.full_persona_data) for use in the KPI + Impact RAG pipeline.
"""

import pandas as pd
from typing import Dict, Tuple


# ======================================================
# 🧩 Utility: Safe string cleaner
# ======================================================
def _clean_value(v) -> str:
    """Convert NaN, None, 'NaN' etc. to empty string and strip spaces."""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return s


# ======================================================
# 🧩 Core: Persona builder
# ======================================================
def build_persona_prompt(persona_data: Dict) -> Tuple[str, Dict]:
    """
    Build an LLM-ready persona_info string and cleaned JSON.

    Args:
        persona_data (dict): Raw row from s360.full_persona_data

    Returns:
        (persona_text, cleaned_json)
    """
    if not persona_data:
        return "", {}

    # --- Clean and normalize ---
    clean = {k: _clean_value(v) for k, v in persona_data.items()}

    # --- Only keep meaningful attributes ---
    fields_of_interest = [
        "account",
        "subsidiary",
        "working_group",
        "business_unit",
        "service_line",
        "client_name",
        "client_designation",
        "seniority_level",
        "reporting_manager",
        "reporting_manager_designation",
        "email_address",
        "location",
        "lead_priority",
        "reachout_channel",
        "reachout_lever",
        "status",
        "context",
        "internal_research",   # ✅ newly included
        "external_research",
    ]

    lines = []
    for field in fields_of_interest:
        val = clean.get(field, "")
        if val:
            # Format for readability: Title Case label → value
            label = field.replace("_", " ").title()
            lines.append(f"{label}: {val}")

    persona_text = "\n".join(lines).strip()
    return persona_text, clean


# ======================================================
# 🧪 Example standalone test
# ======================================================
if __name__ == "__main__":
    sample = {
        "account": "Kellanova",
        "client_name": "ANKUSH RAISINGHANI",
        "client_designation": "VP, KNA HUMAN RESOURCES",
        "seniority_level": "VP",
        "business_unit": "HR",
        "service_line": "DAC",
        "working_group": "Business",
        "internal_research": "Focused on HR transformation & automation at Kellanova.",
        "email_address": "ankush.raisinghani@kellanova.com",
        "location": "Chicago, Illinois, United States",
    }

    text, clean = build_persona_prompt(sample)

