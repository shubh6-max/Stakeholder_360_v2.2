"""
s360_rag/persona_builder.py
---------------------------------
Builds a clean, LLM-ready persona description from session data
(s360.full_persona_data) for use in the KPI + Impact RAG pipeline.
Enriched with LinkedIn data from scout.linkedin_clients_data.
"""

import re
import pandas as pd
from typing import Dict, Tuple
from sqlalchemy import text
from utils.db import get_engine


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
# 🧩 LinkedIn Enrichment
# ======================================================
def _fetch_linkedin_info(client_name: str, email: str) -> Dict[str, str]:
    """
    Fetch LinkedIn enrichment (title + description_html) from scout.linkedin_clients_data
    using either client_name or email match.
    """
    try:
        engine = get_engine()
        sql = """
            SELECT client_present_title, client_present_description_html
            FROM scout.linkedin_clients_data
            WHERE LOWER(client_name) = LOWER(:name)
               OR LOWER(email_id) = LOWER(:email)
            LIMIT 1
        """
        with engine.begin() as conn:
            row = conn.execute(text(sql), {"name": client_name, "email": email}).mappings().first()
        if not row:
            return {}
        title = row.get("client_present_title") or ""
        desc_html = row.get("client_present_description_html") or ""
        desc_text = re.sub(r"<[^>]*>", "", desc_html)  # remove HTML tags
        return {
            "linkedin_title": title.strip(),
            "linkedin_description": desc_text.strip(),
        }
    except Exception as e:
        print(f"[WARN] LinkedIn enrichment failed: {e}")
        return {}


# ======================================================
# 🧩 Core: Persona builder
# ======================================================
def build_persona_prompt(persona_data: Dict) -> Tuple[str, Dict]:
    """
    Build an LLM-ready persona_info string and cleaned JSON.

    Enriches the persona using LinkedIn data if available.

    Returns:
        (persona_text, cleaned_json)
    """
    if not persona_data:
        return "", {}

    # --- Clean and normalize ---
    clean = {k: _clean_value(v) for k, v in persona_data.items()}

    name = clean.get("client_name", "")
    email = clean.get("email_address", "")

    # --- LinkedIn enrichment ---
    li_info = _fetch_linkedin_info(name, email)
    # print("li_info",li_info)
    linkedin_title = li_info.get("linkedin_title", "")
    linkedin_desc = li_info.get("linkedin_description", "")

    # --- Combine with internal research ---
    internal_context = clean.get("internal_research", "")
    enriched_about = " ".join(filter(None, [linkedin_title, linkedin_desc, internal_context])).strip()

    # --- Structured narrative for LLM ---
    persona_text = (
        f"{clean.get('client_name', 'Unknown')} "
        f"is {clean.get('client_designation', '')} "
        f"at {clean.get('account', '')}. "
        f"They work in the {clean.get('business_unit', '')} unit under "
        f"{clean.get('service_line', '')} service line "
        f"and are part of the {clean.get('working_group', '')} group. "
        f"Their seniority level is {clean.get('seniority_level', '')}. "
        f"linkedin info: {enriched_about}."
    )

    # --- Compact formatted key:value backup text ---
    fields_of_interest = [
        "account", "subsidiary", "working_group", "business_unit",
        "service_line", "client_name", "client_designation", "seniority_level",
        "reporting_manager", "reporting_manager_designation",
        "email_address", "location", "lead_priority",
        "reachout_channel", "reachout_lever", "status",
        "context", "internal_research", "external_research"
    ]
    lines = []
    for f in fields_of_interest:
        v = clean.get(f, "")
        if v:
            lines.append(f"{f.replace('_', ' ').title()}: {v}")
    fallback_text = "\n".join(lines)

    persona_final = persona_text + "\n\nAdditional Details:\n" + fallback_text

    # --- Return both narrative and structured JSON ---
    clean.update({
        "linkedin_title": linkedin_title,
        "linkedin_description": linkedin_desc,
        "persona_narrative": persona_text
    })

    return persona_final.strip(), clean


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
    # print("\n===== Persona Info (for LLM) =====\n")
    # print(text)
    # print("\n===== Cleaned JSON =====\n")
    # print(clean)
