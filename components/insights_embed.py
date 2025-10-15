# components/insights_embed.py
from __future__ import annotations

import streamlit as st
from typing import Optional, Dict, Any

from features.insights.pipeline import run_end_to_end
from components.insights_view import render_insights
from utils.auth import is_authenticated


def _get_user_email() -> str:
    try:
        return (st.session_state.get("user") or {}).get("email", "")
    except Exception:
        return ""


def _ensure_state():
    ss = st.session_state
    ss.setdefault("insights_cache", {})   # { company: result_dict }
    ss.setdefault("kpi_sources", {})      # { person_id: result_dict }


def render_company_insights_inline(
    *,
    company: str,
    person_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Minimal inline runner that:
      - Executes the pipeline with dynamic params (no UI knobs)
      - Converts pipeline dataclass → dict
      - Stores result in session for later KPI generation
      - Renders the insights card (dict contract)
    """
    if not is_authenticated():
        st.info("Please log in to use Insights.")
        return None

    _ensure_state()

    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin:6px 0 6px;">
          <span style="font-size:18px">🧠</span>
          <h3 style="margin:0;">Company Insights (Public Sources)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # One simple button; everything else is automatic/dynamic
    run = st.button("Run Insights 🚀", key=f"ins_btn_{company}")
    result_dict: Optional[Dict[str, Any]] = None

    if run:
        if not (company or "").strip():
            st.error("Company name is required.")
        else:
            res = run_end_to_end(
                company.strip(),
                force_rerun=False,
                by_user_email=_get_user_email(),
            )
            if res:
                result_dict = res.to_dict()  # <-- convert dataclass to dict
                # Persist for reuse
                st.session_state["insights_cache"][company] = result_dict
                if person_id:
                    st.session_state["kpi_sources"][str(person_id)] = result_dict

    # Show last known cached result for this company if present
    if result_dict is None:
        result_dict = st.session_state["insights_cache"].get(company)

    with st.container(border=True):
        render_insights(result_dict)  # renderer expects a dict

    return result_dict
