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
    ss.setdefault("kpi_sources", {})      # { person_id: result_dict }  (for later KPI generation)


def render_company_insights_inline(
    *,
    company: str,
    person_id: Optional[str] = None,
    default_top_k: int = 12,
    default_chunk_tokens: int = 900,
) -> Optional[Dict[str, Any]]:
    """
    Renders a small inline card to run the Annual-Report Insights pipeline and display results.
    - Stores result in:
        st.session_state["insights_cache"][company]
        st.session_state["kpi_sources"][person_id]   (if provided)
    - Returns the result dict or None.
    """
    if not is_authenticated():
        st.info("Please log in to use Insights.")
        return None

    _ensure_state()

    # Section header
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin:6px 0 6px;">
          <span style="font-size:18px">🧠</span>
          <h3 style="margin:0;">Company Insights (Annual Report)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Controls row: minimally intrusive (no side effects until user clicks)
    colA, colB, colC, colD = st.columns([2, 1, 1, 1])
    with colA:
        st.caption(f"Company: **{company or '—'}**")
    with colB:
        top_k = st.number_input("Top-K", min_value=3, max_value=20, step=1, value=int(default_top_k), key=f"ins_topk_{company}")
    with colC:
        chunk_tokens = st.number_input("Chunk tokens", min_value=300, max_value=1500, step=50, value=int(default_chunk_tokens), key=f"ins_chunk_{company}")
    with colD:
        force = st.checkbox("Force re-run", value=False, key=f"ins_force_{company}")

    run = st.button("Run Insights 🚀", key=f"ins_btn_{company}")

    result = None
    if run:
        if not (company or "").strip():
            st.error("Company name is required.")
        else:
            result = run_end_to_end(
                company.strip(),
                force_rerun=bool(force),
                top_k=int(top_k),
                chunk_tokens=int(chunk_tokens),
                by_user_email=_get_user_email(),
            )
            if result:
                # Persist for reuse
                st.session_state["insights_cache"][company] = result
                if person_id:
                    st.session_state["kpi_sources"][str(person_id)] = result

    # Display last known for this company if available (even if we didn't just run)
    if result is None:
        result = st.session_state["insights_cache"].get(company)

    with st.container(border=True):
        render_insights(result)

    return result
