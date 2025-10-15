# components/insights_embed.py
from __future__ import annotations

import time
import streamlit as st
from typing import Optional, Dict, Any, Mapping

from features.insights.pipeline import run_end_to_end
from components.insights_view import render_insights
from utils.auth import is_authenticated


def _get_user_email() -> str:
    try:
        return (st.session_state.get("user") or {}).get("email", "")
    except Exception:
        return ""


def _ensure_state() -> None:
    ss = st.session_state
    ss.setdefault("insights_cache", {})     # { company: result_dict }
    ss.setdefault("kpi_sources", {})        # { person_id: result_dict }
    ss.setdefault("insights_running", set())  # debouncer set of company names


def _to_dict(result: Any) -> Optional[Dict[str, Any]]:
    """Accept dict or dataclass-like object and return plain dict."""
    if result is None:
        return None
    if isinstance(result, Mapping):
        # make a shallow copy to avoid accidental mutation of state
        return dict(result)
    # dataclass or object with attributes (InsightResult)
    try:
        return {
            "company": getattr(result, "company", ""),
            "pdf_url": getattr(result, "pdf_url", ""),
            "risk_factors_markdown": getattr(result, "risk_factors_markdown", ""),
            "table_markdown": getattr(result, "table_markdown", ""),
            "elapsed_seconds": float(getattr(result, "elapsed_seconds", 0.0) or 0.0),
            "cached": bool(getattr(result, "cached", False)),
            "source_type": getattr(result, "source_type", "annual_pdf"),
        }
    except Exception:
        return None


def render_company_insights_inline(
    *,
    company: str,
    person_id: Optional[str] = None,
    autorun: bool = False,                  # set True to run automatically (e.g., after filters are set)
    button_label: str = "Run Insights 🚀",
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
          <h3 style="margin:0;">Company Insights</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    company = (company or "").strip()
    if not company:
        st.warning("Select a company to run insights.")
        return None

    key_base = company.lower().replace(" ", "_")
    run_clicked = st.button(button_label, key=f"ins_btn_{key_base}")

    result_dict: Optional[Dict[str, Any]] = st.session_state["insights_cache"].get(company)

    # Debounce: prevent concurrent runs for same company
    running_set = st.session_state["insights_running"]

    should_run = run_clicked or (autorun and result_dict is None and company not in running_set)

    if should_run and company not in running_set:
        running_set.add(company)
        try:
            with st.spinner("Fetching & generating insights…"):
                res = run_end_to_end(
                    company,
                    force_rerun=False,
                    by_user_email=_get_user_email(),
                )
                result_dict = _to_dict(res)
                if result_dict:
                    st.session_state["insights_cache"][company] = result_dict
                    if person_id:
                        st.session_state["kpi_sources"][str(person_id)] = result_dict
                else:
                    st.warning("No insights were generated. Try again or pick another company.")
        except Exception as e:
            st.error(f"Insights pipeline failed: {e}")
        finally:
            # small delay so spinner doesn’t flicker too fast on quick responses
            time.sleep(0.05)
            running_set.discard(company)

    with st.container(border=True):
        render_insights(result_dict)

    return result_dict
