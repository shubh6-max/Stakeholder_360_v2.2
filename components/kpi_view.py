# components/kpi_view.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union
import streamlit as st

# Accept either the typed KPIResult (from features/insights/retrieve.py)
# or a plain dict with the same shape.
KPIResultLike = Union[Dict[str, Any], "KPIResult"]


def _get(result: KPIResultLike, key: str, default: Any = "") -> Any:
    if hasattr(result, key):
        return getattr(result, key)
    if isinstance(result, dict):
        return result.get(key, default)
    return default


def _kpis(result: KPIResultLike) -> List[Dict[str, Any]]:
    if hasattr(result, "kpis"):
        return _get(result, "kpis") or []
    return result.get("kpis") or []


def _coerce_str(v: Any) -> str:
    return ("" if v is None else str(v)).strip()


def _badge(text: str) -> None:
    text = _coerce_str(text)
    if not text:
        return
    st.markdown(
        f"""
        <span style="
          display:inline-flex;align-items:center;gap:6px;
          padding:6px 10px;border-radius:999px;
          background:#e6f4ff;color:#0f172a;border:1px solid #cfe0ff;
          font-weight:600;font-size:13px;">
          {text}
        </span>
        """,
        unsafe_allow_html=True,
    )


def _pill(label: str, value: str) -> str:
    label = _coerce_str(label)
    value = _coerce_str(value)
    if not value:
        value = "—"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'padding:4px 8px;border-radius:7px;background:#f8fafc;'
        f'border:1px solid #e2e8f0;font-size:12px;">'
        f'<b>{label}:</b> {value}</span>'
    )


def _sources_block(sources: Iterable[Dict[str, Any]]) -> None:
    sources = list(sources or [])
    if not sources:
        return
    with st.expander("Sources (report excerpts)"):
        for i, s in enumerate(sources, 1):
            txt = _coerce_str(s.get("text"))
            url = _coerce_str(s.get("pdf_url"))
            idx = s.get("chunk_index", "—")

            st.markdown(
                f"""
                <div style="border:1px dashed #d0d7e2;border-radius:10px;padding:10px;margin-bottom:8px;background:#fff;">
                  <div style="font-size:12px;color:#475569;margin-bottom:6px;">
                    <b>Chunk:</b> {idx} &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Source:</b> {"<a href='"+url+"' target='_blank'>PDF</a>" if url else "N/A"}
                  </div>
                  <div style="font-size:13px;line-height:1.4;white-space:pre-wrap;">{txt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_kpis(result: KPIResultLike) -> None:
    """
    Render Top-2 KPI cards.

    Expected fields on `result`:
      - company (str)
      - persona_name (str)
      - persona_title (str)
      - business_unit (str)
      - service_line (str)
      - working_group (str)
      - k_used (int, optional)
      - kpis: List[{
          title, why_it_matters, how_to_measure, suggested_initiatives: [str], sources: [{text, pdf_url, chunk_index}]
        }]
    """
    if not result:
        st.info("No KPI data available. Run insights first.")
        return

    company = _get(result, "company", "")
    persona = _get(result, "persona_name", "")
    title = _get(result, "persona_title", "")
    bu = _get(result, "business_unit", "")
    sl = _get(result, "service_line", "")
    wg = _get(result, "working_group", "")
    k_used = _get(result, "k_used", None)

    # Header
    st.markdown(
        f"""
        <div style="
          display:flex;justify-content:space-between;align-items:center;
          background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
          padding:10px 12px;margin:8px 0;">
          <div style="display:flex;align-items:center;gap:10px;">
            <img src="https://img.icons8.com/?size=100&id=0uRhf2mft47s&format=png&color=000000" width="22" height="22">
            <div style="font-size:16px;font-weight:700;">Top KPIs for {persona or 'Stakeholder'}</div>
          </div>
          <div>
            {"".join([
              _pill("Company", company),
              " ",
              _pill("Title", title),
              " ",
              _pill("BU", bu),
              " ",
              _pill("Service Line", sl),
              " ",
              _pill("Working Group", wg),
            ])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if k_used:
        _badge(f"retrieval K = {int(k_used)}")

    # KPI cards
    kpis = _kpis(result)
    if not kpis:
        st.warning("No KPIs returned by the model.")
        return

    for i, kpi in enumerate(kpis, 1):
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:16px;font-weight:800;margin-bottom:4px;'>#{i} — { _coerce_str(kpi.get('title')) or 'Untitled KPI' }</div>",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Why it matters**")
                st.write(_coerce_str(kpi.get("why_it_matters")) or "—")

            with col2:
                st.markdown("**How to measure**")
                st.write(_coerce_str(kpi.get("how_to_measure")) or "—")

            # st.markdown("**Suggested initiatives**")
            # inits = [s for s in (kpi.get("suggested_initiatives") or []) if _coerce_str(s)]
            # if inits:
            #     st.markdown("\n".join([f"- {_coerce_str(x)}" for x in inits]))
            # else:
            #     st.write("—")

            _sources_block(kpi.get("sources") or [])
