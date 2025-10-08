# components/insights_view.py
from __future__ import annotations

from typing import Optional, Dict, Any
import streamlit as st


def _badge(text: str, href: Optional[str] = None, *, color="#0ea5e9") -> None:
    """
    Small pill/badge. If href provided -> clickable.
    """
    safe_text = (text or "").replace("<", "&lt;").replace(">", "&gt;")
    style = f"""
      display:inline-flex;align-items:center;gap:6px;
      padding:6px 10px;border-radius:999px;
      background:{color}1a;color:#0f172a;
      border:1px solid {color}66;font-weight:600;
      font-size:13px;text-decoration:none;
    """
    icon = "📄" if href else "🏷️"
    if href:
        st.markdown(f'<a href="{href}" target="_blank" style="{style}">{icon} {safe_text}</a>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="{style}">{icon} {safe_text}</span>', unsafe_allow_html=True)


def _meta_row(items: list[tuple[str, str]]) -> None:
    """
    Render a light metadata row (e.g., runtime, cache status).
    """
    bits = []
    for label, val in items:
        safe_label = label.replace("<", "&lt;").replace(">", "&gt;")
        safe_val = val.replace("<", "&lt;").replace(">", "&gt;")
        bits.append(f'<span style="opacity:.75">{safe_label}:</span> <b>{safe_val}</b>')
    html = " &nbsp;•&nbsp; ".join(bits)
    st.markdown(f'<div style="margin-top:6px;color:#334155;font-size:13px;">{html}</div>', unsafe_allow_html=True)


def _section_header(title: str, icon: str) -> None:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin:6px 0 6px;">
          <span style="font-size:18px">{icon}</span>
          <h3 style="margin:0;">{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insights(result: Dict[str, Any] | None) -> None:
    """
    Pretty-print the pipeline result dict returned by features.insights.pipeline.run_end_to_end().
    Expected keys:
      - company (str)
      - pdf_url (str)
      - risk_factors_markdown (str)
      - table_markdown (str)
      - elapsed_seconds (float)
      - cached (bool)
    """
    if not result:
        st.info("No insights to display.")
        return

    company = result.get("company") or ""
    pdf_url = result.get("pdf_url") or ""
    risk_md = (result.get("risk_factors_markdown") or "").strip()
    table_md = (result.get("table_markdown") or "").strip()
    elapsed = result.get("elapsed_seconds") or 0.0
    cached = bool(result.get("cached"))

    # Header row: company + PDF badge
    st.markdown(
        f"""
        <div style="
          display:flex;justify-content:space-between;align-items:center;
          background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
          padding:10px 12px;margin-bottom:10px;">
          <div style="display:flex;align-items:center;gap:10px;">
            <img src="https://img.icons8.com/?size=100&id=114049&format=png&color=000000" width="22" height="22">
            <div style="font-size:16px;font-weight:700;">Insights for {company}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # URL badge + meta
    cols = st.columns([1, 1])
    with cols[0]:
        if pdf_url:
            _badge("Annual Report (PDF / source)", pdf_url)
        else:
            _badge("No PDF URL detected", None, color="#ef4444")

    with cols[1]:
        _meta_row([
            ("Runtime", f"{elapsed:.2f}s"),
            ("Cache", "hit" if cached else "miss"),
        ])

    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:10px 0;'/>", unsafe_allow_html=True)

    # Risk factors
    _section_header("Risk Factors", "⚠️")
    if risk_md:
        st.markdown(risk_md)
    else:
        st.caption("No risk factors extracted.")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Pain points table
    _section_header("Pain Points / Functions / Regions", "📊")
    if table_md:
        st.markdown(table_md)
    else:
        st.caption("No table extracted.")

    # Footer note
    st.markdown(
        "<div style='opacity:.7;font-size:12px;margin-top:10px;'>Generated from annual/earnings report excerpts.</div>",
        unsafe_allow_html=True,
    )
