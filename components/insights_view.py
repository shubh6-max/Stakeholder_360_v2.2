# components/insights_view.py
from __future__ import annotations

from typing import Optional, Dict, Any, Mapping
import streamlit as st

def _badge(text: str, href: Optional[str] = None, *, color="#0ea5e9") -> None:
    """
    Small pill/badge. If href provided -> clickable.
    """
    safe_text = (text or "").replace("<", "&lt;").replace(">", "&gt;")
    style = (
        "display:inline-flex;align-items:center;gap:6px;"
        "padding:6px 10px;border-radius:999px;"
        f"background:{color}1a;color:#0f172a;"
        f"border:1px solid {color}66;font-weight:600;"
        "font-size:13px;text-decoration:none;"
    )
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
        safe_label = (label or "").replace("<", "&lt;").replace(">", "&gt;")
        safe_val = (val or "").replace("<", "&lt;").replace(">", "&gt;")
        bits.append(f'<span style="opacity:.75">{safe_label}:</span> <b>{safe_val}</b>')
    html = " &nbsp;•&nbsp; ".join(bits)
    st.markdown(
        f'<div style="margin-top:6px;color:#334155;font-size:13px;">{html}</div>',
        unsafe_allow_html=True,
    )

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

def _as_mapping(result: Any) -> Mapping[str, Any] | None:
    """
    Accept either a dict OR a dataclass-like object (e.g., InsightResult),
    returning a dict-like mapping we can read from safely.
    """
    if result is None:
        return None
    if isinstance(result, Mapping):
        return result
    # dataclass or object with attributes
    try:
        return {
            "company": getattr(result, "company", ""),
            "pdf_url": getattr(result, "pdf_url", ""),
            "risk_factors_markdown": getattr(result, "risk_factors_markdown", ""),
            "table_markdown": getattr(result, "table_markdown", ""),
            "elapsed_seconds": getattr(result, "elapsed_seconds", 0.0),
            "cached": getattr(result, "cached", False),
            "source_type": getattr(result, "source_type", "annual_pdf"),
        }
    except Exception:
        return None

def render_insights(result: Dict[str, Any] | Any | None) -> None:
    """
    Pretty-print the pipeline result (dict or dataclass).
    Expected keys/attrs:
      - company (str)
      - pdf_url (str)
      - risk_factors_markdown (str)
      - table_markdown (str)
      - elapsed_seconds (float)
      - cached (bool)
      - source_type (str)
    """
    res = _as_mapping(result)
    if not res:
        st.info("No insights to display.")
        return

    company = res.get("company") or ""
    pdf_url = res.get("pdf_url") or ""
    # risk_md = (res.get("risk_factors_markdown") or "").strip()
    table_md = (res.get("table_markdown") or "").strip()
    elapsed = float(res.get("elapsed_seconds") or 0.0)
    cached = bool(res.get("cached"))
    source_type = (res.get("source_type") or "annual_pdf").replace("_", " ")

    # Header row: company + meta box
    st.markdown(
        f"""
        <div style="
          display:flex;justify-content:space-between;align-items:center;
          background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
          padding:10px 12px;margin-bottom:10px;">
          <div style="display:flex;align-items:center;gap:10px;">
            <img src="https://img.icons8.com/?size=100&id=114049&format=png&color=000000" width="22" height="22" alt="info">
            <div style="font-size:16px;font-weight:700;">Insights for {company}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            ("Source", source_type),
        ])

    st.markdown(
        "<hr style='border:none;border-top:1px solid #e5e7eb;margin:10px 0;'/>",
        unsafe_allow_html=True,
    )

    # Risk factors
    # _section_header("Risk Factors", "⚠️")
    # if risk_md:
    #     st.markdown(risk_md)
    # else:
    #     st.caption("No risk factors extracted.")

    # st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Pain points table
    _section_header("Pain Points / Functions / Regions", "📊")
    if table_md:
        st.markdown(table_md)
    else:
        st.caption("No table extracted.")

    st.markdown(
        "<div style='opacity:.7;font-size:12px;margin-top:10px;'>"
        "Generated from annual report excerpts."
        "</div>",
        unsafe_allow_html=True,
    )
