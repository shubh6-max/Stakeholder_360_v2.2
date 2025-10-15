# components/sections_readonly.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st

# Reuse field configs so labels/ordering match the edit page
from components.aggrid_sections_all import LEFT_SECTIONS, RIGHT_SECTIONS


def _fmt_cell(v: Any, label: str) -> str:
    """Uniform display formatting for read-only tables — handles NaN, None, and empty values robustly."""

    # Convert to string once safely
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return "–"

    s = str(v).strip()

    # Handle stringified NaN/None/Null cases
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return "–"

    # clickable LinkedIn links
    if "linkedin" in label.lower() and s:
        return f"<a href='{s}' target='_blank' style='color:#1a0dab;text-decoration:underline'>{s}</a>"

    # mailto for emails
    if "email" in label.lower() and "@" in s:
        return f"<a href='mailto:{s}' style='color:#1a0dab;text-decoration:underline'>{s}</a>"

    return s


def _table_html(
    title: str,
    pairs: List[Tuple[str, str]],
    *,
    header_bg: str = "#dbeafe",
    field_bg: str = "#f7f9fc",
    gap_px: int = 8,
) -> str:
    """Builds a section header and a compact 2-col table."""
    head = (
        f"<div style='background:{header_bg};border:1px solid #cfe0ff;"
        f"border-radius:8px 8px 0 0;padding:8px 12px;font-weight:700;color:#1f2a44;'>{title}</div>"
    )
    rows = []
    for field, value in pairs:
        rows.append(
            f"<tr style='border-bottom:1px solid #eee;'>"
            f"<td style='width:40%;padding:8px 10px;background:{field_bg};font-weight:600;"
            f"border-right:1px solid #e7eefc'>{field}</td>"
            f"<td style='padding:8px 10px;background:#fff;'>{value}</td>"
            f"</tr>"
        )
    body = (
        "<table style='width:100%;border:1px solid #e6eaf0;border-top:none;"
        "border-radius:0 0 8px 8px;border-collapse:separate;border-spacing:0;'>"
        + "".join(rows)
        + "</table>"
    )
    wrapper = f"<div style='margin:0 0 {gap_px}px 0;'>{head}{body}</div>"
    return wrapper


def _pairs_for_section(row: pd.Series | Dict[str, Any],
                       fields: List[Tuple[str, str, str]]) -> List[Tuple[str, str]]:
    get = row.get if isinstance(row, dict) else row.get
    pairs: List[Tuple[str, str]] = []
    for label, col, _typ in fields:
        raw = get(col, None)
        pairs.append((str(label), _fmt_cell(raw, label)))
    return pairs


def render_sections_readonly(row: pd.Series | Dict[str, Any],
                             *,
                             gap: int = 10) -> None:
    """Render all profile sections (two columns) as plain HTML tables."""
    if row is None:
        st.info("Select a stakeholder to view details.")
        return

    left, right = st.columns(2, gap="small")

    with left:
        for title, key, fields in LEFT_SECTIONS:
            html = _table_html(title, _pairs_for_section(row, fields), gap_px=gap)
            st.markdown(html, unsafe_allow_html=True)

    with right:
        for title, key, fields in RIGHT_SECTIONS:
            html = _table_html(title, _pairs_for_section(row, fields), gap_px=gap)
            st.markdown(html, unsafe_allow_html=True)
