# components/persona_kpi_cards.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import math

# -----------------------------
# Small helpers
# -----------------------------
from contextlib import contextmanager

@contextmanager
def _sized_card(min_height_px: int):
    st.markdown(f"<div style='min-height:{min_height_px}px'>", unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)

def _inline_meta(label: str, items: List[str]) -> None:
    items = [s for s in (items or []) if s.strip()]
    if not items:
        return
    st.markdown(
        f"""
        <div style="
            display:flex;align-items:center;gap:18px;
            margin:4px 0 10px 0; line-height:2;">
            <span style="
                font-size:12px;
                font-weight:800;
                color:#0f172a;
                background:#eef2ff;
                border:1px solid #dbeafe;
                padding:2px 8px;
                border-radius:5px;">{label}</span>
            <span style="font-size:12px;color:#334155;">
                {', '.join(items)}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _func_blocks_from_preview(preview: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Preview JSON → [{function, kpis[], industry[]}, ...]"""
    blocks: List[Dict[str, Any]] = []
    if not isinstance(preview, dict):
        return blocks
    for fn_label, payload in preview.items():
        if not isinstance(payload, dict):
            continue
        kpis = payload.get("strategic_kpis") or []
        inds = payload.get("Industry") or []
        blocks.append({
            "function": str(fn_label).strip(),
            "kpis": [str(k).strip() for k in kpis if str(k).strip()],
            "industry": [str(i).strip() for i in inds if str(i).strip()],
        })
    return blocks

def _chunk(lst: List[str], ncols: int) -> List[List[str]]:
    """Split a list into ncols columns (balanced)."""
    if ncols <= 1 or not lst:
        return [lst]
    # round up rows
    rows = math.ceil(len(lst) / ncols)
    cols: List[List[str]] = []
    for i in range(ncols):
        cols.append(lst[i*rows:(i+1)*rows])
    return cols

def _pill(text: str):
    st.markdown(
        f"""
        <span style="
            display:inline-block;
            padding:4px 10px;
            border-radius:999px;
            border:1px solid #e2e8f0;
            background:#f8fafc;
            font-size:11px;
            font-weight:600;
            color:#1e293b;
            margin-right:6px;
            margin-bottom:6px;
        ">{text}</span>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Public renderers
# -----------------------------
st.markdown("""
<style>
/* Softer container look */
[data-testid="stContainer"] > div[style*="border: 1px"]{
  border-radius: 14px !important;
  border: 1px solid #e5e7eb !important;
  box-shadow: 0 6px 16px rgba(15,23,42,0.06) !important;
  padding: 14px !important;
}
</style>
""", unsafe_allow_html=True)

def render_persona_kpi_preview(
    preview: Dict[str, Any],
    *,
    card_cols: int = 3,   # how many columns of cards across
    kpi_cols: int = 2,    # how many columns of KPI bullets inside each card
    max_kpis_per_card: int = 7,
) -> None:
    """
    Render Persona Functions & KPIs as cards laid out across 'card_cols' columns.
    Inside each card, the KPI bullets are arranged into 'kpi_cols' columns.
    """
    # st.markdown("#### Persona Functions & KPIs")
    blocks = _func_blocks_from_preview(preview)
    if not blocks:
        st.info("No persona KPIs generated yet. Click **Generate** above.")
        return

    # Lay cards in streamlit columns (round-robin)
    columns = st.columns(max(1, card_cols))
    # st.write(max(1, card_cols))
    for idx, b in enumerate(blocks):
        col = columns[idx % card_cols]
        with col:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <div style="width:10px;height:10px;border-radius:999px;background:#2563eb;opacity:.9;"></div>
                        <div style="font-weight:800;color:#0f172a;">{b['function'] or 'Business Function'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Industry chips
                inds = [s for s in (b["industry"] or []) if s.strip()]
                _inline_meta("Industry focus:", inds)

                # Dynamically size card based on industry count (each chip row ~ 24px)
                # Base height covers title + margins; add 22px per extra industry item.
                base = 0
                extra = max(0, len(inds) - 1) * 22
                min_h = base + extra

                # ── KPIs: single vertical list (no columns)
                kpis = (b["kpis"] or [])[:max_kpis_per_card]

                with _sized_card(min_h):
                    if not kpis:
                        st.write("- _No KPIs_")
                    else:
                        for item in kpis:
                            st.write(f"- {item}")



def render_impacts_block(
    impacts: List[str],
    best_src: Optional[Dict[str, Any]],
    *,
    impact_cols: int = 2,      # columns for impact bullets
    show_source: bool = True,
) -> None:
    """
    Render Top Impact Pointers in 'impact_cols' columns.
    Shows a compact source chip if best_src is provided.
    """
    st.markdown("#### Results")
    if not impacts:
        st.info("No impact pointers found. Top suggested KPIs are shown above.")
        return

    # Impacts across columns
    cols = st.columns(max(1, impact_cols),gap="medium")
    chunks = _chunk(impacts, max(1, impact_cols))
    with st.container(border=True):
        for i, c in enumerate(cols):
            with c:
                for j, item in enumerate(chunks[i], 1):
                    st.write(f"{item}")

        if show_source and best_src:
            meta = best_src.get("metadata") or {}
            file_name = meta.get("file_name") or "—"
            case_name = meta.get("case_study_name") or "Case Study"
            spans = meta.get("spans") or []
            slides = ", ".join([f"Slide {s.get('slide_index')}" for s in spans]) if spans else "—"

            st.markdown(
                f"""
                <div style="margin-top:10px;color:#475569;font-size:12px;">
                    <span style="display:inline-flex;align-items:center;gap:6px;
                                 background:#f1f5ff;border:1px solid #dbeafe;color:#1e40af;
                                 padding:4px 8px;border-radius:999px;font-size:11px;font-weight:700;">
                        <span style="width:8px;height:8px;border-radius:999px;background:#60a5fa;"></span>
                        Source
                    </span>
                    &nbsp; {file_name} • {case_name} • {slides}
                </div>
                """,
                unsafe_allow_html=True,
            )
