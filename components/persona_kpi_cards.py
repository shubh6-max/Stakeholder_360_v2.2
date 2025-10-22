# components/persona_kpi_cards.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager
import streamlit as st
import math

# ──────────────────────────────────────────────────────────────────────────────
# CSS (soft card look)
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
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
                font-size:12px;font-weight:800;color:#0f172a;
                background:#eef2ff;border:1px solid #dbeafe;
                padding:2px 8px;border-radius:5px;">{label}</span>
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
    if ncols <= 1 or not lst:
        return [lst]
    rows = math.ceil(len(lst) / ncols)
    return [lst[i*rows:(i+1)*rows] for i in range(ncols)]

# Priority colors
_PRIORITY_COLORS = {1: "#ef4444", 2: "#f59e0b", 3: "#3b82f6"}  # red, yellow, blue
_DEFAULT_DOT = "#94a3b8"  # grey

def _dot_color_for_rank(rank: Optional[int]) -> str:
    return _PRIORITY_COLORS.get(rank, _DEFAULT_DOT)

def _auto_card_cols(blocks: List[Dict[str, Any]], max_cols: int = 3) -> int:
    """Auto-detect number of columns based on content density."""
    if not blocks:
        return 1
    n_cards = len(blocks)
    max_kpis = max(len(b.get("kpis", [])) for b in blocks)
    max_inds = max(len(b.get("industry", [])) for b in blocks)

    if n_cards <= 2:
        return n_cards
    if max_kpis >= 10 or max_inds >= 5:
        cols = 1
    elif max_kpis >= 6 or max_inds >= 3:
        cols = 2
    else:
        cols = 3
    return max(1, min(cols, max_cols, n_cards))

# ──────────────────────────────────────────────────────────────────────────────
# Public renderers
# ──────────────────────────────────────────────────────────────────────────────
def render_persona_kpi_preview(
    preview: Dict[str, Any],
    *,
    card_cols: Optional[int] = None,   # None → auto
    max_cols: int = 3,                 # cap for auto mode
    max_kpis_per_card: int = 7,
    priorities: Optional[Dict[str, int]] = None,  # {function_label: 1..3}
    show_legend: bool = True,          # show color legend below cards
) -> None:
    """
    Render Persona Functions & KPIs as cards.
    - Inline Industry focus
    - Single vertical KPI list
    - Auto columns (1–3)
    - Dynamic dot colors by rank
    - Optional legend
    """
    blocks = _func_blocks_from_preview(preview)
    if not blocks:
        st.info("No persona KPIs generated yet. Click **Generate** above.")
        return

    # Auto-detect columns
    cols_count = card_cols if isinstance(card_cols, int) and card_cols > 0 else _auto_card_cols(blocks, max_cols=max_cols)
    columns = st.columns(cols_count)
    
    # ─── LEGEND (optional) ─────────────────────────────────────────────────────
    if show_legend:
        st.markdown("""
        <div style="display:flex;gap:18px;align-items:center;
                    margin-top:12px;padding:8px 12px;
                    border:1px solid #e2e8f0;border-radius:10px;
                    background:#f8fafc;font-size:12px;color:#334155;">
            <span style="display:flex;align-items:center;gap:6px;">
                <i style="width:10px;height:10px;border-radius:999px;background:#ef4444;"></i>
                Most preferred / top priority
            </span>
            <span style="display:flex;align-items:center;gap:6px;">
                <i style="width:10px;height:10px;border-radius:999px;background:#f59e0b;"></i>
                Medium priority
            </span>
            <span style="display:flex;align-items:center;gap:6px;">
                <i style="width:10px;height:10px;border-radius:999px;background:#3b82f6;"></i>
                Lower but relevant
            </span>
            <span style="display:flex;align-items:center;gap:6px;">
                <i style="width:10px;height:10px;border-radius:999px;background:#94a3b8;"></i>
                Other supporting functions
            </span>
        </div>
        """, unsafe_allow_html=True)

    for idx, b in enumerate(blocks):
        col = columns[idx % cols_count]
        with col:
            with st.container(border=True):
                fn_label = b['function'] or 'Business Function'
                # rank from provided priorities; else 1→3 by order
                rank = priorities.get(fn_label) if isinstance(priorities, dict) else ((idx + 1) if idx < 3 else None)
                dot = _dot_color_for_rank(rank)

                # Title with dot
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <div title="Priority {rank if rank else '—'}"
                             style="width:10px;height:10px;border-radius:999px;background:{dot};opacity:.95;"></div>
                        <div style="font-weight:800;color:#0f172a;">{fn_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Inline industry
                inds = [s for s in (b["industry"] or []) if s.strip()]
                _inline_meta("Industry focus:", inds)

                # Adjust height
                base = 0
                extra = max(0, len(inds) - 1) * 22
                min_h = base + extra

                # KPI list
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
    impact_cols: int = 2,
    show_source: bool = True,
) -> None:
    """Top Impact Pointers + compact source chip."""
    st.markdown("#### Results")
    if not impacts:
        st.info("No impact pointers found. Top suggested KPIs are shown above.")
        return

    cols = st.columns(max(1, impact_cols), gap="medium")
    chunks = _chunk(impacts, max(1, impact_cols))
    with st.container(border=True):
        for i, c in enumerate(cols):
            with c:
                for item in chunks[i]:
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
