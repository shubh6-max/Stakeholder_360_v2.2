# components/impact_cards.py
from __future__ import annotations
from typing import List, Dict, Optional
import streamlit as st
import html

def _pill(text: str) -> str:
    t = html.escape(text)
    return f"""
    <span style="
        display:inline-block;padding:4px 10px;margin:3px 6px 0 0;
        border-radius:999px;border:1px solid #e2e8f0;background:#f8fafc;
        font-size:12px;font-weight:600;color:#0f172a;
    ">{t}</span>"""

def _impact_li(point: str) -> str:
    return f"""
    <li style="margin:0 0 8px 0;line-height:1.35;color:#0f172a;">
        {html.escape(point)}
    </li>"""

def render_impact_results(
    *,
    persona_kpis: List[str],
    results: List[Dict],            # each: {case_title, impact_pointers, reason, source_file, score?}
    # title: str = "Persona → Case Study Matches",
    empty_note: Optional[str] = "No impact pointers found for this persona. Showing only KPIs.",
    show_source_badge: bool = True,
    show_score: bool = True,
):
    # st.markdown(f"### {html.escape(title)}")

    # Persona KPI Card
    with st.container(border=True,height=100):
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;margin: -8px 0 8px 0;">
              <img src="https://img.icons8.com/?size=100&id=SYYdTqwzlx8d&format=png&color=1A1A1A" width="18" height="18"/>
              <div style="font-weight:700;">Persona KPIs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if persona_kpis:
            st.markdown(
                "<div style='display:flex;flex-wrap:wrap'>"
                + "".join(_pill(k) for k in persona_kpis)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No KPIs generated.")

    # If no matches, stop here with a friendly note
    if not results:
        if empty_note:
            st.info(empty_note)
        return

    # Grid of result cards
    #  - Use columns to get 1–3 responsive layout
    n = len(results)
    cols = st.columns(min(3, max(1, n)))

    for i, item in enumerate(results):
        col = cols[i % len(cols)]
        with col:
            with st.container(border=True):
                # Header row (icon + title)
                left_icon = "https://img.icons8.com/?size=100&id=0uRhf2mft47s&format=png&color=1A1A1A"
                title = html.escape(item.get("case_title") or "Case Study")
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:10px;margin:-8px 0 6px 0;">
                      <img src="{left_icon}" width="18" height="18"/>
                      <div style="font-weight:700;">{title}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Source + Score badges
                badges = []
                if show_source_badge and item.get("source_file"):
                    badges.append(_pill(item["source_file"].split("/")[-1]))
                # if show_score and item.get("max_sim") is not None:
                    # badges.append(_pill(f"sim: {item['max_sim']:.2f}"))
                if badges:
                    st.markdown(
                        "<div style='display:flex;flex-wrap:wrap;margin-bottom:6px'>"
                        + "".join(badges)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                # Impact pointers
                points = item.get("impact_pointers") or []
                if points:
                    st.markdown(
                        "<ul style='padding-left:18px;margin:6px 0 6px 0;'>"
                        + "".join(_impact_li(p) for p in points)
                        + "</ul>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No impact pointers extracted.")

        # Reason
    with st.container(border=True):
        reason = item.get("reason") or ""
        if reason:
            st.markdown(
                "<div style='font-size:12px;'><b>Why this matches:</b></div>"
                f"<div style='font-size:12px;color:#475569'>{html.escape(reason)}</div>",
                unsafe_allow_html=True,
            )
