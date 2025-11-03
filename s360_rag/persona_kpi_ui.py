"""
s360_rag/persona_kpi_ui.py
-------------------------------------------------
UI component to generate persona KPIs and display
top relevant impacts using the RAG pipeline.
Integrates seamlessly with main_app.
"""

import streamlit as st
from typing import Dict
from s360_rag.persona_builder import build_persona_prompt
from s360_rag.pipelines import run_persona_rag_pipeline
import re

# ======================================================
# 🎨 Helper: Dynamic Card Renderer
# ======================================================
def _render_impact_cards(top_impacts: list):
    """Render impact pointers as stacked modern cards with hyperlink detection."""
    if not top_impacts:
        st.warning("No relevant impacts found.")
        return

    url_pattern = re.compile(r'^(https?://[^\s]+)$', re.IGNORECASE)

    for i, item in enumerate(top_impacts, start=1):
        source_val = item.get("FileName", "–")
        # Detect if FileName is a valid URL
        if isinstance(source_val, str) and re.match(url_pattern, source_val.strip()):
            source_display = f"<a href='{source_val.strip()}' target='_blank' style='color:#2563eb;text-decoration:none;'>Case Study Link</a>"
        else:
            source_display = source_val or "–"

        with st.container(border=True):
            st.markdown(
                f"""
                <div style="
                    background-color:#e0f2fe;
                    border-radius:12px;
                    padding:14px 18px;
                    margin-bottom:12px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.08);
                    border:1px solid #e5e7eb;">
                    <div style="font-weight:600;color:#1f2937;font-size:15px;margin-bottom:8px;">
                        Impact {i}
                    </div>
                    <div style="font-size:15px;line-height:1.5;color:#111827;margin-bottom:10px;font-weight:500;">
                        {item.get('Impact','–')}
                    </div>
                    <div style="font-size:13px;color:#4b5563;">
                        🌐 <b>Industry:</b> {item.get('Industry','–')}<br>
                        🧭 <b>Business Group:</b> {item.get('BusinessGroup','–')}<br>
                        💼 <b>Use Case:</b> {item.get('UseCase','–')}<br>
                        📁 <b>Source:</b> {source_display}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
# ======================================================
# 🚀 Main Persona KPI + Impact Renderer
# ======================================================
def render_persona_kpi_block(persona_data: Dict):
    """
    1️⃣ Build persona_info
    2️⃣ Use cached results if available
    3️⃣ Generate KPIs & impacts only when needed
    """

    # st.markdown("---")

    persona_name = persona_data.get("client_name", "Unknown Persona")
    persona_info, clean_json = build_persona_prompt(persona_data)
    if not persona_info:
        st.warning("Persona data is empty or invalid.")
        return

    # ======================================================
    # 🧠 Cache Key Logic
    # ======================================================
    cache_key = f"persona_{hash(persona_info)}"
    if "rag_cache" not in st.session_state:
        st.session_state["rag_cache"] = {}

    # ======================================================
    # 🔁 Use Cached Insights if Present
    # ======================================================
    if cache_key in st.session_state["rag_cache"]:
        results = st.session_state["rag_cache"][cache_key]
        kpis = results.get("persona_kpis", [])
    else:
        # ======================================================
        # 🧩 Generate Fresh Insights
        # ======================================================
        with st.spinner("Generating KPIs & Impact insights..."):
            try:
                results = run_persona_rag_pipeline(persona_info)
            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
                return

        # Store results to cache for reuse
        st.session_state["rag_cache"][cache_key] = results
        kpis = results.get("persona_kpis", [])
        st.session_state["s360.persona_results"] = results
        st.session_state["s360.persona_kpis"] = kpis
        st.session_state["s360.persona_name"] = persona_name
        st.session_state["s360.persona_info"] = persona_info

    # ======================================================
    # 📊 Render KPI Tags
    # ======================================================
    if kpis:
        kpi_html = " ".join(
            [
                f"<span style='background:#e0f2fe;padding:6px 10px;"
                f"border-radius:10px;margin:3px;display:inline-block;'>{k}</span>"
                for k in kpis
            ]
        )
        st.markdown(kpi_html, unsafe_allow_html=True)
    else:
        st.info("No KPIs found for this persona.")

    # ======================================================
    # 💡 Render Top Impacts
    # ======================================================
    # st.subheader("Relevant Impact Pointers")
        # Layout: header + button aligned to right
    top, spacer, right = st.columns([0.35, 0.35, 0.13])
    with top:
        st.markdown(
            """
            <div style="
                display:flex;align-items:center;
                background:#f8fafc;
                border:1px solid #e2e8f0;
                border-radius:12px;
                padding:10px 14px;
                margin-bottom:8px;">
                <img src="https://img.icons8.com/?size=100&id=0uRhf2mft47s&format=png&color=000000"
                     width="22" height="22" alt="info"
                     style="margin-right:8px;">
                <div style="font-size:16px;font-weight:700;">
                    Relevant Impact Pointers
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    _render_impact_cards(results.get("top_impacts", []))

    # ======================================================
    # 💾 Persist in session_state for reuse by other components
    # ======================================================
    st.session_state["s360.persona_results"] = results
    st.session_state["s360.persona_kpis"] = kpis
    st.session_state["s360.persona_name"] = persona_name
    st.session_state["s360.persona_info"] = persona_info
