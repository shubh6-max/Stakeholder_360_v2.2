# pages/insights.py
from __future__ import annotations

import streamlit as st

from utils.layout import apply_global_style, render_topbar
from utils.auth import is_authenticated
from components.insights_view import render_insights
from features.insights.pipeline import run_end_to_end
from features.insights.config import get_settings

st.set_page_config(page_title="Company Insights | Stakeholder 360", layout="wide")
apply_global_style()

# ---- Auth guard ----
if not is_authenticated():
    st.warning("⚠️ Please log in first.")
    st.switch_page("pages/login.py")
    st.stop()

user = st.session_state["user"]
render_topbar(user)

st.markdown("## 🔎 Company Insights (Annual Report Mining)")

with st.alert("Heads-up: this feature calls external services (Tavily, Jina, Azure OpenAI). Your request may be cached for faster re-runs.", type="info"):
    pass

st.markdown("---")

cfg = get_settings()

with st.form("insights_run", clear_on_submit=False):
    colA, colB = st.columns([2, 1])
    with colA:
        company = st.text_input(
            "Company name",
            value="",
            placeholder="e.g., Kellanova, Walmart, AbbVie",
            help="Enter the company whose annual/earnings report you want to mine for insights.",
        )

    with colB:
        force = st.checkbox(
            "Force fresh run (ignore cache)",
            value=False,
            help="If checked, we will refresh even if cached insights exist.",
        )

    with st.expander("Advanced options", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            chunk_tokens = st.slider(
                "Chunk size (tokens)",
                min_value=300, max_value=1500, step=50,
                value=int(cfg.chunk_tokens or 900),
                help="How many tokens per chunk for embeddings.",
            )
        with col2:
            top_k = st.slider(
                "Top-K retrieval",
                min_value=3, max_value=20, step=1,
                value=int(cfg.top_k or 12),
                help="How many relevant chunks to retrieve for the LLM.",
            )
        with col3:
            st.caption(
                "Tavily check & Jina trust are controlled by environment variables:\n"
                "- `INSIGHTS_SKIP_TAVILY_CHECK`\n"
                "- `INSIGHTS_TRUST_JINA_URL`"
            )

    run = st.form_submit_button("Run Insights 🚀")

result = None
if run:
    if not company.strip():
        st.error("Please enter a company name.")
    else:
        result = run_end_to_end(
            company.strip(),
            force_rerun=force,
            top_k=int(top_k),
            chunk_tokens=int(chunk_tokens),
            by_user_email=(user or {}).get("email", ""),
        )

st.markdown("---")
render_insights(result)
