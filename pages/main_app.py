# pages/main_app.py
import pandas as pd
import streamlit as st
from typing import List
from urllib.parse import urlencode

from utils.layout import apply_global_style, render_topbar
from utils.auth import is_authenticated, logout, issue_jwt, secret_fingerprint
from utils.db import get_engine

from features.orgchart.builder import (
    build_upward_chain_to_ceo,   # use CEO chain
    choose_person_id,
)
from features.orgchart.renderer import render_upward_graph

from components.details import render_avatar_only, render_info_only

from components.aggrid_sections_all import render_all_sections

from components.insights_embed import render_company_insights_inline

# ==============================================================================
from utils.page_config import set_common_page_config
set_common_page_config(page_title="Stakeholder 360", layout="wide")
# st.set_page_config(page_title="Stakeholder 360", layout="wide")
apply_global_style()

# ---- Auth guard ----
if not is_authenticated():
    st.warning("⚠️ Please log in first.")
    st.switch_page("pages/login.py")
    st.stop()

user = st.session_state["user"]
render_topbar(user)

# -----------------------------
# Data: load once, filter in-app
# -----------------------------
NEEDED_COLS: List[str] = [
    "account",
    "working_group",
    "business_unit",                 # BU needed for coloring
    "service_line",
    "client_name",
    "client_designation",
    "reporting_manager",
    "reporting_manager_designation",
    "email_address",
    "last_update_date",
    "subsidiary",
]

@st.cache_data(ttl=180, show_spinner=False)
def load_centralize_df() -> pd.DataFrame:
    eng = get_engine()
    sql = f"SELECT {', '.join(NEEDED_COLS)} FROM scout.centralize_db"
    with eng.begin() as conn:
        df = pd.read_sql(sql, conn)
    for c in NEEDED_COLS:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

df = load_centralize_df()

def options_for(col: str, frame: pd.DataFrame) -> List[str]:
    vals = sorted([v for v in frame[col].unique().tolist() if v])
    return ["All"] + vals

def apply_filters(frame: pd.DataFrame, **sel) -> pd.DataFrame:
    out = frame
    for k, v in sel.items():
        if v and v != "All":
            out = out[out[k] == v]
    return out

# -----------------------------
# Interdependent filters
# -----------------------------
# st.markdown("### Filters")

ss = st.session_state
ss.setdefault("flt_account", "All")
ss.setdefault("flt_working_group", "All")
ss.setdefault("flt_business_unit", "All")
ss.setdefault("flt_service_line", "All")
ss.setdefault("flt_client_name", "All")
ss.setdefault("flt_subsidiary", "All")

base = df

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    sel_account = st.selectbox("Account", options_for("account", base), key="flt_account")
flt1 = apply_filters(base, account=sel_account)

with c2:
    sel_subsidiary = st.selectbox("Subsidiary/Vertical", options_for("subsidiary", flt1), key="flt_subsidiary")
flt2 = apply_filters(flt1, subsidiary=sel_subsidiary)

with c3:
    sel_working_group = st.selectbox("Working Group", options_for("working_group", flt1), key="flt_working_group")
flt3 = apply_filters(flt1, working_group=sel_working_group)

with c4:
    sel_business_unit = st.selectbox("Business Unit", options_for("business_unit", flt2), key="flt_business_unit")
flt4 = apply_filters(flt2, business_unit=sel_business_unit)

with c5:
    sel_service_line = st.selectbox("Service Line", options_for("service_line", flt3), key="flt_service_line")
flt5 = apply_filters(flt3, service_line=sel_service_line)

with c6:
    sel_client = st.selectbox("Client Name", options_for("client_name", flt4), key="flt_client_name")
flt6 = apply_filters(flt4, client_name=sel_client)

# --- Ready gate: don't render charts/details until a client is selected ---
is_ready = all(
    v != "All"
    for v in [sel_account, sel_client]
)

if not is_ready:
    st.markdown(
        """
        <div style="display:flex;justify-content:center;align-items:center;height:220px;">
            <div style="text-align:center;opacity:.8">
                <div style="font-size:42px;line-height:1.0;margin-bottom:6px;">🧭</div>
                <div style="font-weight:700;font-size:18px;">Select stakeholder to proceed</div>
                <div style="font-size:13px;">Choose <b>Account</b> in filters, including <b>Client Name</b>.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Early exit: keep filters visible but skip the rest of the page
    st.stop()

# -----------------------------
# Two-column layout
# -----------------------------
# one knob to control both columns
CHART_HEIGHT = 500
AVATAR_H = int(CHART_HEIGHT * 0.75)
INFO_H   = CHART_HEIGHT - AVATAR_H


left, right = st.columns([1, 1] )

with left:
    icon_url = "https://img.icons8.com/?size=100&id=11269&format=png&color=000000"

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:8px;">
            <img src="{icon_url}" width="24" height="24">
            <h3 style="margin:0;">Upward Org Chart</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True,height=580):
        row = flt6.iloc[0] if not flt6.empty else (flt5.iloc[0] if not flt5.empty else None)

        if row is not None:
            # CEO chain (persona -> ... -> CEO) then reverse to CEO -> ... -> persona
            chain_rich = build_upward_chain_to_ceo(df, row, max_hops=12)
            chain_linear = list(reversed([(n["name"], n["designation"]) for n in chain_rich]))
            render_upward_graph(chain_linear, height=CHART_HEIGHT, title=None)

            # Link to Full Org Chart (unchanged)
            person_id = choose_person_id(row)
            token = issue_jwt(user)
            qs = urlencode({"person_id": str(person_id), "token": token})
            href = f"./full_org_chart?{qs}"
            icon_url = "https://img.icons8.com/?size=100&id=DvUq7gVLNGVC&format=png&color=000000"

            st.markdown(
                f"""
                <div style="margin-top:-9px;">   <!-- negative pushes upward -->
                    <a href="{href}" target="_blank">
                        <button style="display:flex; align-items:center; gap:6px; padding:6px 10px;">
                            <img src="{icon_url}" width="18" height="18">
                            View Full Org Chart
                        </button>
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )


        else:
            st.info("Adjust filters to select a client.")
with right:
    icon_url = "https://img.icons8.com/?size=100&id=114049&format=png&color=000000"

    st.markdown(
        f"""
        <h3 style="display:flex; align-items:center; gap:8px; margin:0;">
            <img src="{icon_url}" width="22" height="22" style="vertical-align:middle;">
            Details
        </h3>
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True,height=580):
        if 'row' in locals() and row is not None:
            person_name = row.get("client_name", "")

            st.markdown("")
            st.markdown("")
            # Row 1: only profile picture
            render_avatar_only(
                person_name,
                height=AVATAR_H,
                avatar_size=int(AVATAR_H * 0.9),   # <<< new
            )
            render_info_only(
                person_name,
                height=INFO_H,
                show_card=True,
                max_width=520,
            )
            # st.markdown("")
        else:
            st.caption("Select a client to view LinkedIn details.")

# ---- Full-width editable section (below both columns) ----
st.markdown("---")
with st.container(border=True):
    if 'row' in locals() and row is not None:
        # row can be a pandas Series from your filtered df; it must include 'last_update_date'
        render_all_sections(
            row,
            label_width=220,   # tighter label column
            card_gap=12,       # smaller vertical spacing between cards
            )
    else:
        st.info("Select a client above to view details.")


# --- After your two columns & editable tables ---
st.markdown("---")
st.markdown("## Insights")

if 'row' in locals() and row is not None:
    # Decide the company name for insights (we'll use 'account' as the company)
    company_name = (row.get("account") or "").strip()
    # Optional: tie insights to the focal person for KPI generation later
    focal_person_id = choose_person_id(row)  # email if available, else client_name
    render_company_insights_inline(
        company=company_name,
        person_id=str(focal_person_id),
        default_top_k=12,
        default_chunk_tokens=900,
    )
else:
    st.info("Select a client to view company insights.")


st.divider()
if st.button("Logout"):
    logout()
    st.success("You have been logged out.")
    st.switch_page("pages/login.py")
