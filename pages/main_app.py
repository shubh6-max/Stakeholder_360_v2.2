# pages/main_app.py
from typing import List, Dict
import pandas as pd
import streamlit as st
from urllib.parse import urlencode
import traceback
from utils.page_config import set_common_page_config
from utils.layout import apply_global_style, render_topbar
from utils.auth import is_authenticated, logout, issue_jwt
from utils.db import get_engine

from features.orgchart.builder import (
    build_upward_chain_to_ceo,
    choose_person_id,
)
from features.orgchart.renderer import render_upward_graph

from components.details import render_avatar_only, render_info_only
# from components.aggrid_sections_all import render_all_sections

from components.sections_readonly import render_sections_readonly
# from urllib.parse import urlencode, quote

# from components.insights_embed import render_company_insights_inline

from features.insights.retrieve import run_kpis_for_persona
from components.kpi_view import render_kpis

from components.insights_view import render_insights
from features.insights.pipeline import run_end_to_end  # uses DB-first, creates embeddings as needed

# ------------------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------------------
set_common_page_config(page_title="Stakeholder 360", layout="wide")
apply_global_style()

# ---- Auth guard ----
if not is_authenticated():
    st.warning("⚠️ Please log in first.")
    st.switch_page("pages/login.py")
    st.stop()

user = st.session_state["user"]
render_topbar(user)

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
NEEDED_COLS: List[str] = [
    "account",
    "subsidiary",
    "working_group",
    "business_unit",
    "service_line",
    "client_name",
    "client_designation",
    "reporting_manager",
    "reporting_manager_designation",
    "email_address",
    "last_update_date",
    # Below are used in editable sections (some may be empty in DB now)
    "lead_priority",
    "csl_owner",
    "reachout_channel",
    "reachout_lever",
    "status",
    "location",
    "seniority_level",
    "vendor_name",
    "contractor_count",
    "scout_linkedin_connected_flag",
    "pursued_in_past",
    "context",
    "first_outreach_date",
    "last_outreach_date",
]

@st.cache_data(ttl=180, show_spinner=False)
def load_centralize_df() -> pd.DataFrame:
    eng = get_engine()
    sql = f"SELECT {', '.join(NEEDED_COLS)} FROM scout.centralize_db"
    with eng.begin() as conn:
        df = pd.read_sql(sql, conn)
    # Normalize strings
    for c in NEEDED_COLS:
        if c in df.columns and df[c].dtype == "object":
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

df = load_centralize_df()

def options_for(col: str, frame: pd.DataFrame) -> List[str]:
    vals = sorted([v for v in frame[col].dropna().astype(str).str.strip().unique() if v])
    return ["All"] + vals

def apply_filters(frame: pd.DataFrame, **sel) -> pd.DataFrame:
    out = frame
    for k, v in sel.items():
        if v and v != "All":
            out = out[out[k] == v]
    return out

def build_option_catalog(source_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Dynamic dropdown options for AgGrid editors (merged with component defaults)."""
    def vals(col):
        if col not in source_df.columns:
            return []
        return sorted([v for v in source_df[col].dropna().astype(str).str.strip().unique() if v])

    return {
        "working_group": vals("working_group"),
        "business_unit": vals("business_unit"),
        "service_line": vals("service_line"),
        "csl_owner": vals("csl_owner"),
        "reachout_channel": vals("reachout_channel"),
        "reachout_lever": vals("reachout_lever"),
        "status": vals("status"),
        "lead_priority": vals("lead_priority"),
    }

# ------------------------------------------------------------------------------
# Interdependent filters (each one filters the next)
# ------------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("flt_account", "All")
ss.setdefault("flt_subsidiary", "All")
ss.setdefault("flt_working_group", "All")
ss.setdefault("flt_business_unit", "All")
ss.setdefault("flt_service_line", "All")
ss.setdefault("flt_client_name", "All")

base = df

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    sel_account = st.selectbox("Account", options_for("account", base), key="flt_account")
flt_a = apply_filters(base, account=sel_account)

with c2:
    sel_subsidiary = st.selectbox("Subsidiary/Vertical", options_for("subsidiary", flt_a), key="flt_subsidiary")
flt_b = apply_filters(flt_a, subsidiary=sel_subsidiary)

with c3:
    sel_working_group = st.selectbox("Working Group", options_for("working_group", flt_b), key="flt_working_group")
flt_c = apply_filters(flt_b, working_group=sel_working_group)

with c4:
    sel_business_unit = st.selectbox("Business Unit", options_for("business_unit", flt_c), key="flt_business_unit")
flt_d = apply_filters(flt_c, business_unit=sel_business_unit)

with c5:
    sel_service_line = st.selectbox("Service Line", options_for("service_line", flt_d), key="flt_service_line")
flt_e = apply_filters(flt_d, service_line=sel_service_line)

with c6:
    sel_client = st.selectbox("Client Name", options_for("client_name", flt_e), key="flt_client_name")
flt_f = apply_filters(flt_e, client_name=sel_client)

# ---- Ready gate: require a specific client selection
is_ready = (sel_account != "All") and (sel_client != "All")

if not is_ready:
    st.markdown(
        """
        <div style="display:flex;justify-content:center;align-items:center;height:220px;">
            <div style="text-align:center;opacity:.85">
                <div style="font-size:42px;line-height:1.0;margin-bottom:6px;">🧭</div>
                <div style="font-weight:700;font-size:18px;">Select stakeholder to proceed</div>
                <div style="font-size:13px;">Choose an <b>Account</b> and a <b>Client Name</b> from the filters above.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown("---")
# ------------------------------------------------------------------------------
# Two-column layout (Org chart + LinkedIn panel)
# ------------------------------------------------------------------------------
CHART_HEIGHT = 500
AVATAR_H = int(CHART_HEIGHT * 0.75)
INFO_H   = CHART_HEIGHT - AVATAR_H

left, right = st.columns([1, 1])

with left:
    icon_url = "https://img.icons8.com/?size=100&id=11269&format=png&color=000000"
    # st.markdown(
    #     f"""
    #     <div style="display:flex; align-items:center; gap:8px;">
    #         <img src="{icon_url}" width="24" height="24">
    #         <h3 style="margin:0;">Upward Org Chart</h3>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )
    st.markdown(
        f"""
        <div style="
          display:flex;justify-content:space-between;align-items:center;
          background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
          padding:10px 12px;margin-bottom:10px;">
          <div style="display:flex;align-items:center;gap:10px;">
            <img src="https://img.icons8.com/?size=100&id=11269&format=png&color=000000" width="22" height="22" alt="info">
            <div style="font-size:16px;font-weight:700;">Upward Org Chart</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True,height=580):
        row = flt_f.iloc[0] if not flt_f.empty else None
        if row is not None:
            # persona -> ... -> CEO  → reverse to CEO -> ... -> persona
            chain_rich = build_upward_chain_to_ceo(df, row, max_hops=12)
            chain_linear = list(reversed([(n["name"], n["designation"]) for n in chain_rich]))
            render_upward_graph(chain_linear, height=CHART_HEIGHT, title=None)

            # Link to Full Org Chart
            person_id = choose_person_id(row)
            token = issue_jwt(user)
            qs = urlencode({"person_id": str(person_id), "token": token})
            href = f"./full_org_chart?{qs}"

            icon_url = "https://img.icons8.com/?size=100&id=DvUq7gVLNGVC&format=png&color=000000"

            st.markdown(
                f"""
                <style>
                .orgchart-btn {{
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 14px;
                    border: none;
                    border-radius: 10px;             /* rounded-square shape */
                    background: #f1f5ff;             /* default soft blue background */
                    color: #1f2a44;
                    font-weight: 600;
                    font-size: 14px;
                    cursor: pointer;
                    transition: all 0.25s ease;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                }}
                .orgchart-btn:hover {{
                    background: #2563eb;             /* deeper blue on hover */
                    color: #ffffff;                  /* white text on hover */
                    transform: translateY(-2px);
                    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
                }}
                .orgchart-btn img {{
                    width: 18px;
                    height: 18px;
                    transition: filter 0.25s ease;
                }}
                .orgchart-btn:hover img {{
                    filter: brightness(0) invert(1); /* make icon white on hover */
                }}
                </style>

                <div style="margin-top:-19px; display:flex; justify-content:flex-start;">
                    <a href="{href}" target="_blank" style="text-decoration:none;">
                        <button class="orgchart-btn">
                            <img src="{icon_url}" alt="org chart icon"/>
                            <span>View Full Org Chart</span>
                        </button>
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.info("Adjust filters to select a client.")

with right:
    icon_url = "https://img.icons8.com/?size=100&id=114049&format=png&color=000000"
    # st.markdown(
    #     f"""
    #     <h3 style="display:flex; align-items:center; gap:8px; margin:0;">
    #         <img src="{icon_url}" width="12" height="12" style="vertical-align:middle;">
    #         Details
    #     </h3>
    #     """,
    #     unsafe_allow_html=True
    # )
    st.markdown(
        f"""
        <div style="
          display:flex;justify-content:space-between;align-items:center;
          background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
          padding:10px 12px;margin-bottom:10px;">
          <div style="display:flex;align-items:center;gap:10px;">
            <img src="https://img.icons8.com/?size=100&id=Uj9DyJeLazL6&format=png&color=000000" width="22" height="22" alt="info">
            <div style="font-size:16px;font-weight:700;">Linkedin details</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True, height=580):
        if 'row' in locals() and row is not None:
            person_name = row.get("client_name", "")
            render_avatar_only(
                person_name,
                height=AVATAR_H,
                avatar_size=int(AVATAR_H * 0.9),
            )
            render_info_only(
                person_name,
                height=INFO_H,
                show_card=True,
                max_width=520,
            )
        else:
            st.caption("Select a client to view LinkedIn details.")

# ------------------------------------------------------------------------------
# Editable sections (below both columns)
# ------------------------------------------------------------------------------
# st.markdown("---")
# ---- Full-width read-only sections (below both columns) ----
with st.container(border=True, height="content"):
    if 'row' in locals() and row is not None:
        email   = (row.get("email_address") or "").strip()
        client  = (row.get("client_name") or "").strip()
        account = (row.get("account") or "").strip()

        # include a short-lived token so a new tab can open without session hitch
        token = issue_jwt(user)  # your helper (e.g., 120 min expiry)

        if email:
            qs = urlencode({"email": email, "token": token})
        else:
            qs = urlencode({"client": client, "account": account, "token": token})

        href = f"./edit_profile?{qs}"

        st.markdown(
    f"""
    <style>
    .edit-btn {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border: none;
        border-radius: 10px;             /* rounded-square look */
        background: #f1f5ff;             /* default soft blue background */
        color: #1f2a44;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.25s ease;      /* smooth color & shadow transitions */
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }}
    .edit-btn:hover {{
        background: #2563eb;             /* hover color (deep blue) */
        color: #ffffff;                  /* white text on hover */
        transform: translateY(-2px);     /* subtle lift */
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }}
    .edit-btn img {{
        width: 18px;
        height: 18px;
        transition: filter 0.25s ease;
    }}
    .edit-btn:hover img {{
        filter: brightness(0) invert(1); /* make icon white on hover */
    }}
    </style>

    <div style="display:flex;justify-content:flex-end;margin-bottom:8px;">
      <a href="{href}" target="_blank" style="text-decoration:none;">
        <button class="edit-btn">
          <img src="https://img.icons8.com/?size=100&id=82787&format=png&color=000000" alt="edit icon"/>
          <span>Edit tables</span>
        </button>
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)



        # render read-only sections (HTML tables)
        render_sections_readonly(row, gap=8)
    else:
        st.info("Select a client above to view details.")

# --- Insights (RAG) + KPIs (single button) ---
st.markdown("---")
persona_row = row.to_dict()
company_name = (row.get("account") or "").strip()

if st.button("Get insights"):
    try:
        # (Optional) ensure annual report exists & cache the pain-points table on first run
        # This happens implicitly when run_kpis_for_persona calls ensure_company_indexed().
        kpi_result = run_kpis_for_persona(company_name, persona_row, top_k=8)
        with st.container(border=True):
            render_kpis(kpi_result)
    except Exception:
        st.error("KPI generation failed — full traceback below:")
        st.code(traceback.format_exc(), language="python")

# ------------------------------------------------------------------------------

st.divider()
if st.button("Logout"):
    logout()
    st.success("You have been logged out.")
    st.switch_page("pages/login.py")
