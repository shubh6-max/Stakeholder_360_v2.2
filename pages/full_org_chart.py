# pages/full_org_chart.py  (drop-in)
import streamlit as st
import pandas as pd
from utils.layout import apply_global_style
from utils.auth import verify_jwt_verbose
from utils.db import get_engine
from features.orgchart.builder import build_ceo_path_with_reportees_tree
from features.orgchart.renderer import render_org_tree
from sqlalchemy import text

st.set_page_config(page_title="Full Org Chart | Stakeholder 360", layout="wide")
apply_global_style()

# ---------- Query params ----------
params = dict(st.query_params)
token = params.get("token")
person_id = params.get("person_id")

if not token:
    st.error("Unauthorized: missing token."); st.stop()

claims, reason = verify_jwt_verbose(token)
if not claims:
    st.error("Session expired." if (reason or "").startswith("expired") else "Invalid token."); st.stop()

icon_url = "https://img.icons8.com/?size=100&id=11269&format=png&color=000000"

st.markdown(
    f"""
    <h2 style="display:flex; align-items:center; gap:8px; margin:0;">
        <img src="{icon_url}" width="34" height="34" style="vertical-align:middle;">
        Full Org Chart
    </h2>
    """,
    unsafe_allow_html=True
)
# st.caption(f"Authorized user: {claims.get('sub')} · role: {claims.get('role')}")

if not person_id:
    st.warning("No person selected."); st.stop()

# ---------- Load data ----------
NEEDED_COLS = [
    "client_name", "client_designation",
    "reporting_manager", "reporting_manager_designation",
    "email_address", "business_unit",
]

@st.cache_data(ttl=180, show_spinner=False)
def load_all_minimal() -> pd.DataFrame:
    eng = get_engine()
    sql = f"SELECT {', '.join(NEEDED_COLS)} FROM scout.centralize_db"
    with eng.begin() as conn:
        df = pd.read_sql(sql, conn)
    for c in NEEDED_COLS:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

@st.cache_data(ttl=120, show_spinner=False)
def load_match(person_key: str) -> pd.DataFrame:
    eng = get_engine()
    sql = text(f"""
        SELECT {", ".join(NEEDED_COLS)}
        FROM scout.centralize_db
        WHERE email_address = :pid OR client_name = :pid
        LIMIT 1
    """)
    with eng.begin() as conn:
        res = conn.execute(sql, {"pid": person_key})
        rows = res.mappings().all()
        return pd.DataFrame(rows)

all_df = load_all_minimal()
seed_df = load_match(str(person_id).strip())

if seed_df.empty:
    st.warning(f"No matching record found for: `{person_id}`"); st.stop()

seed_row = seed_df.iloc[0]

# ---------- Build CEO→…→persona path + attach direct reportees ----------
tree_data = build_ceo_path_with_reportees_tree(all_df, seed_row, max_hops=12, include_reports_depth=1)
render_org_tree(tree_data, height=560, title=None)

# with st.expander("Record (debug)", expanded=False):
#     st.dataframe(seed_df, use_container_width=True)
