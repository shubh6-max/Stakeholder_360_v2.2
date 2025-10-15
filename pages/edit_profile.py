# pages/edit_profile.py
from __future__ import annotations
import math
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import unquote

from utils.page_config import set_common_page_config
from utils.layout import apply_global_style, render_topbar
from utils.db import get_engine
from utils.auth import is_authenticated, verify_jwt

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# persistence helpers (yours)
from features.stakeholders.service import (
    normalize_payload,
    update_centralize_by_identity
)

# ──────────────────────────────────────────────────────────────────────────────
# Page + style
# ──────────────────────────────────────────────────────────────────────────────
set_common_page_config(page_title="Edit Profile", layout="centered")
apply_global_style()

# ──────────────────────────────────────────────────────────────────────────────
# Query params & auth (allow session OR short-lived token ?token=...)
# ──────────────────────────────────────────────────────────────────────────────
qp = st.query_params

def _qp_get_one(name: str) -> Optional[str]:
    v = qp.get(name)
    if isinstance(v, list):
        v = v[0]
    return unquote(v) if v else None

token   = _qp_get_one("token")
email   = _qp_get_one("email")
client  = _qp_get_one("client")
account = _qp_get_one("account")

auth_ok = False
if is_authenticated():
    auth_ok = True
elif token:
    claims = verify_jwt(token)
    auth_ok = bool(claims)

if not auth_ok:
    st.error("Please log in.")
    st.stop()

user = st.session_state.get("user", {})
render_topbar(user)

# ──────────────────────────────────────────────────────────────────────────────
# Icons (shown above buttons)
# ──────────────────────────────────────────────────────────────────────────────
SAVE_ICON = "https://img.icons8.com/?size=100&id=tFxTV6o3zAPd&format=png&color=000000"
DB_ICON   = "https://img.icons8.com/?size=100&id=25010&format=png&color=000000"

# ──────────────────────────────────────────────────────────────────────────────
# Column set & types (from your schema)
# ──────────────────────────────────────────────────────────────────────────────
# Core columns we display/edit
COLUMNS: List[str] = [
    "account","subsidiary","working_group","business_unit","service_line",
    "lead_priority","client_name","client_designation","seniority_level",
    "reporting_manager","reporting_manager_designation",
    "email_address","linkedin_url","location",
    "internal_research","external_research","personalization_notes",
    "vendor_name","contractor_count",
    "reachout_lever","reachout_channel","pursued_in_past","context","introduction_path",
    "mathco_spoc_1","mathco_spoc_2",
    "first_outreach_date","last_outreach_date",   # treat as text unless you confirm real date type
    "last_update_date"                             # for optimistic concurrency (read-only)
]

# DB type map (subset shown in your message; everything else defaults to 'text')
DB_TYPES: Dict[str, str] = {
    "account": "text",
    "file_name": "text",
    "sr_no": "bigint",
    "input_date": "text",
    "last_update_date": "text",
    "subsidiary": "text",
    "working_group": "text",
    "business_unit": "text",
    "lead_priority": "text",
    "client_name": "text",
    "client_designation": "text",
    "seniority_level": "text",
    "service_line": "text",
    "csl_owner": "text",
    "reporting_manager": "text",
    "reporting_manager_designation": "text",
    "email_address": "text",
    "linkedin_url": "text",
    "location": "text",
    "internal_research": "text",
    "external_research": "text",
    "personalization_notes": "text",
    "vendor_name": "text",
    "contractor_count": "bigint",
    "reachout_lever": "text",
    "reachout_channel": "text",
    "pursued_in_past": "text",
    "context": "text",
    "introduction_path": "text",
    "mathco_spoc_1": "text",
    "mathco_spoc_2": "text",
    # not listed but present in COLUMNS – default to text:
    "first_outreach_date": "text",
    "last_outreach_date": "text",
}

NUMERIC_TYPES = {"bigint", "int", "integer", "smallint", "real", "double", "numeric", "decimal", "float"}

def is_numeric_col(col: str) -> bool:
    t = DB_TYPES.get(col, "text").lower()
    return t in NUMERIC_TYPES

# ──────────────────────────────────────────────────────────────────────────────
# Label helpers (snake_case → Title Case) with stable reverse map
# ──────────────────────────────────────────────────────────────────────────────
def snake_to_label(name: str) -> str:
    specials = {"id":"ID","csl":"CSL","url":"URL"}
    parts = name.split("_")
    titled = []
    for i, p in enumerate(parts):
        lp = p.lower()
        if lp in specials:
            titled.append(specials[lp])
        else:
            titled.append(p.capitalize())
    label = " ".join(titled)
    # curated tweaks
    label = (label
             .replace("Linkedin Url", "LinkedIn URL")
             .replace("Business Unit", "Business Functions")
             .replace("Reporting Manager Designation", "Reporting Manager Title")
             )
    return label

def build_label_maps(cols: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    col2label = {c: snake_to_label(c) for c in cols if c != "last_update_date"}
    seen = {}
    for c, lab in list(col2label.items()):
        if lab in seen:
            col2label[c] = f"{lab} ({c})"
        seen[col2label[c]] = True
    label2col = {v: k for k, v in col2label.items()}
    return col2label, label2col

EDITABLE_FIELDS = [c for c in COLUMNS if c != "last_update_date"]
COL2LABEL, LABEL2COL = build_label_maps(EDITABLE_FIELDS)

# ──────────────────────────────────────────────────────────────────────────────
# Display ↔ DB value conversions (placeholders and coercions)
# ──────────────────────────────────────────────────────────────────────────────
def to_display_value(col: str, raw: Any) -> str:
    """What user sees in the grid."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)) or (isinstance(raw, str) and raw.strip() == ""):
        return "0" if is_numeric_col(col) else "-"
    # everything is string in the grid
    return str(raw)

def from_display_value(col: str, shown: str) -> Any:
    """Convert user-entered cell text to a DB-ready value for this column."""
    if is_numeric_col(col):
        # empty → 0 ; "-" → 0 ; else parse int
        s = (shown or "").strip()
        if s == "" or s == "-":
            return 0
        try:
            # contractor_count is bigint in your schema — cast to int
            return int(float(s))
        except Exception:
            # fallback: keep as 0 if unparsable
            return 0
    else:
        # text-like
        s = (shown or "").strip()
        if s == "-":
            return None   # explicit NULL
        return s

# ──────────────────────────────────────────────────────────────────────────────
# Load / cache the row for stable editing across reruns
# identify by email OR (client_name + account)
# ──────────────────────────────────────────────────────────────────────────────
if not (email or (client and account)):
    st.error("Missing identifier. Pass ?email=<...> or ?client=<...>&account=<...>")
    st.stop()

def _row_cache_key() -> str:
    if email:
        return f"edit_row::{email}"
    return f"edit_row::{client}::{account}"

@st.cache_data(ttl=120, show_spinner=False)
def _fetch_row(email: Optional[str], client: Optional[str], account: Optional[str]) -> Optional[pd.Series]:
    eng = get_engine()
    if email:
        sql = f"""
            SELECT {", ".join(COLUMNS)}
            FROM scout.centralize_db
            WHERE email_address = %(v)s
            LIMIT 1
        """
        params = {"v": email}
    else:
        sql = f"""
            SELECT {", ".join(COLUMNS)}
            FROM scout.centralize_db
            WHERE client_name = %(c)s AND account = %(a)s
            LIMIT 1
        """
        params = {"c": client, "a": account}
    with eng.begin() as con:
        df = pd.read_sql(sql, con, params=params)
    if df.empty:
        return None
    return df.iloc[0]

# Load session copy first (so saved edits persist visually)
ss_key = _row_cache_key()
if ss_key not in st.session_state:
    row = _fetch_row(email, client, account)
    if row is None:
        st.error("Record not found.")
        st.stop()
    st.session_state[ss_key] = row
row: pd.Series = st.session_state[ss_key]

title_name = (row.get("client_name") or "")
title_account = (row.get("account") or "")
st.markdown(f"### Edit: **{title_name}**  ·  _{title_account}_")

# ──────────────────────────────────────────────────────────────────────────────
# Build editable Field/Value table with placeholders applied
# ──────────────────────────────────────────────────────────────────────────────
def _build_edit_df(r: pd.Series) -> pd.DataFrame:
    data = []
    for col in EDITABLE_FIELDS:
        data.append({
            "Field": COL2LABEL[col],
            "Value": to_display_value(col, r.get(col, None)),
        })
    return pd.DataFrame(data)

orig_df = _build_edit_df(row)

# ──────────────────────────────────────────────────────────────────────────────
# GRID FIRST (so Save sees latest model)
# ──────────────────────────────────────────────────────────────────────────────
display_df = orig_df.copy()

gb = GridOptionsBuilder.from_dataframe(display_df)
gb.configure_default_column(
    wrapText=True, autoHeight=True, resizable=True, sortable=False, filter=False
)
gb.configure_column(
    "Field",
    editable=False,
    width=430,
    cellStyle={
        "fontWeight": "600",
        "backgroundColor": "#f7f9fc",
        "borderRight": "1px solid #cfe0ff",
    },
)
gb.configure_column("Value", editable=True)
gb.configure_grid_options(
    domLayout="autoHeight",   # "normal" | "autoHeight" | "print"
    suppressHorizontalScroll=True,
    stopEditingWhenCellsLoseFocus=True,

)
go = gb.build()

resp = AgGrid(
    display_df,
    gridOptions=go,
    theme="streamlit",
    fit_columns_on_grid_load=True,
    update_mode=GridUpdateMode.MODEL_CHANGED,   # <- always have latest in resp["data"]
    data_return_mode=DataReturnMode.AS_INPUT,
    allow_unsafe_jscode=False,
    key="edit_grid",
 
)
edited_df = pd.DataFrame(resp["data"])

# Build before/after maps keyed by DB column names (not labels)
before_map: Dict[str, Any] = {}
for _, r0 in orig_df.iterrows():
    col = LABEL2COL.get(r0["Field"], None)
    if col:
        before_map[col] = r0["Value"]

after_map: Dict[str, Any] = {}
for _, r1 in edited_df.iterrows():
    col = LABEL2COL.get(r1["Field"], None)
    if col:
        after_map[col] = r1["Value"]

# Optional: preview changes
# with st.expander("Preview pending changes", expanded=False):
#     pending = {k: v for k, v in after_map.items() if v != before_map.get(k, "")}
#     if pending:
#         st.json(pending)
#     else:
#         st.caption("No changes yet.")

# ──────────────────────────────────────────────────────────────────────────────
# Centered buttons with icons ABOVE each button label
# (st.button cannot render <img> inside label; this is the most reliable UX)
# ──────────────────────────────────────────────────────────────────────────────
left_sp, c1, c2, right_sp = st.columns([1, 1, 1, 1])

with c1:
    # st.markdown(f"<div style='text-align:center'><img src='{SAVE_ICON}' width='28'/></div>", unsafe_allow_html=True)
    btn_save = st.button("Save", type="primary", use_container_width=True)

with c2:
    # st.markdown(f"<div style='text-align:center'><img src='{DB_ICON}' width='28'/></div>", unsafe_allow_html=True)
    btn_refresh = st.button("Reload from DB", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────────────────────────
if btn_refresh:
    st.cache_data.clear()
    fresh = _fetch_row(email, client, account)
    if fresh is None:
        st.error("Record not found on reload.")
    else:
        st.session_state[ss_key] = fresh
        st.success("Reloaded from DB.")
    st.rerun()

# --- Save ---
# --- Save ---
if btn_save:
    # Detect changes (keys are COLUMN NAMES because we built maps by label->col)
    changed_display: Dict[str, Any] = {
        k: v for k, v in after_map.items()
        if v != before_map.get(k, "")
    }

    if not changed_display:
        st.info("No changes detected.")
        st.stop()

    # Convert display → DB types (handle '-' and numeric coercions)
    changed_db: Dict[str, Any] = {}
    for key, shown in changed_display.items():
        col = key  # keys here are already real column names
        changed_db[col] = from_display_value(col, shown)

    # Optional: validation/normalization (keep your existing checks)
    try:
        clean = normalize_payload(changed_db)
    except ValueError as ve:
        st.error(f"Validation error: {ve}")
        st.stop()

    # Direct DB update by identity
    if email:
        res = update_centralize_by_identity(clean, email=email)
    else:
        res = update_centralize_by_identity(clean, client=client, account=account)

    if res.get("ok"):
        # Update the session row so values reflect instantly
        updated = row.copy()
        for k, v in clean.items():
            updated[k] = v
        if res.get("updated_at"):
            updated["last_update_date"] = res["updated_at"]
        st.session_state[ss_key] = updated

        st.success("Saved to database ✅")
        st.rerun()
    else:
        st.error(f"Save failed: {res.get('error') or 'Unknown error'}")


