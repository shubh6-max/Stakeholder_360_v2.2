# components/aggrid_sections_all.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

# instant-persist via your service helpers
from features.stakeholders.service import (
    normalize_payload,
    diff_changes,
    build_update_key,
    update_centralize_record,
)

# --------------------------
# Section configs (label → column → type)
# Types: text | longtext | email | url | int | bool | date
# --------------------------

LEFT_SECTIONS: List[Tuple[str, str, List[Tuple[str, str, str]]]] = [
    (
        "🪪 Lead Identification & Contact Details",
        "lead_ident",
        [
            ("Business Group",        "working_group",      "text"),
            ("Lead Priority",         "lead_priority",      "text"),
            ("Client Name",           "client_name",        "text"),
            ("Designation",           "client_designation", "text"),
            ("Location (from teams)", "location",           "text"),
            ("Email address",         "email_address",      "email"),
            ("Designation Seniority", "seniority_level", "text"),
        ],
    ),
    
    
    (
        "🏢 Company & Department Info",
        "company",
        [
            ("Business Segment",   "account",       "text"),
            ("Working Group",      "working_group", "text"),
            ("Business Functions", "business_unit", "text"),  # per your note
            ("Service Line",       "service_line",  "text"),
        ],
    ),
    (
        "🧑‍🔧 Contractor Information",
        "contract",
        [
            ("Contractor count",    "contractor_count", "int"),
            ("Vendor Company Name", "vendor_name",      "text"),
        ],
    ),
    
]

RIGHT_SECTIONS: List[Tuple[str, str, List[Tuple[str, str, str]]]] = [
    (
        "🧭 Engagement & Outreach Strategy",
        "engage",
        [
            ("Scope of work/Priorities (internal research)", "internal_research", "longtext"),
            ("Additional Research (External)",               "external_research", "longtext"),
            ("MathCo LinkedIn Connects",                     "scout_linkedin_connected_flag", "bool"),
            ("Introduction Path",                            "introduction_path", "text"),
            ("Pursued in past",                              "pursued_in_past",   "bool"),
            ("If Yes, background/context ?",                 "context",           "longtext"),
            ("Comments",                                     "comments",          "longtext"),
        ],
    ),
    (
        "📈 Lead Status & Tracking",
        "status",
        [
            ("Who will reach out ?",  "csl_owner",         "text"),
            ("Lever for Reachout(s)", "reachout_lever",    "text"),
            ("Reachout Channel",      "reachout_channel",  "text"),
            ("Lead Status",           "status",            "text"),
            ("First Outreach Date",   "first_outreach_date","date"),
            ("Last Outreach Date",    "last_outreach_date", "date"),
        ],
    ),
    (
        "👤 Organizational Hierarchy",
        "org",
        [
            ("1st Degree Manager",        "reporting_manager",             "text"),
            ("1st Degree Manager Title",  "reporting_manager_designation", "text"),
        ],
    ),
    
    
    
]

# --------------------------
# Helpers (keep grid values as strings to avoid Arrow errors)
# --------------------------

def _placeholder(typ: str) -> str:
    return "0" if typ == "int" else "–"

def _to_display(v: Any, typ: str) -> str:
    if v is None or (isinstance(v, str) and not v.strip()):
        return _placeholder(typ)
    if typ == "bool":
        s = str(v).strip().lower()
        return "Yes" if s in ("1","true","yes","y","on") else "No"
    if typ == "date":
        from datetime import datetime, date
        try:
            if isinstance(v, (datetime, date)):
                d = v.date() if isinstance(v, datetime) else v
                return d.isoformat()
            return datetime.fromisoformat(str(v)).date().isoformat()
        except Exception:
            return str(v)
    return str(v)

def _from_display(shown: str, typ: str):
    ph = _placeholder(typ)
    if shown == ph:
        return None if typ == "int" else ""
    if typ == "bool":
        s = str(shown).strip().lower()
        if s in ("yes","true","1","y","on"):  return "true"
        if s in ("no","false","0","n","off"): return "false"
        return ""
    return shown

def _build_df_for_section(row: pd.Series | Dict[str, Any],
                          fields: List[Tuple[str, str, str]]) -> pd.DataFrame:
    get = row.get if isinstance(row, dict) else row.get
    data = []
    for label, col, typ in fields:
        raw = get(col, None)
        data.append({
            "Field": str(label),
            "Value": _to_display(raw, typ),
            "__field": col,
            "__type": typ,
            "__raw": raw,
        })
    df = pd.DataFrame(data)
    df["Field"] = df["Field"].astype(str)
    df["Value"] = df["Value"].astype(str)
    return df

def _render_grid(
    df_display: pd.DataFrame,
    key: str,
    *,
    label_width: int = 280,           # <== control label column width here
) -> pd.DataFrame:
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_default_column(
        wrapText=True, autoHeight=True, resizable=True, sortable=False, filter=False
    )

    # Field (label) column look: light background + blue right border
    gb.configure_column(
        "Field",
        editable=False,
        width=label_width,
        cellStyle={
            "fontWeight": "600",
            "backgroundColor": "#f7f9fc",
            "borderRight": "1px solid #cfe0ff",
        },
    )

    gb.configure_column("Value", editable=True)

    gb.configure_grid_options(
        
        domLayout="autoHeight",
        suppressHorizontalScroll=True,
        stopEditingWhenCellsLoseFocus=True,
        getRowStyle=JsCode("""
            function(params){
              return (params.rowIndex % 2 === 0)
                ? {'background':'#ffffff'} : {'background':'#fbfcff'};
            }
        """),
    )

    go = gb.build()
    resp = AgGrid(
        df_display,
        gridOptions=go,
        theme="streamlit",
        height=240,                       # min height so all cards feel consistent
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        data_return_mode=DataReturnMode.AS_INPUT,
        key=key,
        enable_enterprise_modules=False,
    )
    return resp["data"]

def _save_changes(before: pd.DataFrame, after: pd.DataFrame, original_row: pd.Series | Dict[str, Any], section_key: str):
    before_map = before.set_index("__field")["Value"].to_dict()
    after_map  = after.set_index("__field")["Value"].to_dict()
    types_map  = before.set_index("__field")["__type"].to_dict()

    changed: Dict[str, Any] = {}
    for f, new_shown in after_map.items():
        if new_shown != before_map.get(f):
            changed[f] = _from_display(new_shown, types_map.get(f, "text"))

    if not changed:
        return

    try:
        cleaned = normalize_payload(changed)
    except ValueError as ve:
        st.error(f"{section_key}: {ve}")
        return

    delta = diff_changes(original_row, cleaned)
    if not delta:
        return

    res = update_centralize_record(
        {k: v[1] for k, v in delta.items()},
        key=build_update_key(original_row),
        orig_last_update=original_row.get("last_update_date"),
        editor_identifier=(st.session_state.get("user") or {}).get("email", ""),
    )

    if res.get("ok"):
        st.toast("Saved", icon="✅")
        try:
            st.cache_data.clear()
        except Exception:
            pass
        if res.get("updated_at"):
            original_row["last_update_date"] = res["updated_at"]
        st.rerun()
    elif res.get("conflict"):
        st.warning("Record changed by someone else. Reloading…")
        st.rerun()
    else:
        st.error(f"Save failed: {res.get('error') or 'Unknown error'}")

def _render_section_card(
    title: str,
    sec_key: str,
    fields: List[Tuple[str, str, str]],
    row: pd.Series | Dict[str, Any],
    *,
    label_width: int = 280,
    card_gap: int = 16,                 # <== control vertical gap between cards
) -> None:
    # wrapper with controllable margin
    st.markdown(
        f'<div class="s360-card" style="margin-bottom:{card_gap}px;">',
        unsafe_allow_html=True,
    )

    # header
    st.markdown(
        f'<div class="s360-card-header">{title}</div>',
        unsafe_allow_html=True,
    )

    # grid
    before = _build_df_for_section(row, fields)
    display_df = before[["Field", "Value"]].copy()
    try:
        ag_data = _render_grid(display_df, key=f"ag_{sec_key}", label_width=label_width)
        after = before.copy()
        after["Value"] = ag_data["Value"].astype(str)
    except Exception as e:
        st.error(f"{sec_key} AgGrid error: {e}")
        st.table(display_df)
        st.markdown('</div>', unsafe_allow_html=True)  # close wrapper
        return

    # footer + close wrapper
    st.markdown('<div class="s360-card-footer"></div></div>', unsafe_allow_html=True)

    # save
    _save_changes(before, after, row, section_key=sec_key)

# --------------------------
# Public entry: render all sections in two columns
# --------------------------

def render_all_sections(
    row: pd.Series | Dict[str, Any],
    *,
    label_width: int = 280,     # <== public knob for label column width
    card_gap: int = 16,         # <== public knob for vertical spacing
) -> None:
    if row is None:
        st.info("Select a client to view details.")
        return

    left, right = st.columns(2)

    with left:
        for title, key, fields in LEFT_SECTIONS:
            _render_section_card(
                title, key, fields, row,
                label_width=label_width,
                card_gap=card_gap,
            )

    with right:
        for title, key, fields in RIGHT_SECTIONS:
            _render_section_card(
                title, key, fields, row,
                label_width=label_width,
                card_gap=card_gap,
            )
