# components/aggrid_sections_all.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

from features.stakeholders.service import (
    normalize_payload,
    diff_changes,
    build_update_key,
    update_centralize_record,
)

# -----------------------------------------------------------------------------
# Section configs (label → db_column → type)
# -----------------------------------------------------------------------------

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
            ("Designation Seniority", "seniority_level",    "text"),
        ],
    ),
    (
        "🏢 Company & Department Info",
        "company",
        [
            ("Business Segment",   "account",       "text"),
            ("Working Group",      "working_group", "text"),
            ("Business Functions", "business_unit", "text"),
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
            ("MathCo LinkedIn Connects", "scout_linkedin_connected_flag", "bool"),
            ("Introduction Path",        "introduction_path",             "text"),
            ("Pursued in past",          "pursued_in_past",               "bool"),
            ("If Yes, background/context ?", "context",                   "longtext"),
        ],
    ),
    (
        "📈 Lead Status & Tracking",
        "status",
        [
            ("Who will reach out ?",   "csl_owner",           "text"),
            ("Lever for Reachout(s)",  "reachout_lever",      "text"),
            ("Reachout Channel",       "reachout_channel",    "text"),
            ("Lead Status",            "status",              "text"),
            ("First Outreach Date",    "first_outreach_date", "date"),
            ("Last Outreach Date",     "last_outreach_date",  "date"),
        ],
    ),
    (
        "👤 Organizational Hierarchy",
        "org",
        [
            ("Reporting Manager",       "reporting_manager",             "text"),
            ("Reporting Manager Title", "reporting_manager_designation", "text"),
        ],
    ),
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _placeholder(typ: str) -> str:
    return "0" if typ == "int" else "–"

def _to_display(v: Any, typ: str) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and not v.strip()):
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
        return None if typ in ("int","date") else ""
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
    editable: bool,
    label_width: int = 240,
    row_height: int = 30,
) -> pd.DataFrame:
    gb = GridOptionsBuilder.from_dataframe(df_display)

    gb.configure_default_column(
        wrapText=True, autoHeight=True, resizable=True, sortable=False, filter=False
    )

    # Field column
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

    # Value column
    gb.configure_column("Value", editable=bool(editable))

    # Optional per-row dropdown
    if "__opts" in df_display.columns:
        gb.configure_column("__opts", hide=True, suppressToolPanel=True, editable=False)
        gb.configure_column(
            "Value",
            cellEditorSelector=JsCode("""
                function(params){
                  try{
                    const opts = params.data.__opts;
                    if (Array.isArray(opts) && opts.length){
                      return { component: 'agSelectCellEditor', params: { values: opts } };
                    }
                  }catch(e){}
                  return undefined;
                }
            """),
        )

    # ---- Key part for no extra beige and compact tables ----
    gb.configure_grid_options(
        domLayout="autoHeight",                # let grid height = rows height
        suppressHorizontalScroll=False,        # allow natural scroll when needed
        stopEditingWhenCellsLoseFocus=True,
        rowHeight=row_height,
        onGridReady=JsCode("""
          function(p){
            // fit columns to available width on first render
            p.api.sizeColumnsToFit();
          }
        """),
        onFirstDataRendered=JsCode("""
          function(p){
            p.api.sizeColumnsToFit();
          }
        """),
        getRowStyle=JsCode("""
          function(params){
            return (params.rowIndex % 2 === 0)
              ? {'background':'#ffffff','padding':'2px 0'}
              : {'background':'#fbfcff','padding':'2px 0'};
          }
        """),
    )

    go = gb.build()

    resp = AgGrid(
        df_display,
        gridOptions=go,
        theme="streamlit",
        fit_columns_on_grid_load=False,   # we now do it via onGridReady
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.NO_UPDATE if not editable else GridUpdateMode.VALUE_CHANGED,
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
    elif res.get("conflict"):
        st.warning("Record changed by someone else. Please refresh.")
    else:
        st.error(f"Save failed: {res.get('error') or 'Unknown error'}")

def _section_state_keys(sec_key: str) -> Dict[str, str]:
    return {
        "edit": f"s360_edit_{sec_key}",
        "dirty": f"s360_dirty_{sec_key}",
    }

def _render_section_card(
    title: str,
    sec_key: str,
    fields: List[Tuple[str, str, str]],
    row: pd.Series | Dict[str, Any],
    *,
    label_width: int = 240,
    card_gap: int = 4,
    max_card_width: int = 760,
    option_catalog: Optional[Dict[str, List[str]]] = None,
    edit_override: Optional[bool] = None,
    save_trigger: Optional[bool] = None,
    show_local_buttons: bool = True,
) -> None:

    keys = _section_state_keys(sec_key)
    ss = st.session_state
    ss.setdefault(keys["edit"], False)
    ss.setdefault(keys["dirty"], False)

    if edit_override is not None:
        ss[keys["edit"]] = bool(edit_override)

    is_edit = bool(ss[keys["edit"]])

    # WRAPPER (lightweight; no extra closers later)
    # st.markdown(
    #     f'''
    #     <div class="s360-card" style="margin-bottom:{card_gap}px; max-width:{max_card_width}px;">
    #     ''',
    #     unsafe_allow_html=True,
    # )

    # Header (compact)
    cols = st.columns([1, 0.001])  # kill right-side header space
    with cols[0]:
        st.markdown(
            '<div class="s360-card-header" style="padding:6px 10px;margin:2px 0 6px 0;"><b>'
            + title +
            '</b></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        if show_local_buttons:
            pass  # no per-card edit buttons (global toolbar handles it)

    # Build data
    before = _build_df_for_section(row, fields)
    display_df = before[["Field", "Value"]].copy()

    # Attach dropdown options
    if option_catalog:
        opts_per_row: List[List[str]] = []
        for _, meta in before[["__field"]].itertuples():
            opts_per_row.append(list(option_catalog.get(meta) or []))
        display_df["__opts"] = opts_per_row

    # Grid
    try:
        ag_data = _render_grid(
            display_df,
            key=f"ag_{sec_key}",
            editable=is_edit,
            label_width=label_width,
        )
        after = before.copy()
        after["Value"] = ag_data["Value"].astype(str)
    except Exception as e:
        st.error(f"{sec_key} AgGrid error: {e}")
        st.table(display_df)
        # st.markdown("</div>", unsafe_allow_html=True)  # close wrapper
        return

    # CLOSE the wrapper (just once)
    # st.markdown("</div>", unsafe_allow_html=True)

    # SAVE logic
    if save_trigger:   # global save-all
        _save_changes(before, after, row, section_key=sec_key)
        return

    if is_edit and st.session_state[keys["dirty"]]:
        _save_changes(before, after, row, section_key=sec_key)
        st.session_state[keys["edit"]] = False
        st.session_state[keys["dirty"]] = False
        st.rerun()


# -----------------------------------------------------------------------------
# Public entry point: with global toolbar
# -----------------------------------------------------------------------------

def render_all_sections(
    row: pd.Series | Dict[str, Any],
    *,
    label_width: int = 240,
    card_gap: int = 4,
    max_card_width: int = 760,
    option_catalog: Optional[Dict[str, List[str]]] = None,
) -> None:
    if row is None:
        st.info("Select a client to view details.")
        return

    # ---- Global toolbar ----
    st.markdown(
        '<div style="display:flex;gap:8px;align-items:center;margin:2px 0 8px 0;">'
        '<span style="font-weight:700;">Edit mode:</span>',
        unsafe_allow_html=True,
    )
    gk = "s360_global_edit"
    ss = st.session_state
    ss.setdefault(gk, False)

    col_a, col_b, col_c = st.columns([0.12, 0.12, 0.12])
    if not ss[gk]:
        with col_a:
            if st.button("Edit all", key="btn_global_edit", use_container_width=True):
                ss[gk] = True
                st.rerun()
    else:
        with col_a:
            save_all = st.button("Save all", key="btn_global_save", type="primary", use_container_width=True)
        with col_b:
            cancel_all = st.button("Cancel all", key="btn_global_cancel", use_container_width=True)
        if cancel_all:
            ss[gk] = False
            st.rerun()

    # st.markdown('</div>', unsafe_allow_html=True)

    global_edit = bool(ss[gk])

    left, right = st.columns(2)

    with left:
        for title, key, fields in LEFT_SECTIONS:
            _render_section_card(
                title, key, fields, row,
                label_width=label_width,
                card_gap=card_gap,
                max_card_width=max_card_width,
                option_catalog=option_catalog,
                edit_override=global_edit,
                save_trigger=st.session_state.get("btn_global_save", False),
                show_local_buttons=not global_edit,
            )

    with right:
        for title, key, fields in RIGHT_SECTIONS:
            _render_section_card(
                title, key, fields, row,
                label_width=label_width,
                card_gap=card_gap,
                max_card_width=max_card_width,
                option_catalog=option_catalog,
                edit_override=global_edit,
                save_trigger=st.session_state.get("btn_global_save", False),
                show_local_buttons=not global_edit,
            )

    if global_edit and st.session_state.get("btn_global_save", False):
        ss[gk] = False
        st.rerun()

