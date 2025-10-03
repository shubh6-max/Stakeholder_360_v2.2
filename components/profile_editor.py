# components/profile_editor.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st

from features.stakeholders.service import (
    get_distincts,
    normalize_payload,
    diff_changes,
    build_update_key,
    update_centralize_record,
)

# ---------- small helpers ----------

def _get_editor_id() -> str:
    # Best-effort user identifier for audit trail
    u = st.session_state.get("user") or {}
    return (u.get("email") or u.get("username") or u.get("first_name") or "").strip()

def _select_or_add(label: str, options: List[str], value: str | None, key: str) -> str | None:
    """
    Allows user to choose an existing option or select 'Add new…' to type a custom value.
    Returns None or the chosen/typed string (trimmed; empty -> None).
    """
    add_token = "➕ Add new…"
    opts = [""] + options + [add_token]
    idx_default = 0
    if value:
        try:
            idx_default = opts.index(value)
        except ValueError:
            # value not in list, keep as custom
            idx_default = 0

    chosen = st.selectbox(label, opts, index=idx_default, key=f"{key}_sel")
    if chosen == add_token:
        new_val = st.text_input(f"{label} (new)", key=f"{key}_new")
        return new_val.strip() or None
    if chosen == "":
        return None
    return chosen.strip() or None

def _bool_input(label: str, value: Any, key: str) -> Optional[bool]:
    # Convert various truthy strings to bool; default False for UI, but keep None if unchecked and value is None
    default = bool(value) if value is not None else False
    b = st.checkbox(label, value=default, key=key)
    # If original was None and user left unchecked, keep None; otherwise b
    return b if value is not None or b else None

def _date_input(label: str, value: Any, key: str):
    # value may be None, date, or datetime
    import datetime as _dt
    default = None
    if isinstance(value, _dt.date):
        default = value
    elif isinstance(value, _dt.datetime):
        default = value.date()
    return st.date_input(label, value=default, format="YYYY-MM-DD", key=key)

def _text_area(label: str, value: str | None, key: str, height: int = 94, max_chars: int | None = None):
    return st.text_area(label, value=value or "", key=key, height=height, max_chars=max_chars)

def _text_input(label: str, value: str | None, key: str, placeholder: str = ""):
    return st.text_input(label, value=value or "", key=key, placeholder=placeholder)

def _number_input(label: str, value: Any, key: str, min_value: int = 0, step: int = 1):
    try:
        v = int(value) if value is not None else 0
    except Exception:
        v = 0
    return st.number_input(label, value=v, min_value=min_value, step=step, key=key)

# ---------- public renderer ----------

def render_profile_editor(row) -> None:
    """
    Full multi-section editor with optimistic-lock save.
    Everyone can edit (no role checks).
    """
    if row is None:
        return

    # We need last_update_date for optimistic lock; if not present, lock will allow NULL-on-first-save
    orig_last_update = row.get("last_update_date", None)
    # UI header
    st.markdown("### Edit Profile")

    # Distinct options for selectboxes (cached by service)
    distincts = get_distincts()
    accounts = distincts.get("account", [])
    wgroups  = distincts.get("working_group", [])
    bus      = distincts.get("business_unit", [])
    slines   = distincts.get("service_line", [])
    mgrs     = distincts.get("reporting_manager", [])

    # Toggle / state
    st.session_state.setdefault("editing_profile", False)
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        if not st.session_state["editing_profile"]:
            if st.button("✏️ Edit"):
                st.session_state["editing_profile"] = True
                st.rerun()
        else:
            if st.button("✖️ Cancel"):
                st.session_state["editing_profile"] = False
                st.rerun()

    with col_b:
        if st.session_state["editing_profile"]:
            st.caption(_last_update_text(orig_last_update))

    if not st.session_state["editing_profile"]:
        st.info("Click **Edit** to update fields.")
        return

    # ---- EDIT FORM ----
    with st.form("profile_edit_form", clear_on_submit=False):
        # --- Lead Identification & Contact Details ---
        with st.expander("🪪 Lead Identification & Contact Details", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                lead_priority = _text_input("Lead Priority", row.get("lead_priority"), "lead_priority")
                client_name   = _text_input("Client Name", row.get("client_name"), "client_name")
                designation   = _text_input("Designation", row.get("client_designation"), "client_designation")
            with c2:
                location     = _text_input("Location (from teams)", row.get("location"), "location")
                email        = _text_input("Email address", row.get("email_address"), "email_address", placeholder="name@company.com")
                linkedin_url = _text_input("LinkedIn URL", row.get("linkedin_url"), "linkedin_url", placeholder="https://...")

        # --- Company & Department Info ---
        with st.expander("🏢 Company & Department Info", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                account      = _select_or_add("Business Segment (Account)", accounts, row.get("account"), "account")
                working_grp  = _select_or_add("Working Group", wgroups, row.get("working_group"), "working_group")
            with c2:
                business_fn  = _select_or_add("Business Functions (Business Unit)", bus, row.get("business_unit"), "business_unit")
                service_line = _select_or_add("Service Line", slines, row.get("service_line"), "service_line")

        # --- Organizational Hierarchy ---
        with st.expander("👥 Organizational Hierarchy", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                rep_mgr = _select_or_add("1st Degree Manager", mgrs, row.get("reporting_manager"), "reporting_manager")
            with c2:
                rep_mgr_title = _text_input("1st Degree Manager Title", row.get("reporting_manager_designation"),
                                            "reporting_manager_designation")

        # --- Lead Status & Tracking ---
        with st.expander("📈 Lead Status & Tracking", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                csl_owner   = _text_input("Who will reach out? (Owner)", row.get("csl_owner"), "csl_owner")
                lever       = _text_input("Lever for Reach out(s)", row.get("reachout_lever"), "reachout_lever")
            with c2:
                channel     = _text_input("Reachout Channel", row.get("reachout_channel"), "reachout_channel")
                status      = _text_input("Lead Status", row.get("status"), "status")

            d1, d2 = st.columns(2)
            with d1:
                first_outreach = _date_input("First Outreach Date", row.get("first_outreach_date"), "first_outreach_date")
            with d2:
                last_outreach  = _date_input("Last Outreach Date", row.get("last_outreach_date"), "last_outreach_date")

        # --- Engagement & Outreach Strategy ---
        with st.expander("🧭 Engagement & Outreach Strategy", expanded=False):
            internal_research = _text_area("Scope / Priorities (Internal research)", row.get("internal_research"),
                                           "internal_research", height=100)
            external_research = _text_area("Additional Research (External)", row.get("external_research"),
                                           "external_research", height=100)
            e1, e2, e3 = st.columns(3)
            with e1:
                intro_path  = _text_input("Introduction Path", row.get("introduction_path"), "introduction_path")
            with e2:
                pursued     = _bool_input("Pursued in past", row.get("pursued_in_past"), "pursued_in_past")
            with e3:
                linked_flag = _bool_input("MathCo LinkedIn Connects", row.get("scout_linkedin_connected_flag"),
                                          "scout_linkedin_connected_flag")

            s1, s2, s3 = st.columns(3)
            with s1:
                spoc1 = _text_input("MathCo SPOC 1", row.get("mathco_spoc_1"), "mathco_spoc_1")
            with s2:
                spoc2 = _text_input("MathCo SPOC 2", row.get("mathco_spoc_2"), "mathco_spoc_2")
            with s3:
                spoc3 = _text_input("MathCo SPOC 3", row.get("mathco_spoc_3"), "mathco_spoc_3")

            context      = _text_area("If yes, background/context?", row.get("context"), "context", height=80)
            comments     = _text_area("Comments", row.get("comments"), "comments", height=80)

        # --- Expertise & Experience ---
        with st.expander("💡 Expertise & Experience", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                seniority = _text_input("Designation Seniority", row.get("seniority_level"), "seniority_level")
                kpi       = _text_input("KPI", row.get("kpi"), "kpi")
            with c2:
                intel_sum = _text_area("Intel Summary", row.get("intel_summary"), "intel_summary", height=80)
            email_tpl = _text_area("Email Template", row.get("email_template"), "email_template", height=80)
            impact    = _text_area("Impact Pointers", row.get("impact_pointers"), "impact_pointers", height=80)

        # --- Contractor Information ---
        with st.expander("🧑‍🔧 Contractor Information", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                contractor_cnt = _number_input("Contractor count", row.get("contractor_count"), "contractor_count")
            with c2:
                vendor_name = _text_input("Vendor Company Name", row.get("vendor_name"), "vendor_name")

        # ---- submit row ----
        col_save = st.columns([1, 3])[0]
        with col_save:
            save_clicked = st.form_submit_button("💾 Save changes")

    # Handle save outside the form context
    if save_clicked:
        raw = {
            # Lead Identification & Contact
            "lead_priority": lead_priority,
            "client_name": client_name,
            "client_designation": designation,
            "location": location,
            "email_address": email,
            "linkedin_url": linkedin_url,

            # Company & Department Info
            "account": account,
            "working_group": working_grp,
            "business_unit": business_fn,
            "service_line": service_line,

            # Org Hierarchy
            "reporting_manager": rep_mgr,
            "reporting_manager_designation": rep_mgr_title,

            # Lead Status & Tracking
            "csl_owner": csl_owner,
            "reachout_lever": lever,
            "reachout_channel": channel,
            "status": status,
            "first_outreach_date": first_outreach,
            "last_outreach_date": last_outreach,

            # Engagement & Outreach
            "internal_research": internal_research,
            "external_research": external_research,
            "introduction_path": intro_path,
            "pursued_in_past": pursued,
            "mathco_spoc_1": spoc1,
            "mathco_spoc_2": spoc2,
            "mathco_spoc_3": spoc3,
            "scout_linkedin_connected_flag": linked_flag,
            "context": context,
            "comments": comments,

            # Expertise & Experience
            "seniority_level": seniority,
            "kpi": kpi,
            "intel_summary": intel_sum,
            "email_template": email_tpl,
            "impact_pointers": impact,

            # Contractor
            "contractor_count": contractor_cnt,
            "vendor_name": vendor_name,
        }

        try:
            cleaned = normalize_payload(raw)
        except ValueError as ve:
            st.error(str(ve))
            return

        delta = diff_changes(row, cleaned)
        if not delta:
            st.info("No changes detected.")
            return

        key = build_update_key(row)
        res = update_centralize_record(
            {k: v[1] for k, v in delta.items()},
            key=key,
            orig_last_update=orig_last_update,
            editor_identifier=_get_editor_id(),
        )

        if res.get("ok"):
            st.success("Saved successfully.")
            # refresh caches + exit edit mode
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.session_state["editing_profile"] = False
            st.rerun()
        elif res.get("conflict"):
            st.warning("This record was modified by someone else. Please reload and try again.")
        else:
            st.error(f"Save failed: {res.get('error') or 'Unknown error'}")


def _last_update_text(ts) -> str:
    if not ts:
        return "_No previous update_"
    if isinstance(ts, datetime):
        return f"_Last updated: {ts.strftime('%Y-%m-%d %H:%M:%S')}_"
    try:
        # ts may be string
        dt = datetime.fromisoformat(str(ts))
        return f"_Last updated: {dt.strftime('%Y-%m-%d %H:%M:%S')}_"
    except Exception:
        return f"_Last updated: {ts}_"
