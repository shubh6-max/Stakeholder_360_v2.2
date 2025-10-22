# features/insights/persona_fn_kpi_runtime.py
from __future__ import annotations

import json
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import streamlit as st
from sqlalchemy import text

from components.persona_kpi_cards import render_persona_kpi_preview, render_impacts_block
from utils.db import get_engine
from utils.rag_env import get_chat_llm, get_embeddings
from features.insights.case_retriever import (
    resolve_persona,
    get_distinct_case_functions,
    retrieve_by_functions,
)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt & helpers
# ──────────────────────────────────────────────────────────────────────────────

# IMPORTANT: All literal braces are escaped with double braces `{{ ... }}` so
# Python .format() does not treat them as placeholders.
PROMPT_FN_KPI = """You are a B2B customer success analyst.

Analyze the stakeholder's role and generate the following:
- Business Functions
- Top 5–7 strategic KPIs for each
- Likely industry focus

Respond in STRICT JSON (no prose) with this shape:
{{
  "Business Function": {{
    "strategic_kpis": ["KPI1", "KPI2", "..."],
    "Industry": ["Sector1", "Sector2"]
  }}
}}

Context:
Account: {account}
Subsidiary: {subsidiary}
Working Group: {working_group}
Business Unit: {business_unit}
Service Line: {service_line}
Designation: {client_designation}
Seniority: {seniority_level}
Reporting Manager Title: {reporting_manager_designation}
Location: {location}

LinkedIn:
- Present Title: {client_present_title}
- Present Summary (html stripped): {client_present_description}

SharePoint Intel: {sharepoint_intel}

Internal Research: {internal_research}
External Research: {external_research}
Personalization Notes: {personalization_notes}
"""

_TAG_RE = re.compile(r"<[^>]+>")
_BULLET_RE = re.compile(r"^\s*(?:[-•\u2022]\s+|\d{1,2}[.)]\s+|\*)\s*(.+)$")

def _strip_html(s: Optional[str]) -> str:
    if not s:
        return ""
    clean = _TAG_RE.sub(" ", s)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```json\s*", "", s, flags=re.I | re.M)
    s = re.sub(r"^```\s*", "", s, flags=re.M)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _parse_llm_json(text: str) -> Dict[str, Any]:
    """
    Try hard to parse JSON from LLM output:
    - remove code fences
    - if whole string fails, find the first {...} block
    """
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        # greedy match first JSON object
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    raise ValueError("LLM did not return valid JSON.")

def _dedupe_bullets(lines: List[str], threshold: float = 0.88) -> List[str]:
    if not lines:
        return []
    emb = get_embeddings()
    kept: List[str] = []
    kept_vecs: List[np.ndarray] = []
    for line in lines:
        if not kept:
            kept.append(line)
            kept_vecs.append(np.array(emb.embed_query(line), dtype=np.float32))
            continue
        v = np.array(emb.embed_query(line), dtype=np.float32)
        sims = [float(np.dot(v / (np.linalg.norm(v)+1e-8), kv / (np.linalg.norm(kv)+1e-8))) for kv in kept_vecs]
        if max(sims) < threshold:
            kept.append(line)
            kept_vecs.append(v)
    return kept

def _extract_bullets(text: str, max_points: int = 5) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        m = _BULLET_RE.match(line)
        if m:
            val = m.group(1).strip()
            if val:
                out.append(val)
        if len(out) >= max_points * 2:
            break
    if not out:
        parts = [p.strip() for p in re.split(r"[•\u2022;]\s*", text) if p.strip()]
        out = parts[: max_points * 2]
    out = _dedupe_bullets(out, threshold=0.88)
    return out[:max_points]

def _flatten_schema(json_obj: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return [(function_label, payload_dict)]"""
    out: List[Tuple[str, Dict[str, Any]]] = []
    if not isinstance(json_obj, dict):
        return out
    for fn_label, payload in json_obj.items():
        if isinstance(payload, dict):
            out.append((str(fn_label).strip(), payload))
    return out

def _collect_from_preview(preview: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Extract function labels and KPI names from preview JSON."""
    fn_labels: List[str] = []
    kpi_names: List[str] = []
    for fn_label, payload in _flatten_schema(preview):
        fn_labels.append(fn_label)
        for k in (payload or {}).get("strategic_kpis") or []:
            kpi_names.append(str(k).strip())
    # de-dup while preserving order
    fn_labels = list(dict.fromkeys([x for x in fn_labels if x]))
    kpi_names = list(dict.fromkeys([x for x in kpi_names if x]))
    return fn_labels, kpi_names

# ──────────────────────────────────────────────────────────────────────────────
# LinkedIn signals
# ──────────────────────────────────────────────────────────────────────────────

SQL_LINKEDIN_BY_URL = """
SELECT client_present_title, client_present_description_html
FROM scout.linkedin_clients_data
WHERE client_url = :u
ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
LIMIT 1
"""
SQL_LINKEDIN_BY_EMAIL = """
SELECT client_present_title, client_present_description_html
FROM scout.linkedin_clients_data
WHERE lower(email_id) = lower(:e)
ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
LIMIT 1
"""
SQL_LINKEDIN_BY_NAME = """
SELECT client_present_title, client_present_description_html
FROM scout.linkedin_clients_data
WHERE lower(client_name) = lower(:n)
ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
LIMIT 1
"""

def _load_linkedin_signals(engine, *, linkedin_url: Optional[str], email: Optional[str], name: Optional[str]) -> Dict[str, str]:
    with engine.begin() as conn:
        if linkedin_url:
            li = conn.execute(text(SQL_LINKEDIN_BY_URL), {"u": linkedin_url}).mappings().first()
            if li:
                return {
                    "client_present_title": li["client_present_title"] or "",
                    "client_present_description": _strip_html(li["client_present_description_html"]),
                }
        if email:
            li = conn.execute(text(SQL_LINKEDIN_BY_EMAIL), {"e": email}).mappings().first()
            if li:
                return {
                    "client_present_title": li["client_present_title"] or "",
                    "client_present_description": _strip_html(li["client_present_description_html"]),
                }
        if name:
            li = conn.execute(text(SQL_LINKEDIN_BY_NAME), {"n": name}).mappings().first()
            if li:
                return {
                    "client_present_title": li["client_present_title"] or "",
                    "client_present_description": _strip_html(li["client_present_description_html"]),
                }
    return {"client_present_title": "", "client_present_description": ""}

# ──────────────────────────────────────────────────────────────────────────────
# Function mapping (persona → case functions)
# ──────────────────────────────────────────────────────────────────────────────

def _map_functions_to_case(function_labels: List[str], case_fn_universe: List[str]) -> Tuple[List[str], Dict[str, float]]:
    if not function_labels or not case_fn_universe:
        return [], {}

    from features.insights.case_retriever import embed_strings as _embed_strings
    p_vecs = _embed_strings(function_labels)
    c_vecs = _embed_strings(case_fn_universe)

    picks: List[Tuple[str, float]] = []
    for p_label, pv in zip(function_labels, p_vecs):
        pv = np.array(pv, dtype=np.float32)
        best = ("", -1.0)
        for c_label, cv in zip(case_fn_universe, c_vecs):
            cv = np.array(cv, dtype=np.float32)
            sim = float(np.dot(pv/(np.linalg.norm(pv)+1e-8), cv/(np.linalg.norm(cv)+1e-8)))
            if sim > best[1]:
                best = (c_label, sim)
        if best[0]:
            picks.append(best)

    picks.sort(key=lambda x: x[1], reverse=True)
    chosen: List[str] = []
    conf: Dict[str, float] = {}
    for lbl, s in picks:
        if lbl not in chosen:
            chosen.append(lbl)
            conf[lbl] = float(s)
        if len(chosen) >= 5:
            break
    return chosen, conf

# ──────────────────────────────────────────────────────────────────────────────
# DB writes (Save)
# ──────────────────────────────────────────────────────────────────────────────

SQL_WRITE_SHAREPOINT = """
UPDATE scout.centralize_db
SET internal_research = :ir
WHERE sr_no = :pid
"""

SQL_UPSERT_PERSONA_FUNCTION = """
INSERT INTO insights.persona_functions (persona_id, function_label, confidence)
VALUES (:pid, :fn, :conf)
ON CONFLICT (persona_id, function_label)
DO UPDATE SET confidence = EXCLUDED.confidence,
              updated_at = NOW()
"""

SQL_UPSERT_PERSONA_KPI = """
INSERT INTO insights.persona_kpis (persona_id, kpi_name, kpi_desc, weight, patterns, function_label, embedding)
VALUES (:pid, :name, :desc, :w, CAST(:patterns AS JSONB), :fn, :emb)
ON CONFLICT (persona_id, kpi_name)
DO UPDATE SET kpi_desc = EXCLUDED.kpi_desc,
              weight   = EXCLUDED.weight,
              patterns = EXCLUDED.patterns,
              function_label = EXCLUDED.function_label,
              embedding= EXCLUDED.embedding
"""

def _weight_from_text(name: str, desc: str) -> float:
    hi = ["margin", "profit", "revenue", "retention", "churn", "nps", "lead time", "on-time", "cost to serve", "ots", "otif", "inventory turns"]
    lo = ["tickets", "incidents", "bugs", "emails", "calls"]
    base = 1.0
    txt = f"{name} {desc}".lower()
    if any(k in txt for k in hi):
        base += 0.3
    if any(k in txt for k in lo):
        base -= 0.2
    return max(0.5, min(2.0, base))

# ──────────────────────────────────────────────────────────────────────────────
# Public UI block
# ──────────────────────────────────────────────────────────────────────────────

def render_persona_fn_kpi_block():
    # st.markdown("---")
    with st.container(border=True):
        # st.subheader("Persona Functions, KPIs & Impact Pointers")
        kpi_right_col,kpi_left_col= st.columns([2,6])
        with kpi_right_col:
            st.markdown(
        f"""
        <div style="
          display:flex;justify-content:space-between;align-items:center;
          background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
          padding:10px 12px;margin-bottom:10px;">
          <div style="display:flex;align-items:center;gap:10px;">
            <img src="https://img.icons8.com/?size=100&id=nNRpaPWMdCDG&format=png&color=000000" width="22" height="22" alt="info">
            <div style="font-size:16px;font-weight:700;">Persona KPIs & Impact Pointers</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

        sel = (st.session_state.get("s360") or {}).get("selected_row") or st.session_state.get("s360.selected_row") or {}
        if not sel:
            st.info("Select a stakeholder first.")
            return

        engine = get_engine()

        # Resolve persona: email first, else client_name only
        persona = resolve_persona(
            engine,
            email=(sel.get("email_address") or "").strip() or None,
            client_name=(sel.get("client_name") or "").strip() or None,
        )
        if not persona:
            st.error("Could not resolve persona in scout.centralize_db using email or client_name.")
            return

        persona_id = int(persona["persona_id"])
        linkedin_url = persona.get("linkedin_url") or sel.get("linkedin_url")
        li = _load_linkedin_signals(
            engine,
            linkedin_url=linkedin_url or None,
            email=(persona.get("email_address") or "").strip() or None,
            name=(persona.get("persona_name") or "").strip() or None,
        )

        # SharePoint intel input
        st.session_state.setdefault("sharepoint_intel", "")
        sp_intel = st.text_area(
            "SharePoint intel:",
            value=st.session_state.get("sharepoint_intel", ""),
            height=120,
        )
        st.session_state["sharepoint_intel"] = sp_intel

        # Buttons
        c1, c2, c3 = st.columns([1, 1, 1])
        generated_now = False

        # Generate
        if c1.button("Generate"):
            llm = get_chat_llm(temperature=0.1)
            msg = PROMPT_FN_KPI.format(
                account=persona.get("account",""),
                subsidiary=sel.get("subsidiary",""),
                working_group=persona.get("working_group",""),
                business_unit=persona.get("business_unit",""),
                service_line=persona.get("service_line",""),
                client_designation=persona.get("client_designation",""),
                seniority_level=persona.get("seniority_level",""),
                reporting_manager_designation=persona.get("reporting_manager_designation",""),
                location=persona.get("location",""),
                client_present_title=li.get("client_present_title",""),
                client_present_description=li.get("client_present_description",""),
                sharepoint_intel=sp_intel or "",
                internal_research=persona.get("internal_research","") or "",
                external_research=persona.get("external_research","") or "",
                personalization_notes=persona.get("personalization_notes","") or "",
            )
            raw = llm.invoke(msg).content
            try:
                data = _parse_llm_json(raw)
            except Exception:
                # nudge the model once to return JSON-only
                raw2 = llm.invoke("Return JSON ONLY. No prose.\n\n" + msg).content
                data = _parse_llm_json(raw2)

            st.session_state["persona_fn_kpis_preview"] = data
            st.success("Generated persona functions & KPIs.")
            # Show cards: 3 across, KPIs in 2 columns within each card
            render_persona_kpi_preview(data, card_cols=3, kpi_cols=2)
            generated_now = True

        # Save
        if c2.button("Save"):
            data = st.session_state.get("persona_fn_kpis_preview")
            if not data:
                st.warning("Generate first, then Save.")
            else:
                # 1) SharePoint intel write-back
                with engine.begin() as conn:
                    conn.execute(text(SQL_WRITE_SHAREPOINT), {"ir": sp_intel or "", "pid": persona_id})

                # 2) Upsert persona functions (confidence via mapping to case functions)
                case_fn_universe = get_distinct_case_functions(engine)
                preview_fn_labels, _ = _collect_from_preview(data)
                mapped, conf = _map_functions_to_case(preview_fn_labels, case_fn_universe)
                with engine.begin() as conn:
                    for lbl in preview_fn_labels:
                        # Use mapped confidence if label appears; default 0.75 otherwise
                        best = max(0.75, float(conf.get(lbl, 0.0)))
                        conn.execute(text(SQL_UPSERT_PERSONA_FUNCTION),
                                     {"pid": persona_id, "fn": lbl, "conf": best})

                # 3) Upsert KPIs per function
                emb = get_embeddings()
                with engine.begin() as conn:
                    for fn_label, payload in _flatten_schema(data):
                        for name in (payload or {}).get("strategic_kpis") or []:
                            name = str(name).strip()
                            if not name:
                                continue
                            desc = ""
                            # light heuristic weight
                            hi = ["margin","profit","revenue","retention","churn","nps","lead time","on-time","cost to serve","ots","otif","inventory turns"]
                            lo = ["tickets","incidents","bugs","emails","calls"]
                            weight = 1.0 + (0.3 if any(k in (name.lower()) for k in hi) else 0.0) - (0.2 if any(k in (name.lower()) for k in lo) else 0.0)
                            weight = float(max(0.5, min(2.0, weight)))
                            patterns: List[str] = []
                            v = emb.embed_query(name)
                            conn.execute(text(SQL_UPSERT_PERSONA_KPI), {
                                "pid": persona_id,
                                "name": name,
                                "desc": desc,
                                "w": weight,
                                "patterns": json.dumps(patterns, ensure_ascii=False),
                                "fn": fn_label,
                                "emb": v,
                            })
                st.success("Saved SharePoint intel, persona functions, and KPIs.")

        # Fetch Impact Pointers
        fetch_clicked = c3.button("Fetch Impact Pointers")

        # If preview exists (even before fetch), render KPI cards so user sees output
        preview = st.session_state.get("persona_fn_kpis_preview", {})
        if preview and not generated_now:
            render_persona_kpi_preview(preview, card_cols=3, kpi_cols=2)

        # Auto-fetch impacts after Generate OR when Fetch clicked
        if generated_now or fetch_clicked:
            # Collect function labels & KPIs from preview; if none, fall back to DB
            fn_labels, kpi_names = _collect_from_preview(preview)
            if not fn_labels or not kpi_names:
                with engine.begin() as conn:
                    rows = conn.execute(
                        text("SELECT kpi_name, function_label FROM insights.persona_kpis WHERE persona_id = :pid"),
                        {"pid": persona_id},
                    ).mappings().all()
                for r in rows:
                    name = (r.get("kpi_name") or "").strip()
                    if name:
                        kpi_names.append(name)
                    fn = (r.get("function_label") or "").strip()
                    if fn:
                        fn_labels.append(fn)
                # de-dup
                fn_labels = list(dict.fromkeys(fn_labels))
                kpi_names = list(dict.fromkeys(kpi_names))

            # Map persona → case functions (embedding-based)
            case_fn_universe = get_distinct_case_functions(engine)
            mapped_labels, _ = _map_functions_to_case(fn_labels, case_fn_universe)

            # Build retrieval query
            parts = [
                st.session_state.get("sharepoint_intel") or "",
                li.get("client_present_title",""),
                li.get("client_present_description",""),
                " ".join(kpi_names[:8]) if kpi_names else "",
                persona.get("internal_research") or "",
                persona.get("external_research") or "",
                persona.get("personalization_notes") or "",
            ]
            qtext = " ".join([p for p in parts if p]).strip()[:2000]
            if not qtext:
                st.info("No query context available. Please add SharePoint intel or generate KPIs first.")
                return

            rows = retrieve_by_functions(
                engine,
                query=qtext,
                function_labels=mapped_labels,
                sections=["impact", "kpi"],
                strict_first_k=24,
                fallback_k=24,
            )

            # Aggregate top-5 impact pointers
            impacts: List[str] = []
            best_src: Optional[Dict[str, Any]] = None
            for r in rows:
                pts = _extract_bullets(r["text"], max_points=5)
                for p in pts:
                    if p not in impacts:
                        impacts.append(p)
                if impacts and not best_src:
                    best_src = r
                if len(impacts) >= 5:
                    break
            impacts = impacts[:5]

            # Render: KPI cards + impacts in columns with source chip
            if impacts:
                render_impacts_block(impacts, best_src, impact_cols=2)
            else:
                # No impacts → show only KPI cards (already rendered above)
                st.info("No impact pointers found. Showing suggested KPIs above.")
