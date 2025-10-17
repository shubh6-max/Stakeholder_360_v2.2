# scripts/daily_kpi_cache.py
from __future__ import annotations

# =========================
# Headless Streamlit shim
# =========================
import sys, types, logging
def _install_streamlit_dummy_if_headless():
    try:
        import streamlit as _st  # real package is present
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                return  # real Streamlit runtime: keep it
        except Exception:
            pass
    except Exception:
        pass

    dummy = types.ModuleType("streamlit")
    class _DummyST:
        def spinner(self, *_a, **_k):
            class _CM:
                def __enter__(self_s): return None
                def __exit__(self_s, *exc): return False
            return _CM()
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def write(self, *a, **k): pass
        def info(self, *a, **k): pass
        def success(self, *a, **k): pass

    dummy.st = _DummyST()
    dummy.spinner = dummy.st.spinner
    dummy.warning = dummy.st.warning
    dummy.error   = dummy.st.error
    dummy.write   = dummy.st.write
    dummy.info    = dummy.st.info
    dummy.success = dummy.st.success
    sys.modules["streamlit"] = dummy

_install_streamlit_dummy_if_headless()
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# =========================
# Imports
# =========================
import os, time, json, random, threading
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from collections import deque

from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine

from utils.db import get_engine
from utils.rag_db import ensure_rag_schema
from features.insights.store import ensure_kpi_cache, upsert_kpis, _persona_key
from features.insights.retrieve import run_kpis_for_persona

# =========================
# Tunables (override via env)
# =========================
BATCH_SIZE        = int(os.getenv("KPI_BATCH_SIZE", "250"))
CACHE_TTL_D       = int(os.getenv("KPI_CACHE_TTL_DAYS", "14"))
MAX_WORKERS       = int(os.getenv("KPI_MAX_WORKERS", "1"))     # conservative for S0
RETRIES           = int(os.getenv("KPI_RETRIES", "2"))
BASE_BACKOFF      = float(os.getenv("KPI_BACKOFF_SECS", "1.2"))

# Rate limiting knobs (for Azure OpenAI)
EMBED_RPM         = int(os.getenv("KPI_EMBED_RPM", "40"))      # requests/min across the process
THROTTLE_SECS     = float(os.getenv("KPI_THROTTLE_SECS", "0")) # optional fixed delay per persona

# =========================
# Lightweight rate limiter
# =========================
class RateLimiter:
    """Sliding-window limiter: allow at most max_calls within window_secs."""
    def __init__(self, max_calls:int, window_secs:float=60.0):
        self.max_calls = max_calls
        self.window = window_secs
        self.events = deque()
        self.lock = threading.Lock()
    def acquire(self):
        if self.max_calls <= 0:
            return
        with self.lock:
            now = time.time()
            while self.events and now - self.events[0] > self.window:
                self.events.popleft()
            if len(self.events) >= self.max_calls:
                sleep_for = self.window - (now - self.events[0]) + 0.01
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self.events.append(time.time())

_embed_rl = RateLimiter(EMBED_RPM, 60.0)

def _is_azure_429(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg and ("rate limit" in msg or "exceeded" in msg)

def _retry_after_seconds_from(err: Exception) -> Optional[float]:
    # Try best-effort: SDKs sometimes carry response/header; else parse message
    retry_after = None
    try:
        ra = getattr(getattr(err, "response", None), "headers", {}).get("Retry-After")
        if ra:
            retry_after = float(ra)
    except Exception:
        pass
    if retry_after is not None:
        return retry_after
    # crude parse in the message: "... retry after 60 seconds ..."
    txt = str(err).lower().replace("-", " ").replace("_", " ")
    toks = txt.split()
    for i, tok in enumerate(toks):
        if tok.isdigit():
            sec = int(tok)
            if 1 <= sec <= 86400:
                # check preceding word "after"
                if i >= 1 and toks[i-1] in {"after", "in"}:
                    return float(sec)
    return None

# =========================
# SQL: select candidates
# =========================
SELECT_CANDIDATES_SQL = """
WITH src AS (
  SELECT
    COALESCE(NULLIF(TRIM(account), ''), '')                         AS company_name,
    COALESCE(NULLIF(TRIM(subsidiary), ''), '')                      AS subsidiary,
    COALESCE(NULLIF(TRIM(working_group), ''), '')                   AS working_group,
    COALESCE(NULLIF(TRIM(business_unit), ''), '')                   AS business_unit,
    COALESCE(NULLIF(TRIM(service_line), ''), '')                    AS service_line,
    COALESCE(NULLIF(TRIM(client_name), ''), '')                     AS client_name,
    COALESCE(NULLIF(TRIM(client_designation), ''), '')              AS client_designation,
    COALESCE(NULLIF(TRIM(reporting_manager_designation), ''), '')   AS reporting_manager_designation,
    COALESCE(NULLIF(TRIM(email_address), ''), '')                   AS email_address,
    COALESCE(NULLIF(TRIM(location), ''), '')                        AS location,
    COALESCE(NULLIF(TRIM(lead_priority), ''), 'Z')                  AS lead_priority,

    -- Normalize last_update_date to timestamptz (no AT TIME ZONE on text)
    CASE
      WHEN last_update_date IS NULL THEN NULL
      WHEN pg_typeof(last_update_date)::text = 'timestamp with time zone'
        THEN last_update_date::timestamptz
      WHEN pg_typeof(last_update_date)::text = 'timestamp without time zone'
        THEN (last_update_date::timestamp)::timestamptz
      WHEN pg_typeof(last_update_date)::text = 'date'
        THEN (last_update_date::timestamp)::timestamptz
      WHEN pg_typeof(last_update_date)::text = 'text'
           AND last_update_date ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}([ T][0-9]{2}:[0-9]{2}(:[0-9]{2})?)?$'
        THEN (last_update_date::timestamp)::timestamptz
      WHEN pg_typeof(last_update_date)::text = 'text'
           AND last_update_date ~ '^[0-9]{2}/[0-9]{2}/[0-9]{4}$'
        THEN to_timestamp(last_update_date, 'MM/DD/YYYY')::timestamptz
      WHEN pg_typeof(last_update_date)::text = 'text'
           AND last_update_date ~ '^[0-9]{2}-[0-9]{2}-[0-9]{4}$'
        THEN to_timestamp(last_update_date, 'MM-DD-YYYY')::timestamptz
      ELSE NULL
    END AS last_update_at
  FROM scout.centralize_db
  WHERE COALESCE(NULLIF(TRIM(client_name), ''), '') <> ''
    AND COALESCE(NULLIF(TRIM(account), ''), '') <> ''
)
SELECT *
FROM src
ORDER BY
  lead_priority ASC,
  COALESCE(last_update_at, NOW() - INTERVAL '10 years') DESC
LIMIT :n;
"""

def select_candidates(engine: Engine, n: int) -> List[Dict[str, Any]]:
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(SELECT_CANDIDATES_SQL), {"n": int(n)}).mappings().all()
            return [dict(r) for r in rows]
    except Exception:
        pass

    # Fallback: simpler ordering
    fallback_sql = """
      WITH src AS (
        SELECT
          COALESCE(NULLIF(TRIM(account), ''), '')                         AS company_name,
          COALESCE(NULLIF(TRIM(subsidiary), ''), '')                      AS subsidiary,
          COALESCE(NULLIF(TRIM(working_group), ''), '')                   AS working_group,
          COALESCE(NULLIF(TRIM(business_unit), ''), '')                   AS business_unit,
          COALESCE(NULLIF(TRIM(service_line), ''), '')                    AS service_line,
          COALESCE(NULLIF(TRIM(client_name), ''), '')                     AS client_name,
          COALESCE(NULLIF(TRIM(client_designation), ''), '')              AS client_designation,
          COALESCE(NULLIF(TRIM(reporting_manager_designation), ''), '')   AS reporting_manager_designation,
          COALESCE(NULLIF(TRIM(email_address), ''), '')                   AS email_address,
          COALESCE(NULLIF(TRIM(location), ''), '')                        AS location,
          COALESCE(NULLIF(TRIM(lead_priority), ''), 'Z')                  AS lead_priority
        FROM scout.centralize_db
        WHERE COALESCE(NULLIF(TRIM(client_name), ''), '') <> ''
          AND COALESCE(NULLIF(TRIM(account), ''), '') <> ''
      )
      SELECT *
      FROM src
      ORDER BY lead_priority ASC
      LIMIT :n;
    """
    with engine.begin() as conn2:
        rows = conn2.execute(text(fallback_sql), {"n": int(n)}).mappings().all()
        return [dict(r) for r in rows]

# =========================
# Persona helpers
# =========================
def build_persona_blob(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": row.get("client_designation", ""),
        "working_group": row.get("working_group", ""),
        "business_unit": row.get("business_unit", ""),
        "service_line": row.get("service_line", ""),
        "manager_title": row.get("reporting_manager_designation", ""),
        "subsidiary": row.get("subsidiary", ""),
        "region_or_location": row.get("location", ""),
        "email": row.get("email_address", ""),
    }

def persona_row_for_rag(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "client_name": row.get("client_name", ""),
        "client_designation": row.get("client_designation", ""),
        "working_group": row.get("working_group", ""),
        "business_unit": row.get("business_unit", ""),
        "service_line": row.get("service_line", ""),
        "subsidiary": row.get("subsidiary", ""),
        "location": row.get("location", ""),
        "email_address": row.get("email_address", ""),
        "reporting_manager_designation": row.get("reporting_manager_designation", ""),
    }

# =========================
# Filter already cached within TTL
# =========================
def filter_not_cached(engine: Engine, candidates: List[Dict[str, Any]], ttl_days: int) -> List[Dict[str, Any]]:
    enriched = []
    for r in candidates:
        company = (r.get("company_name") or "").strip()
        if not company:
            continue
        pblob = build_persona_blob(r)
        pkey = _persona_key(company, pblob)
        enriched.append((r, company, pkey))

    by_co: Dict[str, List[str]] = {}
    for _r, co, pk in enriched:
        by_co.setdefault(co, []).append(pk)

    cached: Dict[Tuple[str, str], bool] = {}
    with engine.begin() as conn:
        for co, keys in by_co.items():
            if not keys:
                continue
            unique_keys = list(set(keys))
            sql = text("""
              SELECT persona_key
              FROM rag.persona_kpi_cache
              WHERE company_name = :c
                AND persona_key IN :keys
                AND created_at >= NOW() - (:ttl * INTERVAL '1 day')
            """).bindparams(bindparam("keys", expanding=True))
            rows = conn.execute(sql, {"c": co, "keys": unique_keys, "ttl": int(ttl_days)}).fetchall()
            hit = {row[0] for row in rows}
            for k in unique_keys:
                cached[(co, k)] = (k in hit)

    out: List[Dict[str, Any]] = []
    for r, co, pk in enriched:
        if not cached.get((co, pk), False):
            r["_persona_key"] = pk
            out.append(r)
    return out

# =========================
# Worker with 429 handling
# =========================
def process_one(engine: Engine, row: Dict[str, Any]) -> Tuple[str, str, str, bool, Optional[str]]:
    company = (row.get("company_name") or "").strip()
    persona_name = (row.get("client_name") or "").strip()
    if not company:
        return company, persona_name, "", False, "company is required"

    persona_blob = build_persona_blob(row)
    pk = row.get("_persona_key") or _persona_key(company, persona_blob)
    persona_rag = persona_row_for_rag(row)

    if THROTTLE_SECS > 0:
        time.sleep(THROTTLE_SECS)

    backoff = BASE_BACKOFF
    for attempt in range(1, RETRIES + 1):
        try:
            # Acquire embedding quota token BEFORE embedding-heavy path
            _embed_rl.acquire()

            ui_payload = run_kpis_for_persona(company, persona_rag, top_k=8)

            upsert_kpis(
                engine,
                company=company,
                persona_key=pk,
                persona_blob=persona_blob,
                kpis_json=ui_payload,
                k_used=int(ui_payload.get("k_used") or 8),
            )
            return company, persona_name, pk, True, None

        except Exception as e:
            if _is_azure_429(e):
                wait = _retry_after_seconds_from(e) or 60.0
                time.sleep(wait + random.uniform(0, 0.25))
            else:
                if attempt >= RETRIES:
                    return company, persona_name, pk, False, f"{e}"
                time.sleep(backoff + random.uniform(0, 0.25))
                backoff *= 2

    return company, persona_name, pk, False, "unknown error"

# =========================
# Main
# =========================
def main() -> int:
    eng = get_engine()
    ensure_rag_schema(eng)
    ensure_kpi_cache(eng)

    candidates = select_candidates(eng, n=BATCH_SIZE)
    if not candidates:
        print("No personas in centralize DB.")
        return 0

    eligible = filter_not_cached(eng, candidates, ttl_days=CACHE_TTL_D)
    if not eligible:
        print(f"No eligible personas (all cached within last {CACHE_TTL_D} days).")
        return 0

    eligible = eligible[:BATCH_SIZE]
    print(f"Selected {len(eligible)} personas (not cached in last {CACHE_TTL_D} days).")

    ok = fail = 0

    if MAX_WORKERS <= 1:
        for i, row in enumerate(eligible, 1):
            c, n, pk, success, err = process_one(eng, row)
            if success:
                ok += 1
                print(f"[{i}/{len(eligible)}]  OK  :: {n} @ {c}")
            else:
                fail += 1
                print(f"[{i}/{len(eligible)}]  FAIL:: {n} @ {c} — {err}")
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(process_one, eng, r): r for r in eligible}
            i = 0
            for fut in as_completed(futures):
                i += 1
                row = futures[fut]
                try:
                    c, n, pk, success, err = fut.result()
                    if success:
                        ok += 1
                        print(f"[{i}/{len(eligible)}]  OK  :: {n} @ {c}")
                    else:
                        fail += 1
                        print(f"[{i}/{len(eligible)}]  FAIL:: {n} @ {c} — {err}")
                except Exception as e:
                    fail += 1
                    company = (row.get("company_name") or "").strip()
                    name = (row.get("client_name") or "").strip()
                    print(f"[{i}/{len(eligible)}]  EXC :: {name} @ {company} — {e}")

    print(f"Done. Success={ok}, Fail={fail}")
    return 0 if fail == 0 else 2

if __name__ == "__main__":
    sys.exit(main())
# =========================