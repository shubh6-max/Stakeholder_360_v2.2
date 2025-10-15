# features/stakeholders/service.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from typing import Dict, Optional
import re
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from utils.db import get_engine

# ----------------------------
# Public surface
# ----------------------------

# Columns the UI is allowed to update.
ALLOWED_UPDATE_FIELDS: Tuple[str, ...] = (
    # Lead Identification & Contact
    "lead_priority", "client_name", "client_designation", "location",
    "email_address", "linkedin_url",

    # Company & Department Info
    "account", "working_group", "business_unit", "service_line",

    # Org Hierarchy (editable fields)
    "reporting_manager", "reporting_manager_designation",

    # Lead Status & Tracking
    "csl_owner", "reachout_lever", "reachout_channel", "status",

    # Engagement & Outreach
    "internal_research", "external_research", "introduction_path",
    "pursued_in_past", "context", "comments",
    "mathco_spoc_1", "mathco_spoc_2", "mathco_spoc_3",
    "scout_linkedin_connected_flag",

    # Expertise & Experience
    "seniority_level", "kpi", "intel_summary", "email_template", "impact_pointers",

    # Contractor Info
    "contractor_count", "vendor_name",

    # Dates (optional in UI)
    "first_outreach_date", "last_outreach_date",
)

# Type helpers
BOOL_FIELDS: Tuple[str, ...] = (
    "pursued_in_past",
    "scout_linkedin_connected_flag",
    "recent_role_change_last_3_months",
)

INT_FIELDS: Tuple[str, ...] = ("contractor_count",)

DATE_FIELDS: Tuple[str, ...] = ("first_outreach_date", "last_outreach_date")

URL_FIELDS: Tuple[str, ...] = ("linkedin_url", "intel_link")

EMAIL_FIELDS: Tuple[str, ...] = ("email_address",)

# For select dropdowns (distincts)
DISTINCT_COLS: Tuple[str, ...] = (
    "account", "working_group", "business_unit", "service_line", "reporting_manager"
)

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", re.IGNORECASE)


@dataclass
class UpdateKey:
    """How we identify a record for updates."""
    email_address: Optional[str]
    client_name: Optional[str]
    account: Optional[str]


# ----------------------------
# Distincts for dropdowns
# ----------------------------

def get_distincts(engine: Optional[Engine] = None) -> Dict[str, List[str]]:
    """
    Return distinct sorted values for dropdown-driven columns.
    """
    eng = engine or get_engine()
    out: Dict[str, List[str]] = {}
    with eng.begin() as conn:
        for col in DISTINCT_COLS:
            sql = text(f"SELECT DISTINCT {col} FROM scout.centralize_db WHERE {col} IS NOT NULL AND {col} <> ''")
            vals = [r[0] for r in conn.execute(sql).fetchall()]
            # normalized, sorted, non-empty
            vals = sorted({str(v).strip() for v in vals if str(v).strip()})
            out[col] = vals
    return out


# ----------------------------
# Normalization & Validation
# ----------------------------

def _none_if_blank(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None

def _coerce_bool(v: Any) -> Optional[bool]:
    if v in (None, "", "null"):
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return None

def _coerce_int(v: Any) -> Optional[int]:
    if v in (None, "", "null"):
        return None
    try:
        return int(v)
    except Exception:
        return None

def _coerce_date(v: Any) -> Optional[date]:
    if v in (None, "", "null"):
        return None
    if isinstance(v, (datetime, date)):
        return v.date() if isinstance(v, datetime) else v
    try:
        # Accept ISO yyyy-mm-dd
        return datetime.fromisoformat(str(v)).date()
    except Exception:
        return None

def _valid_url(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://")

def _validate_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Return {field: error} for invalid fields; empty dict means OK.
    """
    errors: Dict[str, str] = {}

    # email
    for f in EMAIL_FIELDS:
        if f in payload and payload[f]:
            if not EMAIL_RE.match(payload[f]):
                errors[f] = "Invalid email."

    # url fields
    for f in URL_FIELDS:
        if f in payload and payload[f]:
            if not _valid_url(payload[f]):
                errors[f] = "URL must start with http(s)://"

    # ints
    for f in INT_FIELDS:
        if f in payload and payload[f] is not None:
            try:
                iv = int(payload[f])
                if iv < 0:
                    errors[f] = "Must be ≥ 0"
            except Exception:
                errors[f] = "Must be an integer"

    # dates
    for f in DATE_FIELDS:
        if f in payload and payload[f] is not None:
            if not isinstance(payload[f], (date, datetime)):
                errors[f] = "Invalid date"

    # cross-field: first_outreach_date <= last_outreach_date
    f1, f2 = payload.get("first_outreach_date"), payload.get("last_outreach_date")
    if f1 and f2:
        d1 = f1.date() if isinstance(f1, datetime) else f1
        d2 = f2.date() if isinstance(f2, datetime) else f2
        if d1 > d2:
            errors["last_outreach_date"] = "Must be ≥ First outreach date"

    return errors


def normalize_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only ALLOWED_UPDATE_FIELDS and coerce types.
    Empty strings become NULL (None). Dates become `date`.
    """
    out: Dict[str, Any] = {}
    for k in ALLOWED_UPDATE_FIELDS:
        if k not in raw:
            continue
        v = raw[k]

        if k in BOOL_FIELDS:
            out[k] = _coerce_bool(v)
            continue
        if k in INT_FIELDS:
            out[k] = _coerce_int(v)
            continue
        if k in DATE_FIELDS:
            out[k] = _coerce_date(v)
            continue

        # default: trimmed string or None
        out[k] = _none_if_blank(v)

    # final validation
    errors = _validate_payload(out)
    if errors:
        raise ValueError(f"Validation failed: {errors}")
    return out


def diff_changes(original_row: Union[pd.Series, Dict[str, Any]],
                 edited_payload: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    """
    Returns a dict of changed fields: {field: (old, new)}
    Only compares ALLOWED_UPDATE_FIELDS.
    """
    orig = original_row.to_dict() if isinstance(original_row, pd.Series) else dict(original_row)
    changes: Dict[str, Tuple[Any, Any]] = {}
    for k, new_val in edited_payload.items():
        if k not in ALLOWED_UPDATE_FIELDS:
            continue
        old_val = orig.get(k)
        # Normalize dates to date for comparison
        if isinstance(old_val, datetime):
            old_val = old_val.date()
        if isinstance(new_val, datetime):
            new_val = new_val.date()
        if old_val != new_val:
            changes[k] = (old_val, new_val)
    return changes


# ----------------------------
# Key building & update
# ----------------------------

def build_update_key(original_row: Union[pd.Series, Dict[str, Any]]) -> UpdateKey:
    src = original_row.to_dict() if isinstance(original_row, pd.Series) else dict(original_row)
    email = _none_if_blank(src.get("email_address"))
    client_name = _none_if_blank(src.get("client_name"))
    account = _none_if_blank(src.get("account"))
    return UpdateKey(email_address=email, client_name=client_name, account=account)


def update_centralize_record(
    changes: Dict[str, Any],
    *,
    key: UpdateKey,
    orig_last_update: Optional[Union[datetime, str]],
    editor_identifier: Optional[str] = None,   # e.g., user email for audit
    engine: Optional[Engine] = None,
) -> Dict[str, Any]:
    """
    Apply partial update with optimistic locking via last_update_date.
    Returns:
      {
        "ok": bool,
        "conflict": bool,
        "updated_at": datetime | None,
        "error": str | None,
      }
    """
    if not changes:
        return {"ok": False, "conflict": False, "updated_at": None, "error": "No changes."}

    # Compose dynamic SET clause
    set_cols = [f"{col} = :{col}" for col in changes.keys()]
    set_clause = ", ".join(set_cols + ["last_update_date = now()"])

    # WHERE key: prefer email, else (client_name & account)
    where_email = "email_address = :key_email"
    where_fallback = "(client_name = :key_client AND account = :key_account)"
    where_key = where_email if key.email_address else where_fallback

    # Optimistic lock (allow NULL the first time)
    where_lock = "(last_update_date = :orig_lu OR :orig_lu IS NULL)"

    sql = text(f"""
        UPDATE scout.centralize_db
        SET {set_clause}
        WHERE {where_key}
          AND {where_lock}
        RETURNING last_update_date
    """)

    params: Dict[str, Any] = {**changes}
    params["key_email"] = key.email_address
    params["key_client"] = key.client_name
    params["key_account"] = key.account
    params["orig_lu"] = orig_last_update

    eng = engine or get_engine()
    try:
        with eng.begin() as conn:
            row = conn.execute(sql, params).fetchone()
            if not row:
                # conflict or key not found
                return {"ok": False, "conflict": True, "updated_at": None, "error": None}

            updated_at = row[0]

            # Optional audit (best-effort)
            try:
                _audit_changes(conn, key, changes, editor_identifier, updated_at)
            except Exception:
                # Don't fail update if audit insert fails
                pass

            return {"ok": True, "conflict": False, "updated_at": updated_at, "error": None}

    except Exception as e:
        return {"ok": False, "conflict": False, "updated_at": None, "error": str(e)}


def _audit_changes(
    conn: Connection,
    key: UpdateKey,
    changes: Dict[str, Any],
    editor_identifier: Optional[str],
    updated_at: datetime,
) -> None:
    """
    Best-effort append-only audit. Creates row only if table exists.
    You can create the table:
      create table if not exists scout.centralize_db_audit(
        id bigserial primary key,
        edited_at timestamptz not null default now(),
        edited_by text,
        client_key text,
        changes jsonb
      );
    """
    client_key = (
        key.email_address
        if key.email_address
        else f"{key.client_name}|{key.account}"
    )
    sql = text("""
        INSERT INTO scout.centralize_db_audit(edited_at, edited_by, client_key, changes)
        VALUES (:ts, :by, :ck, CAST(:chg AS JSONB))
    """)
    conn.execute(sql, {
        "ts": updated_at,
        "by": editor_identifier or "",
        "ck": client_key or "",
        "chg": pd.Series(changes).to_json(),
    })


_ALLOWED_COLUMNS = set([
    "account","subsidiary","working_group","business_unit","service_line",
    "lead_priority","client_name","client_designation","seniority_level",
    "reporting_manager","reporting_manager_designation",
    "email_address","linkedin_url","location",
    "internal_research","external_research","personalization_notes",
    "vendor_name","contractor_count",
    "reachout_lever","reachout_channel","pursued_in_past","context","introduction_path",
    "mathco_spoc_1","mathco_spoc_2",
    "first_outreach_date","last_outreach_date",
])

def update_centralize_by_identity(
    payload: Dict[str, object],
    *,
    email: Optional[str] = None,
    client: Optional[str] = None,
    account: Optional[str] = None,
) -> Dict[str, object]:
    """
    Direct UPDATE for scout.centralize_db by identity.
    Identity is either:
      - email_address = :email, OR
      - (client_name = :client AND account = :account)

    Returns: {"ok": bool, "updated_at": iso, "rows": int} or {"ok": False, "error": "..."}
    """
    if not payload:
        return {"ok": False, "error": "Empty payload."}

    where = None
    params: Dict[str, object] = {}

    if email:
        where = "email_address = :_email"
        params["_email"] = email.strip()
    elif client and account:
        where = "client_name = :_client AND account = :_account"
        params["_client"] = client.strip()
        params["_account"] = account.strip()
    else:
        return {"ok": False, "error": "Missing identity (email OR client+account)."}

    # Only allow known columns
    cols = [c for c in payload.keys() if c in _ALLOWED_COLUMNS]
    if not cols:
        return {"ok": False, "error": "No allowed columns in payload."}

    # Build SET clause + last_update_date
    set_parts = []
    for c in cols:
        set_parts.append(f'{c} = :{c}')
        params[c] = payload[c]

    # last_update_date as an ISO string (your column is text)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    set_parts.append("last_update_date = :_ts")
    params["_ts"] = ts

    sql = f"""
        UPDATE scout.centralize_db
           SET {", ".join(set_parts)}
         WHERE {where}
    """
    try:
        eng = get_engine()
        with eng.begin() as con:
            res = con.execute(text(sql), params)
            rows = res.rowcount or 0
        return {"ok": rows > 0, "rows": rows, "updated_at": ts}
    except Exception as e:
        return {"ok": False, "error": str(e)}
