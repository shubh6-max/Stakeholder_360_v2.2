# features/insights/fetch_annual.py
from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from PyPDF2 import PdfReader
from sqlalchemy.engine import Engine

from utils.rag_db import (
    ensure_rag_schema,
    get_latest_source,
    get_document_by_source,
    upsert_source_status,
    insert_document,
    SourceRow,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Config (env)
# --------------------------------------------------------------------
HTTP_TIMEOUT = int(os.getenv("RAG_HTTP_TIMEOUT", "45"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
JINA_BASE = os.getenv("JINA_BASE", "https://s.jina.ai/").strip()


# --------------------------------------------------------------------
# Small utils
# --------------------------------------------------------------------
def _clean_company(company: str) -> str:
    return (company or "").strip()


def _get_json(url: str, *, headers: Dict[str, str], params: Dict[str, str]) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return json.loads(r.text)
    except Exception as e:
        logger.warning("GET %s failed: %s", url, e)
        return None


# --------------------------------------------------------------------
# 1) Tavily: does the company publish annual reports? (YES/NO/None)
# --------------------------------------------------------------------
def company_publishes_annual_report(company: str) -> Optional[bool]:
    company = _clean_company(company)
    if not company or not TAVILY_API_KEY:
        return None

    q = f"Does {company} publish an annual report? Answer only YES or NO. filetype:pdf"
    payload = {
        "query": q,
        "include_answer": "basic",
        "max_results": 3,
    }
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}

    try:
        r = requests.post("https://api.tavily.com/search", json=payload, headers=headers, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}
        ans = (data.get("answer") or "").strip().upper()
        if ans.startswith("YES"):
            return True
        if ans.startswith("NO"):
            return False
        # weak heuristic if answer is inconclusive
        joined = " ".join(
            f"{it.get('title','')} {it.get('content','')}" for it in (data.get("results") or [])
        ).lower()
        if "annual report" in joined or "form 10-k" in joined:
            return True
        return None
    except Exception as e:
        logger.warning("Tavily presence check failed for %s: %s", company, e)
        return None


# --------------------------------------------------------------------
# 2) Jina: find a recent annual report PDF URL
# --------------------------------------------------------------------
def jina_find_recent_annual_pdf(company: str) -> Optional[str]:
    company = _clean_company(company)
    if not company or not JINA_API_KEY:
        return None

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Respond-With": "no-content",  # structured JSON
    }
    params = {"q": f"{company} recent annual report filetype:pdf"}

    data = _get_json(JINA_BASE, headers=headers, params=params)
    if not data:
        return None

    # Prefer explicit .pdf
    for it in data.get("data") or []:
        url = (it or {}).get("url")
        if url and url.lower().endswith(".pdf"):
            return url

    # fallback: first plausible URL
    for it in data.get("data") or []:
        url = (it or {}).get("url")
        if url:
            return url

    return None


# --------------------------------------------------------------------
# 3) Download PDF → extract text
# --------------------------------------------------------------------
def pdf_to_text(pdf_url: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not pdf_url:
        return (None, None)
    try:
        resp = requests.get(pdf_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        raw = resp.content
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        text_joined = "\n".join(pages).strip()
        # If nothing could be extracted, still return raw bytes so we can store the file if needed
        return (raw, text_joined)
    except Exception as e:
        logger.warning("PDF fetch/extract failed for %s: %s", pdf_url, e)
        return (None, None)


# --------------------------------------------------------------------
# Orchestrator used by pipeline
# --------------------------------------------------------------------
@dataclass
class AnnualFetchResult:
    status: str                     # 'ingested' | 'present' | 'absent' | 'error'
    source: Optional[SourceRow]
    url: Optional[str]
    document_id: Optional[int]
    text_chars: int
    note: Optional[str] = None


def fetch_or_load_annual_report(engine: Engine, company: str) -> AnnualFetchResult:
    """
    DB-first logic for annual report:
      1) If already ingested → return that.
      2) Else if present in DB (source + document) → return that.
      3) Else decide presence via Tavily (True/False/None).
      4) If False → upsert status 'absent' and return.
      5) Else try to find URL via Jina, download, store document, mark 'present'.
    """
    company = _clean_company(company)
    ensure_rag_schema(engine)

    # 1) Latest source row
    src = get_latest_source(engine, company, "annual_report")
    if src and src.status == "ingested":
        doc = get_document_by_source(engine, src.id)
        if doc:
            return AnnualFetchResult(
                status="ingested",
                source=src,
                url=src.url,
                document_id=doc.id,
                text_chars=len(doc.text or ""),
                note="Loaded from DB (ingested).",
            )
        # if no doc though status says ingested, reset to present to reflow
        src = upsert_source_status(
            engine,
            company=company,
            status="present",
            source_type="annual_report",
            url=src.url,
            note="Doc missing for 'ingested'; resetting to 'present'.",
        )

    # 2) If source is present and we have a document, return it
    if src and src.status == "present" and src.url:
        doc = get_document_by_source(engine, src.id)
        if doc:
            return AnnualFetchResult(
                status="present",
                source=src,
                url=src.url,
                document_id=doc.id,
                text_chars=len(doc.text or ""),
                note="Loaded from DB (present).",
            )

    # 3) Decide presence if unclear
    presence = None
    if src:
        if src.status in ("present", "ingested"):
            presence = True
        elif src.status == "absent":
            presence = False

    if presence is None:
        presence = company_publishes_annual_report(company)

    if presence is False:
        src = upsert_source_status(
            engine,
            company=company,
            status="absent",
            source_type="annual_report",
            note="Tavily indicated no annual report.",
        )
        return AnnualFetchResult(status="absent", source=src, url=None, document_id=None, text_chars=0)

    # 4) Get or refresh URL
    url = (src.url if src else None) or jina_find_recent_annual_pdf(company)
    if not url:
        src = upsert_source_status(
            engine,
            company=company,
            status="absent",
            source_type="annual_report",
            note="No annual report URL found.",
        )
        return AnnualFetchResult(status="absent", source=src, url=None, document_id=None, text_chars=0)

    if src and src.url != url:
        src = upsert_source_status(
            engine,
            company=company,
            status=src.status or "present",
            source_type="annual_report",
            url=url,
            note="URL refreshed",
        )
    elif not src:
        src = upsert_source_status(
            engine,
            company=company,
            status="present",
            source_type="annual_report",
            url=url,
        )

    # 5) Download + extract + store if not already stored
    raw, text_content = pdf_to_text(url)
    if raw is None and text_content is None:
        return AnnualFetchResult(
            status="error", source=src, url=url, document_id=None, text_chars=0, note="Failed to fetch/parse PDF"
        )

    doc = get_document_by_source(engine, src.id)
    if not doc:
        doc = insert_document(
            engine,
            source_id=src.id,
            company=company,
            text_content=text_content or "",
            doc_type="annual_report",
            title=None,
            year=None,
            raw_bytes=raw,  # optional
        )

    return AnnualFetchResult(
        status="present",
        source=src,
        url=url,
        document_id=doc.id if doc else None,
        text_chars=len(text_content or ""),
        note="Fetched & stored document text.",
    )