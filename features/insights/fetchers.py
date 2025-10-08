# features/insights/fetchers.py
from __future__ import annotations

import io
import json
from typing import Dict, Optional, Tuple

import streamlit as st
import requests
from PyPDF2 import PdfReader

from .config import get_settings
from .clients import (
    get_http_session,
    get_jina_session,
    get_tavily,
    get_gpt_client,
    get_gpt_deployment_name,
)


# ---------------------------
# Utilities
# ---------------------------

def _clean_company(company: str) -> str:
    return (company or "").strip()


def _first_or_none(seq):
    if not seq:
        return None
    return seq[0]


def _safe_json(resp: requests.Response) -> Optional[Dict]:
    try:
        return resp.json()
    except Exception:
        try:
            return json.loads(resp.text)
        except Exception:
            return None


# ---------------------------
# Tavily: does the company publish annual reports?  (cached)
# ---------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def company_publishes_annual_report(company_name: str) -> bool:
    """
    Returns True if Tavily + (optionally) GPT verdict says company publishes annual reports.
    If settings.skip_tavily_check is True, we optimistically return True.
    """
    stg = get_settings()
    company = _clean_company(company_name)
    if not company:
        return False

    if stg.skip_tavily_check:
        return True

    tavily = get_tavily()
    query = f'is {company} release annual report answer only YES or NO. No explanation. filetype:pdf'
    try:
        tav = tavily.search(query=query, max_results=3, include_answer="basic")
    except Exception:
        # If Tavily fails, be conservative and return False
        return False

    extracted_answer = (tav or {}).get("answer", "").strip()
    if not extracted_answer:
        return False

    # Use GPT to normalize into strict True/False
    try:
        gpt = get_gpt_client()
        prompt = f"""Return strictly True or False.

Question: "{query}"
Tavily 'answer': "{extracted_answer}"
If the answer clearly indicates the company publishes annual reports, respond True; otherwise False.
"""
        resp = gpt.chat.completions.create(
            model=get_gpt_deployment_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        verdict = resp.choices[0].message.content.strip()
        return verdict.lower() == "true"
    except Exception:
        # If GPT check fails, fall back to a simple heuristic
        return extracted_answer.lower().startswith("yes")


# ---------------------------
# Jina: find recent annual report PDF URL  (cached)
# ---------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def jina_find_recent_annual_pdf(company_name: str) -> Optional[str]:
    """
    Uses Jina search to find a recent annual report URL.
    Returns the first URL string if any, else None.
    We trust Jina’s URL by default (config.trust_jina_url).
    """
    stg = get_settings()
    company = _clean_company(company_name)
    if not company:
        return None

    params = {"q": f"{company} recent annual report filetype:pdf"}
    s = get_jina_session()
    try:
        r = s.get("https://s.jina.ai/", params=params)
        r.raise_for_status()
    except Exception:
        return None

    data = _safe_json(r)
    if not data:
        return None

    items = data.get("data") or []
    for it in items:
        url = it.get("url")
        if url:
            return url

    return None


# ---------------------------
# PDF download + extract text  (cached by URL)
# ---------------------------

def _pdf_bytes_from_url(url: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Downloads the URL and returns (content_bytes, content_type).
    Returns (None, None) on failure.
    """
    s = get_http_session()
    try:
        resp = s.get(url)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        return resp.content, ctype
    except Exception:
        return None, None


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    """
    Best-effort text extraction from PDF bytes using PyPDF2.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception:
        return ""

    parts = []
    for page in getattr(reader, "pages", []):
        try:
            txt = page.extract_text() or ""
            parts.append(txt)
        except Exception:
            # keep going on individual page failures
            continue
    return "\n".join(parts).strip()


@st.cache_data(ttl=3600, show_spinner=True)
def pdf_to_text(pdf_url: str) -> str:
    """
    Downloads the file at pdf_url and extracts text if it's a PDF, or,
    if settings.trust_jina_url is True, tries extraction anyway.
    Returns a (possibly empty) string on failure.
    """
    stg = get_settings()
    if not pdf_url:
        return ""

    data, ctype = _pdf_bytes_from_url(pdf_url)
    if not data:
        return ""

    is_pdf = ctype is not None and "pdf" in ctype
    if not is_pdf and not stg.trust_jina_url:
        # Be strict: only parse if content-type says it's PDF
        return ""

    # Attempt PDF extraction regardless if trust_jina_url is True
    text = _extract_text_from_pdf_bytes(data)

    # If we got nothing, and content-type wasn't PDF, try naive bytes decode
    if not text and not is_pdf:
        try:
            # Some “annual report” pages are HTML — quick fallback
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    return text
