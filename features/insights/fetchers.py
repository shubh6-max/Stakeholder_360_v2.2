# features/insights/fetchers.py
from __future__ import annotations

import io
import json
from typing import Dict, Optional, Tuple, List

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
        return False

    extracted_answer = (tav or {}).get("answer", "").strip()
    if not extracted_answer:
        return False

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
        verdict = (resp.choices[0].message.content or "").strip()
        return verdict.lower() == "true"
    except Exception:
        return extracted_answer.lower().startswith("yes")


# ---------------------------
# Jina: find recent annual / quarterly PDF URLs  (cached)
# ---------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def jina_find_recent_annual_pdf(company_name: str) -> Optional[str]:
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

    for it in (data.get("data") or []):
        url = it.get("url")
        if url:
            return url
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def jina_find_recent_quarterly_pdf(company_name: str) -> Optional[str]:
    """
    Broader query to catch 10-Q / earnings / quarterly PDF decks.
    """
    company = _clean_company(company_name)
    if not company:
        return None

    params = {"q": f"{company} recent quarterly report OR 10-Q OR earnings presentation filetype:pdf"}
    s = get_jina_session()
    try:
        r = s.get("https://s.jina.ai/", params=params)
        r.raise_for_status()
    except Exception:
        return None

    data = _safe_json(r)
    if not data:
        return None

    for it in (data.get("data") or []):
        url = it.get("url")
        if url:
            return url
    return None


# ---------------------------
# News fallback (no scraping): concise context via Tavily
# ---------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def tavily_company_news_context(company_name: str) -> str:
    """
    Returns a compact text context summarizing recent news about
    {company} around stakeholder/customer 360 & analytics.
    We DO NOT scrape pages—only use Tavily answer + snippets.
    """
    company = _clean_company(company_name)
    if not company:
        return ""

    tav = get_tavily()
    q = (
        f"latest news about {company} related to stakeholder 360 OR customer 360 OR analytics "
        f"within last 12 months. Provide a concise summary and bullet list of key items."
    )
    try:
        res = tav.search(query=q, max_results=5, include_answer="advanced")
    except Exception:
        return ""

    ans = (res or {}).get("answer") or ""
    records: List[str] = []
    for item in (res or {}).get("results", []):
        title = (item.get("title") or "").strip()
        snippet = (item.get("content") or "").strip()
        if title:
            records.append(f"- {title}: {snippet[:240]}")

    block = ans.strip()
    if records:
        block += "\n\nTop items:\n" + "\n".join(records)
    return block.strip()


# ---------------------------
# PDF download + extract text  (cached by URL)
# ---------------------------

def _pdf_bytes_from_url(url: str) -> Tuple[Optional[bytes], Optional[str]]:
    s = get_http_session()
    try:
        resp = s.get(url)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        return resp.content, ctype
    except Exception:
        return None, None


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception:
        return ""
    parts: List[str] = []
    for page in getattr(reader, "pages", []):
        try:
            txt = page.extract_text() or ""
            parts.append(txt)
        except Exception:
            continue
    return "\n".join(parts).strip()


@st.cache_data(ttl=3600, show_spinner=True)
def pdf_to_text(pdf_url: str) -> str:
    stg = get_settings()
    if not pdf_url:
        return ""

    data, ctype = _pdf_bytes_from_url(pdf_url)
    if not data:
        return ""

    is_pdf = ctype is not None and "pdf" in ctype
    if not is_pdf and not stg.trust_jina_url:
        return ""

    text = _extract_text_from_pdf_bytes(data)

    if not text and not is_pdf:
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    return text
