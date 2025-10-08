# features/insights/clients.py
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache
from typing import Dict, Optional

from openai import AzureOpenAI

from .config import get_settings


# --------- HTTP session (Jina & generic) ---------
def _build_retrying_session(
    *,
    total: int = 4,
    backoff_factor: float = 0.6,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
    allowed_methods: frozenset = frozenset({"GET", "POST"}),
    timeout: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None,
) -> requests.Session:
    """
    Requests session with sane retry/backoff defaults.
    """
    s = requests.Session()
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    # Defaults (identifiable UA is helpful for troubleshooting)
    default_headers = {
        "User-Agent": "Stakeholder360/insights (Streamlit)",
        "Accept": "application/json, */*;q=0.5",
    }
    if headers:
        default_headers.update(headers)
    s.headers.update(default_headers)

    # Attach a per-request timeout via a wrapper
    if timeout is not None:
        original_request = s.request

        def _request_with_timeout(method, url, **kwargs):
            if "timeout" not in kwargs:
                kwargs["timeout"] = timeout
            return original_request(method, url, **kwargs)

        s.request = _request_with_timeout  # type: ignore[attr-defined]

    return s


@lru_cache(maxsize=1)
def get_http_session() -> requests.Session:
    """
    Shared session for general outbound calls (uses global HTTP_TIMEOUT).
    """
    stg = get_settings()
    return _build_retrying_session(timeout=stg.http_timeout)


@lru_cache(maxsize=1)
def get_jina_session() -> requests.Session:
    """
    Session configured with Jina API auth.
    """
    stg = get_settings()
    headers = {
        "Authorization": f"Bearer {stg.jina_api_key}",
        "X-Respond-With": "no-content",  # request structured JSON where supported
    }
    return _build_retrying_session(timeout=stg.http_timeout, headers=headers)


# --------- Tavily client (lazy) ---------
class TavilyLite:
    """
    Minimal Tavily client wrapper with our retrying session, to avoid
    importing the full SDK if you prefer a light footprint.

    If you prefer the official SDK, you can swap this with:
        from tavily import TavilyClient
        return TavilyClient(api_key=stg.tavily_api_key)
    """
    def __init__(self, api_key: str, session: requests.Session):
        self.api_key = api_key
        self.session = session
        self.base_url = "https://api.tavily.com/search"

    def search(self, *, query: str, max_results: int = 3, include_answer: str = "basic") -> Dict:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": include_answer,
        }
        resp = self.session.post(self.base_url, json=payload)
        resp.raise_for_status()
        return resp.json()


@lru_cache(maxsize=1)
def get_tavily():
    stg = get_settings()
    # Use our shared session for consistency
    return TavilyLite(stg.tavily_api_key, get_http_session())


# --------- Azure OpenAI clients ---------
@lru_cache(maxsize=1)
def get_gpt_client() -> AzureOpenAI:
    """
    Azure OpenAI chat client (for GPT completions).
    """
    stg = get_settings()
    return AzureOpenAI(
        azure_endpoint=stg.azure_endpoint,
        api_key=stg.azure_api_key,
        api_version=stg.azure_gpt_version,
    )


@lru_cache(maxsize=1)
def get_embed_client() -> AzureOpenAI:
    """
    Azure OpenAI embeddings client.
    Note: Same SDK class, different api_version (embed) per your settings.
    """
    stg = get_settings()
    return AzureOpenAI(
        azure_endpoint=stg.azure_endpoint,
        api_key=stg.azure_api_key,
        api_version=stg.azure_embed_version,
    )


# --------- Helpers to access deployment names ---------
def get_gpt_deployment_name() -> str:
    return get_settings().azure_gpt_deployment


def get_embed_deployment_name() -> str:
    return get_settings().azure_embed_deployment
