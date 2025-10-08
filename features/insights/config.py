# features/insights/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, List, Tuple

# It's safe to call even if already called elsewhere.
try:
    from dotenv import load_dotenv  # optional in prod, handy locally
    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class Settings:
    # --- External APIs ---
    tavily_api_key: str
    jina_api_key: str

    # --- Azure OpenAI (chat + embed) ---
    azure_endpoint: str
    azure_api_key: str
    azure_gpt_deployment: str
    azure_gpt_version: str
    azure_embed_deployment: str
    azure_embed_version: str

    # --- Runtime knobs ---
    http_timeout: int = 30               # seconds
    chunk_tokens: int = 900              # tokens per chunk for embedding
    top_k: int = 12                      # retrieval depth
    trust_jina_url: bool = True          # proceed even if not strictly PDF
    skip_tavily_check: bool = False      # allow bypassing Tavily YES/NO
    enable_db_cache: bool = True         # store / read results from DB cache

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self).copy()
        # Never expose secrets if this ends up logged
        for k in ("tavily_api_key", "jina_api_key", "azure_api_key"):
            if k in d and isinstance(d[k], str):
                d[k] = f"{d[k][:4]}…"
        return d


_REQUIRED_VARS: List[str] = [
    "TAVILY_API_KEY",
    "JINA_API_KEY",
    "AZURE_ENDPOINT",
    "AZURE_API_KEY",
    "AZURE_DEPLOYMENT",
    "AZURE_API_VERSION",
    "AZURE_EMBED_DEPLOYMENT",
    "AZURE_EMBED_VERSION",
]

def _read_env() -> Tuple[Dict[str, str], List[str]]:
    env = {k: os.getenv(k, "").strip() for k in _REQUIRED_VARS}
    missing = [k for k, v in env.items() if not v]
    return env, missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env, missing = _read_env()
    if missing:
        # Fail fast with a crisp message (avoid printing actual secret values)
        raise RuntimeError(
            "❌ Insights config is incomplete. Missing environment variables: "
            + ", ".join(missing)
            + "\nPlease add them to your .env or deployment secrets."
        )

    # Optional knobs with sensible defaults
    http_timeout = int(os.getenv("HTTP_TIMEOUT", "30"))
    chunk_tokens = int(os.getenv("INSIGHTS_CHUNK_TOKENS", "900"))
    top_k = int(os.getenv("INSIGHTS_TOP_K", "12"))
    trust_jina_url = os.getenv("INSIGHTS_TRUST_JINA_URL", "true").lower() in ("1", "true", "yes", "y")
    skip_tavily_check = os.getenv("INSIGHTS_SKIP_TAVILY_CHECK", "false").lower() in ("1", "true", "yes", "y")
    enable_db_cache = os.getenv("INSIGHTS_ENABLE_DB_CACHE", "true").lower() in ("1", "true", "yes", "y")

    return Settings(
        tavily_api_key=env["TAVILY_API_KEY"],
        jina_api_key=env["JINA_API_KEY"],
        azure_endpoint=env["AZURE_ENDPOINT"].rstrip("/"),
        azure_api_key=env["AZURE_API_KEY"],
        azure_gpt_deployment=env["AZURE_DEPLOYMENT"],
        azure_gpt_version=env["AZURE_API_VERSION"],
        azure_embed_deployment=env["AZURE_EMBED_DEPLOYMENT"],
        azure_embed_version=env["AZURE_EMBED_VERSION"],
        http_timeout=http_timeout,
        chunk_tokens=chunk_tokens,
        top_k=top_k,
        trust_jina_url=trust_jina_url,
        skip_tavily_check=skip_tavily_check,
        enable_db_cache=enable_db_cache,
    )
