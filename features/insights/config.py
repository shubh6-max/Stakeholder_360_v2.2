# features/insights/config.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from sqlalchemy.engine import Engine
from utils.db import get_engine


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        if default is None:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return default
    return v


def _slug_company(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "company"


@dataclass(frozen=True)
class RAGSettings:
    # ---- Azure OpenAI (Chat) ----
    aoai_endpoint: str = _env("AZURE_ENDPOINT")
    aoai_api_key: str = _env("AZURE_API_KEY")
    aoai_api_version: str = _env("AZURE_API_VERSION", "2024-06-01")
    aoai_chat_deployment: str = _env("AZURE_DEPLOYMENT", "gpt-4o-mini")

    # ---- Azure OpenAI (Embeddings) ----
    aoai_embed_deployment: str = _env("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")
    aoai_embed_api_version: str = _env("AZURE_EMBED_VERSION", "2023-05-15")
    embed_dimension: int = int(os.getenv("AZURE_OPENAI_EMBED_DIM", "1536"))

    # ---- Vector store (PGVector) ----
    collection_prefix: str = os.getenv("RAG_COLLECTION_PREFIX", "rag_co")
    distance: str = os.getenv("RAG_DISTANCE", "cosine")  # cosine|l2|ip

    # ---- Chunking / Retrieval (defaults tuned for annual reports) ----
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "180"))
    min_top_k: int = int(os.getenv("RAG_MIN_TOPK", "6"))
    max_top_k: int = int(os.getenv("RAG_MAX_TOPK", "18"))

    # ---- Caching to SQL (reusing your scout.company_insights) ----
    enable_db_cache: bool = os.getenv("RAG_ENABLE_DB_CACHE", "true").lower() in ("1", "true", "yes")


def get_settings() -> RAGSettings:
    return RAGSettings()


def make_chat_model() -> AzureChatOpenAI:
    """
    LangChain chat LLM (Azure OpenAI).
    """
    stg = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=stg.aoai_endpoint,
        api_key=stg.aoai_api_key,
        api_version=stg.aoai_api_version,
        model=stg.aoai_chat_deployment,
        temperature=0.2,
        timeout=120,
        max_retries=2,
    )


def make_embeddings() -> AzureOpenAIEmbeddings:
    """
    LangChain embeddings (Azure OpenAI). Must match pgvector dimension.
    """
    stg = get_settings()
    return AzureOpenAIEmbeddings(
        azure_endpoint=stg.aoai_endpoint,
        api_key=stg.aoai_api_key,
        api_version=stg.aoai_embed_api_version,
        model=stg.aoai_embed_deployment,
        chunk_size=64,  # batching requests
    )


def pg_conn_str_from_engine(engine: Optional[Engine] = None) -> str:
    """
    Reuse your configured SQLAlchemy engine to get a full connection URL
    (including sslmode=require).
    """
    eng = engine or get_engine()
    # include password in the rendered URL for PGVector
    return eng.url.render_as_string(hide_password=False)


def company_collection_name(company: str) -> str:
    """
    Deterministic PGVector collection name per company.
    """
    stg = get_settings()
    slug = _slug_company(company)
    return f"{stg.collection_prefix}_{slug}"


def infer_top_k(num_chunks: int) -> int:
    """
    Dynamic K based on corpus size; clamped to [min_top_k, max_top_k].
    """
    stg = get_settings()
    if num_chunks <= 0:
        return stg.min_top_k
    base = max(stg.min_top_k, min(stg.max_top_k, (num_chunks // 8) + stg.min_top_k))
    return int(base)
