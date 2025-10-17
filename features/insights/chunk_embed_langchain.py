# features/insights/chunk_embed_langchain.py
from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any, Optional

from sqlalchemy.engine import Engine
from sqlalchemy.sql import text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

from utils.rag_db import insert_chunks, EMBED_DIM, ensure_rag_schema


# -----------------------------------------------------------------------------
# Env / settings
# -----------------------------------------------------------------------------
# Prefer the canonical names but gracefully fall back to older variants
_AZURE_ENDPOINT = (os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
_AZURE_API_KEY = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or ""
_AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")
_AZURE_EMBED_VERSION = os.getenv("AZURE_EMBED_VERSION", "2023-05-15")

# Character-based chunking (keeps us light—no tiktoken dependency here)
DEFAULT_CHUNK_TOKENS = int(os.getenv("RAG_CHUNK_TOKENS", "900"))
DEFAULT_OVERLAP_TOKENS = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _approx_chars(tokens: int) -> int:
    """Very rough 4 chars ~= 1 token heuristic."""
    return max(256, int(tokens * 4))


def _make_splitter(chunk_tokens: Optional[int], overlap_tokens: Optional[int]) -> RecursiveCharacterTextSplitter:
    ct = chunk_tokens or DEFAULT_CHUNK_TOKENS
    ov = overlap_tokens or DEFAULT_OVERLAP_TOKENS
    return RecursiveCharacterTextSplitter(
        chunk_size=_approx_chars(ct),
        chunk_overlap=_approx_chars(ov),
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _make_embedder() -> AzureOpenAIEmbeddings:
    if not (_AZURE_ENDPOINT and _AZURE_API_KEY and _AZURE_EMBED_DEPLOYMENT):
        raise RuntimeError(
            "Azure embeddings not configured. Set AZURE_ENDPOINT, AZURE_API_KEY, "
            "AZURE_EMBED_DEPLOYMENT, AZURE_EMBED_VERSION."
        )

    # NOTE: langchain_openai uses 'openai_api_version' param name
    return AzureOpenAIEmbeddings(
        azure_endpoint=_AZURE_ENDPOINT,
        api_key=_AZURE_API_KEY,
        azure_deployment=_AZURE_EMBED_DEPLOYMENT,
        openai_api_version=_AZURE_EMBED_VERSION,
        chunk_size=128,  # batch size for embed_documents
    )


def _get_doc_text(engine: Engine, document_id: int) -> str:
    """
    Fetch the 'text' from rag.documents by id.
    - Uses scalar_one_or_none() on SA 1.4/2.x for clarity.
    - Gracefully falls back to scalar() for older SA.
    - Returns '' if missing.
    - Narrow exception handling to avoid masking real issues.
    """
    sql = text("SELECT text FROM rag.documents WHERE id = :id")

    try:
        with engine.begin() as conn:  # uses a short transaction; safe for read
            try:
                val = conn.execute(sql, {"id": document_id}).scalar_one_or_none()
            except AttributeError:
                # Older SQLAlchemy: scalar_one_or_none not available
                val = conn.execute(sql, {"id": document_id}).scalar()

        return (val or "").strip()

    except DBAPIError as db_err:
        # DB connectivity/driver issues (keep message visible in your traceback UI)
        raise RuntimeError(f"Database error while reading document {document_id}: {db_err}") from db_err
    except SQLAlchemyError as sa_err:
        # SQLAlchemy layer errors (bad SQL, transaction problems, etc.)
        raise RuntimeError(f"SQLAlchemy error while reading document {document_id}: {sa_err}") from sa_err


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _has_chunks(engine: Engine, document_id: int) -> bool:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT COUNT(*) FROM rag.chunks WHERE document_id=:id"),
                           {"id": document_id}).first()
        return bool(row and int(row[0]) > 0)
    

def chunk_and_embed_document(
    *,
    engine: Engine,
    document_id: int,
    company: str,
    raw_text: Optional[str] = None,      # ← now optional
    chunk_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Idempotent-ish ingestion step:
      - If raw_text is None, loads text from rag.documents.
      - Splits into chunks.
      - Embeds with Azure OpenAI (text-embedding-3-small; 1536 dims).
      - Inserts into rag.chunks via insert_chunks().

    Returns stats dict:
      {
        "document_id": int,
        "company": str,
        "total_chunks_for_doc": int,
        "chunks_created": int,
        "embed_dim": int,
        "note": str
      }
    """
    ensure_rag_schema(engine)

    # ✅ Early-exit if already embedded
    if _has_chunks(engine, document_id):
        with engine.begin() as conn:
            total_row = conn.execute(
                text("SELECT COUNT(*) FROM rag.chunks WHERE document_id = :id"),
                {"id": document_id},
            ).first()
        total_for_doc = int(total_row[0]) if total_row else 0
        return {
            "document_id": document_id,
            "company": company,
            "total_chunks_for_doc": total_for_doc,
            "chunks_created": 0,
            "embed_dim": EMBED_DIM,
            "note": "Already embedded; skipped.",
        }

    text_body = (raw_text or "").strip()
    if not text_body:
        text_body = _get_doc_text(engine, document_id).strip()

    if not text_body:
        return {
            "document_id": document_id,
            "company": company,
            "total_chunks_for_doc": 0,
            "chunks_created": 0,
            "embed_dim": EMBED_DIM,
            "note": "No text found for document.",
        }

    # 1) Split
    splitter = _make_splitter(chunk_tokens, overlap_tokens)
    chunks: List[str] = splitter.split_text(text_body)
    if not chunks:
        return {
            "document_id": document_id,
            "company": company,
            "total_chunks_for_doc": 0,
            "chunks_created": 0,
            "embed_dim": EMBED_DIM,
            "note": "Splitter produced no chunks.",
        }

    # 2) Embed
    embedder = _make_embedder()
    vectors: List[List[float]] = embedder.embed_documents(chunks)
    if not vectors or len(vectors) != len(chunks):
        raise RuntimeError("Embedding failed or length mismatch between chunks and vectors.")

    # 3) Insert
    insert_chunks(
        engine,
        document_id=document_id,
        company=company,
        chunks=chunks,
        embeddings=vectors,
    )

    # 4) Count total for doc (post-insert)
    with engine.begin() as conn:
        total_row = conn.execute(
            text("SELECT COUNT(*) FROM rag.chunks WHERE document_id = :id"),
            {"id": document_id},
        ).first()
        total_for_doc = int(total_row[0]) if total_row else len(chunks)

    return {
        "document_id": document_id,
        "company": company,
        "total_chunks_for_doc": total_for_doc,
        "chunks_created": len(chunks),
        "embed_dim": EMBED_DIM,
        "note": "OK",
    }


def embed_query_vector(text: str) -> List[float]:
    """Embed a single query string (for ad-hoc retrieval)."""
    t = (text or "").strip()
    if not t:
        return [0.0] * EMBED_DIM
    embedder = _make_embedder()
    return embedder.embed_query(t)
