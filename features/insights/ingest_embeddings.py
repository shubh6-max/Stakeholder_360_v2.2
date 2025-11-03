# features/insights/ingest_embeddings.py
from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable, Tuple

import tiktoken
from sqlalchemy.engine import Engine

from openai import AzureOpenAI

from utils.rag_db import (
    bulk_insert_chunks,
    mark_source_ingested,
    get_document_by_id,
    get_chunks_for_document,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# ENV: Azure OpenAI embeddings
# -----------------------------------------------------------------------------
AZURE_ENDPOINT         = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY          = os.getenv("AZURE_API_KEY", "")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")
AZURE_EMBED_VERSION    = os.getenv("AZURE_EMBED_VERSION", "2023-05-15")

# This deployment outputs 1536-dim vectors
EMBED_DIM = 1536

# Batching for API calls (keep well under service limits)
EMBED_BATCH = int(os.getenv("RAG_EMBED_BATCH", "64"))


# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------
@dataclass
class Chunk:
    ord_idx: int           # order in document (0..N-1)
    text: str
    token_count: int


def _get_encoder():
    """cl100k_base is compatible with text-embedding-3-* models."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        # fallback (should not happen in modern tiktoken)
        return tiktoken.encoding_for_model("text-embedding-3-small")


def smart_chunk(
    text: str,
    target_tokens: int = 1800,
    overlap_tokens: int = 120,
    hard_limit_tokens: int = 2400,
) -> List[Chunk]:
    """
    Token-aware chunker for long PDFs.
    - target ~900 tokens per chunk with ~120 overlap
    - never exceed hard_limit_tokens
    """
    enc = _get_encoder()
    toks = enc.encode(text or "")
    n = len(toks)
    if n == 0:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    while start < n:
        end = min(start + target_tokens, n)
        # hard limit safety
        if end - start > hard_limit_tokens:
            end = start + hard_limit_tokens

        # try to extend a little to sentence boundary (look ahead ~50 tokens)
        lookahead = min(end + 50, n)
        # decode a small window and try to split on punctuation
        window = enc.decode(toks[start:lookahead])
        # heuristics: prefer to break after ., ?, !, \n
        split_at = None
        for p in [".\n", ". ", "?\n", "? ", "!\n", "! ", "\n\n", "\n"]:
            k = window.rfind(p)
            if k != -1 and (start + enc.encode(window[:k + 1]).__len__()) - start >= target_tokens * 0.6:
                split_at = start + len(enc.encode(window[:k + 1]))
                break

        if split_at is None:
            split_at = end

        part_tokens = toks[start:split_at]
        part_text = enc.decode(part_tokens).strip()
        if part_text:
            chunks.append(Chunk(ord_idx=idx, text=part_text, token_count=len(part_tokens)))
            idx += 1

        # move with overlap
        start = max(split_at - overlap_tokens, split_at)

    return chunks


# -----------------------------------------------------------------------------
# Embedding client
# -----------------------------------------------------------------------------
def _make_embed_client() -> AzureOpenAI:
    if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_EMBED_DEPLOYMENT and AZURE_EMBED_VERSION):
        raise RuntimeError("❌ Azure embedding env vars missing (AZURE_ENDPOINT, AZURE_API_KEY, AZURE_EMBED_DEPLOYMENT, AZURE_EMBED_VERSION).")
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_EMBED_VERSION,
    )


def _embed_batch(client: AzureOpenAI, batch_texts: List[str]) -> List[List[float]]:
    """
    Returns list of embeddings for a batch of texts.
    """
    if not batch_texts:
        return []
    resp = client.embeddings.create(
        model=AZURE_EMBED_DEPLOYMENT,
        input=batch_texts,
    )
    # ensure order preserved
    vectors = [d.embedding for d in resp.data]
    return vectors


def embed_chunks(text_chunks: List[Chunk]) -> List[Tuple[Chunk, List[float]]]:
    """
    Embed all chunks with batching.
    """
    client = _make_embed_client()

    paired: List[Tuple[Chunk, List[float]]] = []
    i = 0
    while i < len(text_chunks):
        batch = text_chunks[i : i + EMBED_BATCH]
        vecs = _embed_batch(client, [c.text for c in batch])
        if len(vecs) != len(batch):
            raise RuntimeError("Embedding API returned mismatched count.")
        for c, v in zip(batch, vecs):
            if len(v) != EMBED_DIM:
                logger.warning("Unexpected embedding dim: %s (expected %s)", len(v), EMBED_DIM)
            paired.append((c, v))
        i += EMBED_BATCH
    return paired


# -----------------------------------------------------------------------------
# Ingestion orchestrator
# -----------------------------------------------------------------------------
def ingest_document_embeddings(
    engine: Engine,
    *,
    document_id: int,
    company: str,
    source_id: int,
    text_content: str,
    overwrite_if_exists: bool = False,
) -> Dict[str, Any]:
    """
    - If chunks already exist for this document and overwrite_if_exists=False, skip.
    - Otherwise: chunk, embed, bulk insert, mark source as ingested.
    Returns a small dict summary.
    """
    # 0) idempotency
    existing = get_chunks_for_document(engine, document_id)
    if existing and not overwrite_if_exists:
        # still mark ingested (safe no-op) so pipeline is consistent
        mark_source_ingested(engine, source_id)
        return {
            "document_id": document_id,
            "company": company,
            "chunks": len(existing),
            "skipped": True,
            "message": "Chunks already present; skipped embedding.",
        }

    # 1) chunk
    chunks = smart_chunk(text_content or "")
    if not chunks:
        # still insert a tiny placeholder chunk to enable QA later
        chunks = [Chunk(ord_idx=0, text=(text_content or "")[:800], token_count=0)]

    # 2) embed
    pairs = embed_chunks(chunks)

    # 3) shape rows for DB
    rows = []
    for c, vec in pairs:
        rows.append({
            "document_id": document_id,
            "ord_idx": c.ord_idx,
            "text": c.text,
            "embedding": vec,           # pgvector will take this float[] via SQLAlchemy
            "token_count": c.token_count,
        })

    # 4) bulk insert
    bulk_insert_chunks(engine, rows)

    # 5) mark ingested
    mark_source_ingested(engine, source_id)

    return {
        "document_id": document_id,
        "company": company,
        "chunks": len(rows),
        "skipped": False,
        "message": "Embeddings inserted and source marked as ingested.",
    }


# -----------------------------------------------------------------------------
# Optional helper for a single-call pipeline step
# -----------------------------------------------------------------------------
def ensure_doc_embedded(
    engine: Engine,
    *,
    document_id: int,
    company: str,
    source_id: int,
    text_content: str,
) -> Dict[str, Any]:
    """
    Thin wrapper to embed only if needed.
    """
    return ingest_document_embeddings(
        engine,
        document_id=document_id,
        company=company,
        source_id=source_id,
        text_content=text_content,
        overwrite_if_exists=False,
    )
# -----------------------------------------------------------------------------