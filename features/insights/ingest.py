# features/insights/ingest.py
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

# LangChain splitters + Azure OpenAI embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

from utils.rag_db import (
    ensure_rag_schema,
    insert_chunks,
    mark_source_ingested,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ENV / Config
# ---------------------------------------------------------------------
# Chunking defaults — tuned for annual reports (dense paragraphs)
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))         # characters
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))    # characters

# Azure OpenAI (embeddings)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_API_KEY", "").strip()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_ENDPOINT", "").strip()
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small").strip()
AZURE_EMBED_VERSION = os.getenv("AZURE_EMBED_VERSION", "2023-05-15").strip()

# For pgvector dim sanity (text-embedding-3-small → 1536)
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "1536"))


# ---------------------------------------------------------------------
# Results dataclass
# ---------------------------------------------------------------------
@dataclass
class IngestResult:
    company: str
    source_id: int
    document_id: int
    chunks: int
    chars: int
    embed_ms: int
    db_ms: int


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _build_embeddings_client() -> AzureOpenAIEmbeddings:
    if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_EMBED_DEPLOYMENT and AZURE_EMBED_VERSION):
        raise RuntimeError("Azure OpenAI env is incomplete. Check AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_EMBED_*")

    return AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_EMBED_DEPLOYMENT,
        openai_api_version=AZURE_EMBED_VERSION,
        chunk_size=64,  # batch embed size (vectors/request). Tweak if you hit rate limits.
    )


def _split_text(text: str,
                chunk_size: int = DEFAULT_CHUNK_SIZE,
                chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=False,
        separators=["\n\n", "\n", ". ", " ", ""],  # coarse → fine
    )
    return splitter.split_text(text or "")


def _load_document_text(engine: Engine, document_id: int) -> str:
    sql = text("SELECT text FROM rag.documents WHERE id = :id")
    with engine.begin() as conn:
        row = conn.execute(sql, {"id": document_id}).mappings().first()
        return (row["text"] or "") if row else ""


def _purge_existing_chunks(engine: Engine, document_id: int) -> None:
    # Safe idempotency: remove any previous chunks for this document
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM rag.chunks WHERE document_id = :id"), {"id": document_id})


# ---------------------------------------------------------------------
# Public: ingest a stored document (split → embed → write chunks → mark ingested)
# ---------------------------------------------------------------------
def ingest_document(
    engine: Engine,
    *,
    company: str,
    source_id: int,
    document_id: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Optional[IngestResult]:
    """
    Reads the document text from rag.documents, splits into chunks, embeds with Azure,
    writes into rag.chunks, and finally marks the source as 'ingested'.
    """
    ensure_rag_schema(engine)

    text_blob = _load_document_text(engine, document_id)
    if not (text_blob or "").strip():
        logger.warning("ingest_document: empty text for document_id=%s", document_id)
        return None

    # 1) Split
    chunks = _split_text(text_blob, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        logger.warning("ingest_document: produced zero chunks (document_id=%s)", document_id)
        return None

    # 2) Embed (batched by LC client)
    embed_client = _build_embeddings_client()
    t0 = time.time()
    vectors = embed_client.embed_documents(chunks)  # List[List[float]] length == len(chunks)
    t1 = time.time()

    # Sanity: ensure embedding dims match table definition
    if not vectors or len(vectors[0]) != EMBED_DIM:
        raise RuntimeError(
            f"Embedding dim mismatch: got {len(vectors[0]) if vectors else 'N/A'} vs expected {EMBED_DIM}. "
            "Check AZURE_EMBED_DEPLOYMENT / RAG_EMBED_DIM / rag.chunks.embedding dimension."
        )

    # 3) Persist
    _purge_existing_chunks(engine, document_id)
    t2 = time.time()
    insert_chunks(
        engine=engine,
        document_id=document_id,
        company=company,
        chunks=chunks,
        embeddings=vectors,
    )
    t3 = time.time()

    # 4) Mark source as ingested (we only do this after chunks are written)
    mark_source_ingested(engine, source_id)

    return IngestResult(
        company=company,
        source_id=source_id,
        document_id=document_id,
        chunks=len(chunks),
        chars=len(text_blob),
        embed_ms=int((t1 - t0) * 1000),
        db_ms=int((t3 - t2) * 1000),
    )