# utils/rag_db.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Sequence, Dict, Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "1532"))   # text-embedding-3-small → 1536
IVFFLAT_LISTS = int(os.getenv("RAG_IVFFLAT_LISTS", "100"))

# -----------------------------------------------------------------------------
# Dataclasses (lightweight typed rows)
# -----------------------------------------------------------------------------
@dataclass
class SourceRow:
    id: int
    company_name: str
    source_type: str
    url: Optional[str]
    status: Optional[str]
    note: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]

@dataclass
class DocumentRow:
    id: int
    source_id: int
    company_name: str
    doc_type: str
    title: Optional[str]
    year: Optional[int]
    text: str
    created_at: Optional[str]
    updated_at: Optional[str]

# -----------------------------------------------------------------------------
# Schema bootstrap
# -----------------------------------------------------------------------------
def ensure_rag_schema(engine: Engine) -> None:
    """
    Idempotent setup for:
      - rag.company_sources (with updated_at)
      - rag.documents (with updated_at)
      - rag.chunks (pgvector column with EMBED_DIM)
      - IVFFLAT indexes
      - compatibility view rag.sources → rag.company_sources
    If rag.chunks.embedding dim != EMBED_DIM and RAG_AUTO_MIGRATE=1, auto-migrates.
    """
    target_dim = EMBED_DIM
    auto_migrate = os.getenv("RAG_AUTO_MIGRATE", "0").lower() in ("1", "true", "yes")

    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS rag;

    -- canonical sources table
    CREATE TABLE IF NOT EXISTS rag.company_sources (
        id            SERIAL PRIMARY KEY,
        company_name  TEXT NOT NULL,
        source_type   TEXT NOT NULL,      -- e.g., 'annual_report'
        url           TEXT,
        status        TEXT NOT NULL DEFAULT 'present',
        note          TEXT,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at    TIMESTAMPTZ
    );
    -- older installs: add updated_at if missing
    ALTER TABLE rag.company_sources
      ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

    -- documents table
    CREATE TABLE IF NOT EXISTS rag.documents (
        id            SERIAL PRIMARY KEY,
        source_id     INT REFERENCES rag.company_sources(id) ON DELETE CASCADE,
        company_name  TEXT NOT NULL,
        doc_type      TEXT NOT NULL,      -- e.g., 'annual_report'
        title         TEXT,
        year          INT,
        text          TEXT,
        raw_pdf       BYTEA,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at    TIMESTAMPTZ
    );
    ALTER TABLE rag.documents
      ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

    -- pgvector extension must exist before vector columns
    CREATE EXTENSION IF NOT EXISTS vector;

    -- compatibility view rag.sources -> rag.company_sources
    DO $$
    BEGIN
      -- if nothing named rag.sources exists, create the view
      IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'rag' AND c.relname = 'sources'
      ) THEN
        CREATE VIEW rag.sources AS SELECT * FROM rag.company_sources;

      -- if it's a VIEW, keep it fresh
      ELSIF EXISTS (
        SELECT 1 FROM information_schema.views
         WHERE table_schema = 'rag' AND table_name = 'sources'
      ) THEN
        CREATE OR REPLACE VIEW rag.sources AS SELECT * FROM rag.company_sources;

      -- else: a TABLE or other object called rag.sources exists — leave it alone
      END IF;
    END$$;

    -- chunks table with pgvector embedding
    DO $chunks$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema='rag' AND table_name='chunks'
      ) THEN
        CREATE TABLE rag.chunks (
          id           SERIAL PRIMARY KEY,
          document_id  INT REFERENCES rag.documents(id) ON DELETE CASCADE,
          chunk_idx    INT NOT NULL,
          text         TEXT NOT NULL,
          embedding    vector({target_dim}) NOT NULL,
          created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
      END IF;
    END
    $chunks$;
    """

    with engine.begin() as conn:
        conn.execute(text(ddl))

        # Check/auto-migrate embedding dim
        dim_sql = """
        SELECT atttypmod - 4 AS dim
          FROM pg_attribute a
          JOIN pg_class c ON c.oid = a.attrelid
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname='rag' AND c.relname='chunks' AND a.attname='embedding'
        """
        row = conn.execute(text(dim_sql)).first()
        current_dim = int(row[0]) if row and row[0] is not None else None

        if current_dim is not None and current_dim != target_dim:
            if not auto_migrate:
                raise RuntimeError(
                    f"rag.chunks.embedding has dim={current_dim}, but RAG_EMBED_DIM={target_dim}. "
                    f"Set RAG_AUTO_MIGRATE=1 to auto-migrate, or drop/recreate the table."
                )
            conn.execute(text("DROP INDEX IF EXISTS rag.idx_chunks_embedding;"))
            conn.execute(text("DROP INDEX IF EXISTS rag.idx_chunks_doc_idx;"))
            conn.execute(text("DROP TABLE IF EXISTS rag.chunks CASCADE;"))
            conn.execute(text(f"""
                CREATE TABLE rag.chunks (
                  id           SERIAL PRIMARY KEY,
                  document_id  INT REFERENCES rag.documents(id) ON DELETE CASCADE,
                  chunk_idx    INT NOT NULL,
                  text         TEXT NOT NULL,
                  embedding    vector({target_dim}) NOT NULL,
                  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """))

        # IVFFLAT on embedding (cosine operator class by default)
        conn.execute(text(f"""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname='idx_chunks_embedding' AND n.nspname='rag'
          ) THEN
            CREATE INDEX idx_chunks_embedding
              ON rag.chunks USING ivfflat (embedding vector_cosine_ops)
              WITH (lists = {IVFFLAT_LISTS});
          END IF;
        END$$;
        """))

        # Covering index for doc/id traversal
        conn.execute(text("""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname='idx_chunks_doc_idx' AND n.nspname='rag'
          ) THEN
            CREATE INDEX idx_chunks_doc_idx ON rag.chunks(document_id, chunk_idx);
          END IF;
        END$$;
        """))

# -----------------------------------------------------------------------------
# Source helpers
# -----------------------------------------------------------------------------
def upsert_source_status(
    engine: Engine,
    *,
    company: str,
    status: str,
    source_type: str,
    url: Optional[str] = None,
    note: Optional[str] = None,
) -> SourceRow:
    """Insert a status row in rag.company_sources (append-only log of statuses)."""
    company = (company or "").strip()
    source_type = (source_type or "").strip()
    ensure_rag_schema(engine)
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO rag.company_sources (company_name, source_type, url, status, note)
            VALUES (:company, :source_type, :url, :status, :note)
            """),
            {"company": company, "source_type": source_type, "url": url, "status": status, "note": note},
        )
        row = conn.execute(
            text("""
            SELECT id, company_name, source_type, url, status, note,
                   created_at, updated_at
              FROM rag.company_sources
             WHERE company_name = :company AND source_type = :source_type
             ORDER BY created_at DESC
             LIMIT 1
            """),
            {"company": company, "source_type": source_type},
        ).mappings().first()
    return SourceRow(**row)

def get_latest_source(engine: Engine, company: str, source_type: str) -> Optional[SourceRow]:
    company = (company or "").strip()
    source_type = (source_type or "").strip()
    ensure_rag_schema(engine)
    with engine.begin() as conn:
        row = conn.execute(
            text("""
            SELECT id, company_name, source_type, url, status, note,
                   created_at, updated_at
              FROM rag.company_sources
             WHERE company_name = :company AND source_type = :source_type
             ORDER BY created_at DESC
             LIMIT 1
            """),
            {"company": company, "source_type": source_type},
        ).mappings().first()
        return SourceRow(**row) if row else None

def mark_source_ingested(engine: Engine, source_id: int) -> None:
    ensure_rag_schema(engine)
    with engine.begin() as conn:
        conn.execute(
            text("""
            UPDATE rag.company_sources
               SET status = 'ingested', updated_at = NOW()
             WHERE id = :id
            """),
            {"id": source_id},
        )

# -----------------------------------------------------------------------------
# Documents
# -----------------------------------------------------------------------------
def insert_document(
    engine: Engine,
    *,
    source_id: int,
    company: str,
    text_content: str,
    doc_type: str,
    title: Optional[str] = None,
    year: Optional[int] = None,
    raw_bytes: Optional[bytes] = None,
) -> DocumentRow:
    ensure_rag_schema(engine)
    with engine.begin() as conn:
        row = conn.execute(
            text("""
            INSERT INTO rag.documents
                (source_id, company_name, doc_type, title, year, text, raw_pdf)
            VALUES
                (:source_id, :company, :doc_type, :title, :year, :text, :raw_pdf)
            RETURNING id, source_id, company_name, doc_type, title, year, text, created_at, updated_at
            """),
            {
                "source_id": source_id,
                "company": (company or "").strip(),
                "doc_type": (doc_type or "").strip(),
                "title": title,
                "year": year,
                "text": text_content or "",
                "raw_pdf": raw_bytes,
            },
        ).mappings().first()
    return DocumentRow(**row)

def get_document_by_source(engine: Engine, source_id: int) -> Optional[DocumentRow]:
    ensure_rag_schema(engine)
    with engine.begin() as conn:
        row = conn.execute(
            text("""
            SELECT id, source_id, company_name, doc_type, title, year, text,
                   created_at, updated_at
              FROM rag.documents
             WHERE source_id = :sid
             ORDER BY created_at DESC
             LIMIT 1
            """),
            {"sid": source_id},
        ).mappings().first()
        return DocumentRow(**row) if row else None

# Handy small helpers for indexing logic
def latest_document_id_for_company(engine: Engine, company: str) -> Optional[int]:
    """Return most recent rag.documents.id for a company (or None)."""
    company = (company or "").strip()
    ensure_rag_schema(engine)
    with engine.begin() as conn:
        row = conn.execute(
            text("""
            SELECT id
              FROM rag.documents
             WHERE company_name = :c
             ORDER BY created_at DESC
             LIMIT 1
            """),
            {"c": company},
        ).first()
        return int(row[0]) if row else None

def chunk_count_for_document(engine: Engine, document_id: int) -> int:
    """Fast COUNT(*) of chunks for a given document id."""
    ensure_rag_schema(engine)
    with engine.begin() as conn:
        val = conn.execute(
            text("SELECT COUNT(*) FROM rag.chunks WHERE document_id = :d"),
            {"d": int(document_id)},
        ).scalar()
        return int(val or 0)

def has_chunks(engine: Engine, document_id: int) -> bool:
    """True if any chunks exist for the document (short-circuit check)."""
    return chunk_count_for_document(engine, document_id) > 0

# -----------------------------------------------------------------------------
# Chunks (insert)
# -----------------------------------------------------------------------------
def _to_vector_literal(vec: Sequence[float]) -> str:
    """Convert Python list of floats to pgvector text literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

def insert_chunks(
    engine: Engine,
    *,
    document_id: int,
    company: str,  # retained for API compatibility (not stored on chunks)
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
) -> None:
    """
    Insert chunk rows for a document into rag.chunks.
    Normalized design: no company_name column here—derive via document_id.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunks({len(chunks)}) and embeddings({len(embeddings)}) length mismatch")

    # Validate dim on first vector to fail fast
    if embeddings and len(embeddings[0]) != EMBED_DIM:
        raise RuntimeError(f"Embedding dim={len(embeddings[0])} does not match table dim={EMBED_DIM}.")

    payload = []
    for idx, (txt, vec) in enumerate(zip(chunks, embeddings)):
        payload.append({
            "document_id": int(document_id),
            "chunk_idx": int(idx),
            "text": txt or "",
            "embedding": _to_vector_literal(vec),  # text → CAST(:embedding AS vector(dim))
        })

    sql = text(f"""
        INSERT INTO rag.chunks (document_id, chunk_idx, text, embedding)
        VALUES (:document_id, :chunk_idx, :text, CAST(:embedding AS vector({EMBED_DIM})))
    """)

    ensure_rag_schema(engine)
    with engine.begin() as conn:
        conn.execute(sql, payload)

# -----------------------------------------------------------------------------
# Optional: simple similarity search example (L2)
# -----------------------------------------------------------------------------
def search_chunks_l2(
    engine: Engine,
    *,
    company: str,
    query_vector: Sequence[float],
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    L2-op search against rag.chunks joined with rag.documents for company filtering.
    """
    if len(query_vector) != EMBED_DIM:
        raise RuntimeError(f"Query vector dim={len(query_vector)}) does not match table dim={EMBED_DIM}.")

    vec_lit = _to_vector_literal(query_vector)
    sql = text(f"""
        SELECT c.id,
               c.document_id,
               d.company_name,
               c.chunk_idx AS chunk_index,
               c.text,
               (c.embedding <-> CAST(:q AS vector({EMBED_DIM}))) AS score
          FROM rag.chunks c
          JOIN rag.documents d ON d.id = c.document_id
         WHERE d.company_name = :company
         ORDER BY score ASC
         LIMIT :k
    """)
    ensure_rag_schema(engine)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"q": vec_lit, "company": (company or "").strip(), "k": int(k)}).mappings().all()
        return [dict(r) for r in rows]
