# s360_rag/db.py
from __future__ import annotations

import os
from contextlib import contextmanager
from sqlalchemy import Column, Integer, Text, TIMESTAMP, func, text as sql_text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pgvector.sqlalchemy import Vector

# Engine factory (your util)
from utils.db import get_engine

# Config knobs (schema/table names + embed dim)
try:
    from s360_rag.config import (
        RAG_EMBED_DIM,
        SCHEMA,
        TBL_CS_META,
        TBL_CS_CHUNKS,
        TBL_RAG_QUERY_LOGS,
        TBL_PERSONA_KPIS,
    )
except Exception:
    # Sensible fallbacks if config is not importable
    RAG_EMBED_DIM = 1536
    SCHEMA = "scout"
    TBL_CS_META = "cs_meta"
    TBL_CS_CHUNKS = "cs_chunks"
    TBL_RAG_QUERY_LOGS = "rag_query_logs"
    TBL_PERSONA_KPIS = "persona_kpis_store"

IVFFLAT_LISTS = int(os.getenv("RAG_IVFFLAT_LISTS", "100"))

# engine/session
engine = get_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


class CsMeta(Base):
    __tablename__ = TBL_CS_META
    __table_args__ = {"schema": SCHEMA}
    case_id = Column(Text, primary_key=True)
    case_title = Column(Text)
    client = Column(Text)
    industry = Column(Text)
    business_function = Column(Text)
    source_file = Column(Text, nullable=False)
    total_pages = Column(Integer)
    created_at = Column(TIMESTAMP, server_default=func.now())


class CsChunks(Base):
    __tablename__ = TBL_CS_CHUNKS
    __table_args__ = {"schema": SCHEMA}
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Text, index=True)
    page_no = Column(Integer)
    chunk = Column(Text, nullable=False)
    embedding = Column(Vector(RAG_EMBED_DIM), index=False)  # ANN index created separately
    created_at = Column(TIMESTAMP, server_default=func.now())


class RagQueryLogs(Base):
    __tablename__ = TBL_RAG_QUERY_LOGS
    __table_args__ = {"schema": SCHEMA}
    id = Column(Integer, primary_key=True, autoincrement=True)
    persona = Column(Text)
    kpis = Column(Text)        # JSON stringified
    topk_cases = Column(Text)  # JSON stringified
    chosen_case = Column(Text)
    match_found = Column(Text) # "true"/"false"
    latency_ms = Column(Integer)
    ts = Column(TIMESTAMP, server_default=func.now())


class PersonaKPIStore(Base):
    __tablename__ = TBL_PERSONA_KPIS
    __table_args__ = {"schema": SCHEMA}
    id = Column(Integer, primary_key=True, autoincrement=True)
    client_name = Column(Text)
    client_designation = Column(Text)
    seniority_level = Column(Text)
    business_functions = Column(Text)
    industry_hint = Column(Text)
    kpis = Column(Text)  # JSON stringified blob
    created_at = Column(TIMESTAMP, server_default=func.now())


def create_tables() -> None:
    """
    Idempotently ensure:
      - schema exists
      - pgvector extension enabled
      - ORM tables created
      - IVFFLAT cosine index on embeddings
    """
    with engine.begin() as conn:
        conn.execute(sql_text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector;"))

    Base.metadata.create_all(engine)

    # Create ANN index (cosine) on embeddings — safe to run repeatedly
    try:
        with engine.begin() as conn:
            conn.execute(
                sql_text(
                    f"""
                    CREATE INDEX IF NOT EXISTS {SCHEMA}_{TBL_CS_CHUNKS}_vec_idx
                    ON {SCHEMA}.{TBL_CS_CHUNKS}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {IVFFLAT_LISTS});
                    """
                )
            )
    except Exception:
        # Non-fatal (e.g., missing perms). You still have a plain table scan fallback.
        pass


def raw_cosine_topk(session: Session, query_vec: list[float], top_k: int):
    """
    Return rows: (case_id, case_title, source_file, page_no, chunk, distance)
    Using pgvector cosine distance `<=>` (lower = better).
    IMPORTANT: We CAST the bound param to 'vector' so Postgres uses (vector <=> vector).
    """
    sql = sql_text(
        f"""
        SELECT
            m.case_id,
            m.case_title,
            m.source_file,
            c.page_no,
            c.chunk,
            (c.embedding <=> CAST(:emb AS vector)) AS distance
        FROM {SCHEMA}.{TBL_CS_CHUNKS} c
        JOIN {SCHEMA}.{TBL_CS_META}   m ON m.case_id = c.case_id
        ORDER BY c.embedding <=> CAST(:emb AS vector)
        LIMIT :k
        """
    )
    return session.execute(sql, {"emb": query_vec, "k": top_k}).fetchall()


def similarity_from_distance(distance: float) -> float:
    """
    Convert cosine distance to a cosine similarity proxy:
      sim ≈ 1 - distance
    """
    try:
        return max(0.0, min(1.0, 1.0 - float(distance)))
    except Exception:
        return 0.0


@contextmanager
def get_session() -> Session:
    """Safe contextmanager for DB sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
