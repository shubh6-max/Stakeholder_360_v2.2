# scripts/case_chunk_embed.py
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

try:
    from utils.db import get_engine
except Exception:
    get_engine = None  # type: ignore

from utils.rag_env import PgConn, assert_env
from features.insights.case_sectionizer import sectionize
from features.insights.case_vectorstore import chunk_strings, embed_texts

# add near the other imports in scripts/case_chunk_embed.py
try:
    # ensures psycopg2 understands Python list -> vector type
    from pgvector.psycopg2 import register_vector
    register_vector()
except Exception:
    # safe to ignore if already registered or package not present
    pass


Q_SELECT_TARGET_DOCS = """
SELECT d.id, d.file_name, d.source_path, d.source_type, d.full_text, d.slide_page_text::text,
       f.industry, f.business_function, f.case_study_name
FROM insights.case_docs d
LEFT JOIN insights.case_facts f ON f.doc_id = d.id
WHERE (:doc_id IS NULL OR d.id = :doc_id)
ORDER BY d.updated_at DESC
LIMIT :limit
"""

Q_DELETE_EXISTING_CHUNKS = "DELETE FROM insights.case_chunks WHERE doc_id = :doc_id"

Q_INSERT_CHUNK = """
INSERT INTO insights.case_chunks
  (doc_id, chunk_ix, section, text, metadata, embedding)
VALUES
  (:doc_id, :chunk_ix, :section, :text, CAST(:metadata AS JSONB), :embedding)
ON CONFLICT (doc_id, chunk_ix) DO UPDATE
SET section  = EXCLUDED.section,
    text     = EXCLUDED.text,
    metadata = EXCLUDED.metadata,
    embedding= EXCLUDED.embedding,
    updated_at = NOW()
"""

@dataclass
class ChunkPlan:
    section: str
    chunks: List[str]


def _ensure_engine() -> Engine:
    if get_engine is not None:
        return get_engine()
    pg = PgConn.from_env()
    from sqlalchemy import create_engine
    return create_engine(pg.sqlalchemy_url(), future=True)


def _load_candidate_docs(engine: Engine, doc_id: Optional[int], limit: int):
    with engine.begin() as conn:
        rows = conn.execute(
            text(Q_SELECT_TARGET_DOCS),
            dict(doc_id=doc_id, limit=limit),
        ).mappings().all()
    return rows


# ---------- NEW: span detection helpers (maps chunk -> slide/page indices) ----------
import re
import math

_TOKEN_RE = re.compile(r"[A-Za-z0-9%$]+")  # keep words, %, $ for KPIs

def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]

def _overlap_score(chunk_tokens: List[str], slide_tokens: List[str]) -> float:
    if not chunk_tokens or not slide_tokens:
        return 0.0
    set_c = set(chunk_tokens)
    set_s = set(slide_tokens)
    inter = len(set_c & set_s)
    denom = math.sqrt(max(1, len(set_c))) * math.sqrt(max(1, len(set_s)))
    return inter / denom  # cosine-ish proxy using sets; cheap & stable

def _guess_spans_for_chunk(chunk_text: str, slide_page_text: List[Dict]) -> List[Dict]:
    """
    Heuristic: compute token-overlap with each slide/page and keep top matches.
    Returns [{"slide_index": i}, ...] (max 3).
    """
    ctoks = _tokens(chunk_text)
    if not ctoks:
        return []
    scored = []
    for entry in slide_page_text:
        i = int(entry.get("index", 0))
        stoks = _tokens(entry.get("text", ""))
        sc = _overlap_score(ctoks, stoks)
        if sc > 0:
            scored.append((i, sc))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [i for i, sc in scored[:3] if sc >= 0.05]  # small floor to avoid noise
    return [{"slide_index": i} for i in top]


def _build_chunks_for_doc(full_text: str, slide_page_text_json: str) -> Tuple[Dict[str, List[str]], List[Dict]]:
    try:
        spt = json.loads(slide_page_text_json) if slide_page_text_json else []
    except Exception:
        spt = []
    sect = sectionize(full_text, spt)
    out: Dict[str, List[str]] = {}
    for name, body in sect.sections.items():
        if not body.strip():
            continue
        out[name] = chunk_strings(body)
    if not out:
        out["general"] = chunk_strings(full_text)
    return out, spt


def _embed_and_upsert(engine: Engine, doc_row) -> Tuple[int, int]:
    doc_id = int(doc_row["id"])
    meta_base = {
        "file_name": doc_row["file_name"],
        "source_path": doc_row["source_path"],
        "source_type": doc_row["source_type"],
        "industry": doc_row.get("industry"),
        "business_function": doc_row.get("business_function"),
        "case_study_name": doc_row.get("case_study_name"),
    }

    # 1) Build chunks + bring slide/page list for span mapping
    section_chunks, slide_page_text = _build_chunks_for_doc(doc_row["full_text"], doc_row["slide_page_text"])

    # 2) Flatten with precomputed spans for each chunk
    flat_sections: List[str] = []
    flat_texts: List[str] = []
    flat_spans: List[List[Dict]] = []

    for sec, chunks in section_chunks.items():
        for ch in chunks:
            if ch.strip():
                flat_sections.append(sec)
                flat_texts.append(ch)
                flat_spans.append(_guess_spans_for_chunk(ch, slide_page_text))

    if not flat_texts:
        return (0, 0)

    # 3) Embed
    vecs = embed_texts(flat_texts)

    # 4) Upsert (purge existing; keep sequential chunk_ix)
    with engine.begin() as conn:
        conn.execute(text(Q_DELETE_EXISTING_CHUNKS), dict(doc_id=doc_id))
        for ix, (sec, text_val, emb, spans) in enumerate(zip(flat_sections, flat_texts, vecs, flat_spans)):
            meta = {**meta_base, "section": sec, "spans": spans}
            conn.execute(
                text(Q_INSERT_CHUNK),
                dict(
                    doc_id=doc_id,
                    chunk_ix=ix,
                    section=sec,
                    text=text_val,
                    metadata=json.dumps(meta, ensure_ascii=False),
                    embedding=emb,
                ),
            )

    return (len(section_chunks), len(flat_texts))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sectionize, chunk, embed, and upsert into insights.case_chunks (pgvector).")
    parser.add_argument("--doc-id", type=int, default=None, help="Only process this doc_id (for testing)")
    parser.add_argument("--limit", type=int, default=50, help="Max docs to process")
    args = parser.parse_args()

    try:
        assert_env()
    except Exception as e:
        print(f"[FATAL] Missing env: {e}", file=sys.stderr)
        return 2

    engine = _ensure_engine()
    rows = _load_candidate_docs(engine, args.doc_id, args.limit)
    if not rows:
        print("[INFO] No documents found to chunk/embed.")
        return 0

    print(f"[INFO] Processing {len(rows)} document(s).")
    total_chunks = 0
    total_docs = 0
    for r in rows:
        doc_id = r["id"]
        print(f" -> doc_id={doc_id} file={r['file_name']} type={r['source_type']}")
        sections, chunks = _embed_and_upsert(engine, r)
        print(f"    sections={sections} total_chunks={chunks}")
        total_docs += 1
        total_chunks += chunks

    print(f"[DONE] docs={total_docs} chunks={total_chunks}")
    print("Tip: After first bulk insert, create IVFFLAT index for speed:")
    print("""\
-- Run once you have data:
CREATE INDEX IF NOT EXISTS idx_case_chunks_embedding
ON insights.case_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
ANALYZE insights.case_chunks;""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
