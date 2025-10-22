# scripts/case_ingest.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

# Prefer your existing engine helper; fall back to PgConn if not available.
try:
    from utils.db import get_engine  # your repo helper
except Exception:
    get_engine = None  # type: ignore

from utils.rag_env import PgConn, assert_env
from features.insights.case_parsers.pptx_loader import load_pptx_as_documents
from features.insights.case_parsers.pdf_loader import load_pdf_as_documents
from features.insights.case_extract_chain import extract_case_facts


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".pptx", ".pdf"}


def _file_hash(path: Path) -> str:
    """Compute a stable SHA256 hash of the binary file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_file(path: Path) -> Tuple[List[dict], str, List[dict], str, str]:
    """
    Returns:
        docs_meta: list of metadata dicts (for debug)
        full_text: concatenated text
        slide_page_text: [{"index": i, "text": "..."}]
        source_type: "pptx" | "pdf"
        file_name: basename
    """
    ext = path.suffix.lower()
    if ext == ".pptx":
        docs, full_text, spt = load_pptx_as_documents(str(path))
        source_type = "pptx"
    elif ext == ".pdf":
        docs, full_text, spt = load_pdf_as_documents(str(path))
        source_type = "pdf"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs_meta = [d.metadata for d in docs]  # debug only
    return docs_meta, full_text, spt, source_type, os.path.basename(str(path))


def _ensure_engine() -> Engine:
    if get_engine is not None:
        return get_engine()
    # fallback via rag_env
    pg = PgConn.from_env()
    from sqlalchemy import create_engine
    return create_engine(pg.sqlalchemy_url(), future=True)


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────

SQL_FIND_BY_CHECKSUM = """
SELECT d.id, d.file_name, d.page_count
FROM insights.case_docs d
WHERE d.checksum = :checksum
LIMIT 1
"""

SQL_FACTS_EXISTS = """
SELECT 1
FROM insights.case_facts
WHERE doc_id = :doc_id
LIMIT 1
"""


def upsert_case_doc(
    engine: Engine,
    *,
    file_name: str,
    source_path: str,
    source_type: str,
    checksum: str,
    page_count: int,
    full_text: str,
    slide_page_text: List[dict],
    llm_version: Optional[str] = None,
) -> int:
    """
    Insert or update insights.case_docs; return doc_id.
    """
    with engine.begin() as conn:
        r = conn.execute(
            text("""
                 INSERT INTO insights.case_docs
                 (file_name, source_path, source_type, checksum, page_count,
                 full_text, slide_page_text, llm_version)
                VALUES
                    (:file_name, :source_path, :source_type, :checksum, :page_count,
                    :full_text, CAST(:slide_page_text AS JSONB), :llm_version)
                ON CONFLICT (checksum) DO UPDATE
                SET
                    file_name       = EXCLUDED.file_name,
                    source_path     = EXCLUDED.source_path,
                    source_type     = EXCLUDED.source_type,
                    page_count      = EXCLUDED.page_count,
                    full_text       = EXCLUDED.full_text,
                    slide_page_text = EXCLUDED.slide_page_text,
                    llm_version     = EXCLUDED.llm_version,
                    updated_at      = NOW()
                RETURNING id
            """)
,
            dict(
                file_name=file_name,
                source_path=source_path,
                source_type=source_type,
                checksum=checksum,
                page_count=page_count,
                full_text=full_text,
                slide_page_text=json.dumps(slide_page_text, ensure_ascii=False),
                llm_version=llm_version,
            ),
        )
        return int(r.scalar())


def upsert_case_facts(
    engine: Engine,
    *,
    doc_id: int,
    facts: Dict,
    raw_json: Dict,
) -> None:
    """
    Insert or update insights.case_facts for a given doc_id.
    """
    with engine.begin() as conn:
        conn.execute(
            text("""
                 INSERT INTO insights.case_facts
                (doc_id, case_study_name, industry, business_function,
                problem_statement, impact_pointers, solution_approach,
                case_study_link, kpi, raw_llm_json)
                VALUES
                    (:doc_id, :case_study_name, :industry, :business_function,
                    :problem_statement, CAST(:impact_pointers AS JSONB), :solution_approach,
                    :case_study_link, CAST(:kpi AS JSONB), CAST(:raw_llm_json AS JSONB))
                ON CONFLICT (doc_id) DO UPDATE
                SET
                    case_study_name   = EXCLUDED.case_study_name,
                    industry          = EXCLUDED.industry,
                    business_function = EXCLUDED.business_function,
                    problem_statement = EXCLUDED.problem_statement,
                    impact_pointers   = EXCLUDED.impact_pointers,
                    solution_approach = EXCLUDED.solution_approach,
                    case_study_link   = EXCLUDED.case_study_link,
                    kpi               = EXCLUDED.kpi,
                    raw_llm_json      = EXCLUDED.raw_llm_json,
                    extracted_at      = NOW()
            """),
            dict(
                doc_id=doc_id,
                case_study_name=facts.get("case_study_name"),
                industry=facts.get("industry"),
                business_function=facts.get("business_function"),
                problem_statement=facts.get("problem_statement"),
                impact_pointers=json.dumps(facts.get("impact_pointers", []), ensure_ascii=False),
                solution_approach=facts.get("solution_approach"),
                case_study_link=facts.get("case_study_link"),
                kpi=json.dumps(facts.get("kpi", []), ensure_ascii=False),
                raw_llm_json=json.dumps(raw_json, ensure_ascii=False),
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main ingestion routine
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    file: str
    status: str
    doc_id: Optional[int] = None
    reason: Optional[str] = None
    checksum: Optional[str] = None
    pages: Optional[int] = None


def ingest_one(
    engine: Engine,
    path: Path,
    llm_version: Optional[str],
    *,
    reingest_existing: bool,
    backfill_facts: bool,
) -> IngestResult:
    try:
        # 0) Compute checksum first (cheap) and check if doc exists
        checksum = _file_hash(path)
        with engine.begin() as conn:
            existing = conn.execute(text(SQL_FIND_BY_CHECKSUM), {"checksum": checksum}).mappings().first()

        if existing and not reingest_existing:
            # Optionally backfill facts if missing
            if backfill_facts:
                with engine.begin() as conn:
                    has_facts = conn.execute(text(SQL_FACTS_EXISTS), {"doc_id": existing["id"]}).scalar()
                if not has_facts:
                    # Need to parse file to extract full_text for facts
                    _, full_text, slide_page_text, source_type, file_name = _load_file(path)
                    facts_obj, raw_json = extract_case_facts(full_text, llm_version=llm_version)
                    upsert_case_facts(engine, doc_id=int(existing["id"]), facts=facts_obj.model_dump(), raw_json=raw_json)
                    return IngestResult(
                        file=str(path),
                        status="backfilled_facts",
                        doc_id=int(existing["id"]),
                        checksum=checksum,
                        pages=len(slide_page_text),
                    )
            # Skip entirely
            return IngestResult(file=str(path), status="skipped_existing", doc_id=int(existing["id"]), checksum=checksum)

        # 1) Load and stitch text (only if new or reingesting)
        _, full_text, slide_page_text, source_type, file_name = _load_file(path)
        page_count = len(slide_page_text)

        # 2) Upsert case_docs
        doc_id = upsert_case_doc(
            engine,
            file_name=file_name,
            source_path=str(path.resolve()),
            source_type=source_type,
            checksum=checksum,
            page_count=page_count,
            full_text=full_text,
            slide_page_text=slide_page_text,
            llm_version=llm_version,
        )

        # 3) Run LLM extraction (structured JSON) and upsert case_facts
        facts_obj, raw_json = extract_case_facts(full_text, llm_version=llm_version)
        upsert_case_facts(engine, doc_id=doc_id, facts=facts_obj.model_dump(), raw_json=raw_json)

        return IngestResult(file=str(path), status="ok", doc_id=doc_id, checksum=checksum, pages=page_count)

    except Exception as e:
        return IngestResult(file=str(path), status="error", reason=str(e))


def find_files(root: Path, limit: Optional[int] = None) -> List[Path]:
    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
            if limit and len(files) >= limit:
                break
    return files


def write_summary(rows: List[IngestResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"case_ingest_summary_{ts}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "doc_id", "checksum", "pages", "reason"])
        for r in rows:
            w.writerow([r.file, r.status, r.doc_id or "", r.checksum or "", r.pages or "", r.reason or ""])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PPTX/PDF case studies into Postgres (case_docs + case_facts).")
    parser.add_argument("--root", default="data/case_studies", help="Folder to scan for .pptx/.pdf")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files (for testing)")
    parser.add_argument("--llm-version", default=os.getenv("AZURE_DEPLOYMENT", "azure-chat-model"),
                        help="Stored in case_docs.llm_version and raw_llm_json._llm_version")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Where to save the summary CSV")

    # NEW flags:
    parser.add_argument("--reingest-existing", action="store_true",
                        help="Force re-extract and overwrite for files already ingested (same checksum).")
    parser.add_argument("--no-backfill-facts", action="store_true",
                        help="Do NOT backfill case_facts for existing docs that don't have facts yet.")

    args = parser.parse_args()

    try:
        assert_env()
    except Exception as e:
        print(f"[FATAL] Missing env: {e}", file=sys.stderr)
        return 2

    engine = _ensure_engine()

    root = Path(args.root)
    if not root.exists():
        print(f"[FATAL] Root folder not found: {root}", file=sys.stderr)
        return 2

    paths = find_files(root, args.limit)
    if not paths:
        print(f"[INFO] No PPTX/PDF files under: {root}")
        return 0

    print(f"[INFO] Found {len(paths)} file(s) under {root}")

    results: List[IngestResult] = []
    for i, p in enumerate(paths, start=1):
        print(f"[{i}/{len(paths)}] Ingesting: {p}")
        res = ingest_one(
            engine,
            p,
            args.llm_version,
            reingest_existing=args.reingest_existing,
            backfill_facts=not args.no_backfill_facts,
        )
        results.append(res)
        if res.status in ("ok", "backfilled_facts", "skipped_existing"):
            print(f"   -> {res.status} doc_id={res.doc_id} pages={res.pages if res.pages is not None else '—'}")
        else:
            print(f"   -> ERROR: {res.reason}", file=sys.stderr)

    # summary_path = write_summary(results, Path(args.artifacts_dir))
    # print(f"[INFO] Wrote summary: {summary_path}")

    # Exit code: 0 if any success-like status; 1 if all failed
    success_like = {"ok", "backfilled_facts", "skipped_existing"}
    if all(r.status not in success_like for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
