import hashlib
from typing import List, Dict, Tuple
from sqlalchemy import text
from sqlalchemy.engine import Engine
from services.embeddings import embed_texts
from utils.logger import logger

REQUIRED_COLS = [
    "Impact","Industry","Industry_Conf","BusinessGroup","BusinessGroup_Conf","UseCase","UseCase_Conf"
]

def ensure_tables(engine: Engine):
    # Tables created via scripts/bootstrap_db.py, but safe to call.
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS scout;"))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS scout.impact_repository (
          id BIGSERIAL PRIMARY KEY,
          source_type TEXT NOT NULL,
          source_file TEXT,
          impact TEXT NOT NULL,
          industry TEXT,
          industry_conf FLOAT,
          business_group TEXT,
          business_group_conf FLOAT,
          use_case TEXT,
          use_case_conf FLOAT,
          fingerprint TEXT UNIQUE NOT NULL,
          embedding vector(1536),
          created_at TIMESTAMP DEFAULT NOW()
        );"""))
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_impact_repo_embedding_ivfflat
        ON scout.impact_repository USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """))

def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def make_fingerprint(row: Dict, source_type: str, source_file: str) -> str:
    # Strong, stable dedupe: impact text + source_type + source_file basename
    core = "|".join([
        source_type.lower().strip(),
        (source_file or "").strip(),
        _normalize(row.get("Impact",""))
    ])
    return hashlib.sha256(core.encode("utf-8")).hexdigest()

def _validate_df(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

def _rows_for_upsert(df, source_type, source_file) -> List[Dict]:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "impact": (r.get("Impact") or "").strip(),
            "industry": (r.get("Industry") or "") or "",
            "industry_conf": float(r.get("Industry_Conf") or 0) or 0.0,
            "business_group": (r.get("BusinessGroup") or "") or "",
            "business_group_conf": float(r.get("BusinessGroup_Conf") or 0) or 0.0,
            "use_case": (r.get("UseCase") or "") or "",
            "use_case_conf": float(r.get("UseCase_Conf") or 0) or 0.0,
            "source_type": source_type,
            "source_file": source_file,
        })
    return rows

def upsert_impact_df(engine: Engine, df, *, source_type: str, source_file: str) -> Dict[str,int]:
    """
    Idempotent upsert into scout.impact_repository.
    - Computes fingerprint for each row.
    - Embeds only rows not already present.
    - Uses ON CONFLICT DO NOTHING on fingerprint.
    """
    _validate_df(df)
    rows = _rows_for_upsert(df, source_type, source_file)

    # Compute fingerprints
    for row in rows:
        row["fingerprint"] = make_fingerprint(row, source_type, source_file)

    # Filter out existing fingerprints
    fps = tuple(r["fingerprint"] for r in rows)
    existing = set()
    if fps:
        placeholders = ",".join([f":fp{i}" for i in range(len(fps))])
        params = {f"fp{i}": fp for i, fp in enumerate(fps)}
        with engine.begin() as conn:
            q = text(f"SELECT fingerprint FROM scout.impact_repository WHERE fingerprint IN ({placeholders})")
            res = conn.execute(q, params).fetchall()
            existing = set(r[0] for r in res)

    to_insert = [r for r in rows if r["fingerprint"] not in existing]
    skipped = len(rows) - len(to_insert)
    if not to_insert:
        logger.info("No new rows to insert. Skipped=%d", skipped)
        return {"inserted": 0, "skipped": skipped}

    # Embed impacts
    embeddings = embed_texts([r["impact"] for r in to_insert])

    # Bulk insert
    with engine.begin() as conn:
        insert_sql = text("""
            INSERT INTO scout.impact_repository
            (source_type, source_file, impact, industry, industry_conf,
             business_group, business_group_conf, use_case, use_case_conf, fingerprint, embedding)
            VALUES
            (:source_type, :source_file, :impact, :industry, :industry_conf,
             :business_group, :business_group_conf, :use_case, :use_case_conf, :fingerprint, :embedding)
            ON CONFLICT (fingerprint) DO NOTHING
        """)
        for row, emb in zip(to_insert, embeddings):
            params = {**row, "embedding": emb}
            conn.execute(insert_sql, params)

    inserted = len(to_insert)
    logger.info("Upsert complete. Inserted=%d Skipped=%d", inserted, skipped)
    return {"inserted": inserted, "skipped": skipped}
