# scripts/persona_kpi_build.py
from __future__ import annotations
import argparse, json, os, yaml
from pathlib import Path
from typing import List, Dict

from sqlalchemy import text
from utils.db import get_engine
from utils.rag_env import get_embeddings

UPSERT_SQL = """
INSERT INTO insights.persona_kpis (persona_id, kpi_name, kpi_desc, weight, patterns, embedding)
VALUES (:pid, :name, :desc, :weight, :patterns::jsonb, :emb)
ON CONFLICT (persona_id, kpi_name) DO UPDATE
SET kpi_desc = EXCLUDED.kpi_desc,
    weight = EXCLUDED.weight,
    patterns = EXCLUDED.patterns,
    embedding = EXCLUDED.embedding
"""

def embed_texts(strings: List[str]) -> List[List[float]]:
    emb = get_embeddings()
    return emb.embed_documents(strings)

def upsert_yaml(path: Path) -> int:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    pid = int(data["persona_id"])
    kpis: List[Dict] = data["kpis"]

    # build texts for embedding
    texts = [f"{k['name']} {k.get('desc','')}" for k in kpis]
    vecs = embed_texts(texts)

    with get_engine().begin() as conn:
        for k, v in zip(kpis, vecs):
            conn.execute(
                text(UPSERT_SQL),
                dict(
                    pid=pid,
                    name=k["name"],
                    desc=k.get("desc", ""),
                    weight=float(k.get("weight", 1.0)),
                    patterns=json.dumps(k.get("patterns", []), ensure_ascii=False),
                    emb=v,
                ),
            )
    return len(kpis)

def main():
    parser = argparse.ArgumentParser(description="Upsert persona KPIs from YAML files.")
    parser.add_argument("--dir", default="configs/personas", help="Directory with *.yml files")
    parser.add_argument("--file", default=None, help="Single YAML file to process")
    args = parser.parse_args()

    if args.file:
        count = upsert_yaml(Path(args.file))
        print(f"✅ Upserted {count} KPI(s) from {args.file}")
        return

    root = Path(args.dir)
    if not root.exists():
        print(f"[WARN] No directory: {root}")
        return

    total = 0
    for p in sorted(root.glob("*.yml")):
        total += upsert_yaml(p)
        print(f"✅ Upserted KPIs from {p}")
    print(f"Done. Total KPIs upserted: {total}")

if __name__ == "__main__":
    main()
