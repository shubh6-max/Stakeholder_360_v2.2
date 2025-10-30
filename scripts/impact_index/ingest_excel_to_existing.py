# scripts/impact_index/ingest_excel_to_existing.py
import os
import math
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text as sql_text
from langchain_openai import AzureOpenAIEmbeddings
from utils.db import get_engine
from s360_rag.config import AZURE_ENDPOINT, AZURE_API_KEY, AZURE_EMBED_DEPLOYMENT

load_dotenv()

EXCEL_PATH  = os.getenv("UC_EXCEL_PATH", r"C:\Users\ShubhamVishwasPurani\OneDrive - TheMathCompany Private Limited\pranjal_impact_pointers.xlsx")
SHEET_NAME  = os.getenv("UC_SHEET_NAME", "0")
BATCH_SIZE  = int(os.getenv("UC_EMBED_BATCH", "64"))  # tune if needed

def make_embedder() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        model=AZURE_EMBED_DEPLOYMENT,
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
    )

def row_to_ids(r: dict) -> tuple[str, str, str]:
    """
    Returns (case_id, case_title, chunk_text)
    """
    import hashlib
    industry = (r.get("Industry") or "").strip()
    group    = (r.get("Business Group") or "").strip()
    usecase  = (r.get("Use Case") or "").strip()
    impact   = (r.get("Impact") or "").strip()

    case_title = f"{industry} | {group} | {usecase}".strip(" |")
    case_key   = f"{industry}|{group}|{usecase}|{impact}"
    case_id    = hashlib.sha1(case_key.encode("utf-8")).hexdigest()
    chunk_text = f"Industry: {industry} | Business Group: {group} | Use Case: {usecase} | Impact: {impact}"
    return case_id, case_title, chunk_text

def run():
    # 1) Read Excel
    print(f"Reading Excel: {os.path.abspath(EXCEL_PATH)}")
    df = pd.read_excel(EXCEL_PATH, sheet_name=int(SHEET_NAME) if str(SHEET_NAME).isdigit() else SHEET_NAME)
    print(f"Rows to ingest: {len(df)}")

    required = ["Industry", "Business Group", "Use Case", "Impact"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Missing column in Excel: {col}")

    # Normalize dictionary rows
    rows = []
    for _, r in df.iterrows():
        d = {k: (str(r[k]).strip() if pd.notna(r[k]) else "") for k in required}
        rows.append(d)

    # 2) Prepare IDs / titles / texts
    meta_params = []
    case_ids = []
    chunk_texts = []
    source_file = f"excel:{os.path.basename(EXCEL_PATH)}"

    for r in rows:
        cid, title, chunk = row_to_ids(r)
        meta_params.append({
            "cid": cid,
            "title": title,
            "industry": r["Industry"],
            "bfunc": r["Business Group"],
            "src": source_file
        })
        case_ids.append(cid)
        chunk_texts.append(chunk)

    # 3) Batch embeddings
    embedder = make_embedder()
    vectors: list[list[float]] = []

    total = len(chunk_texts)
    num_batches = math.ceil(total / BATCH_SIZE)
    for i in range(num_batches):
        start = i * BATCH_SIZE
        end   = min(start + BATCH_SIZE, total)
        batch = chunk_texts[start:end]
        # embed_documents returns one vector per input, efficient over network
        vecs = embedder.embed_documents(batch)
        vectors.extend(vecs)
        print(f"Embedded {end}/{total}")

    assert len(vectors) == len(chunk_texts), "Embedding count mismatch."

    # 4) DB work (one transaction)
    eng = get_engine()
    with eng.begin() as conn:
        # Be tolerant if extension creation isn't permitted
        conn.execute(sql_text("CREATE SCHEMA IF NOT EXISTS scout;"))
        try:
            conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector;"))
        except Exception:
            pass

        # 4a) Upsert cs_meta in bulk
        upsert_meta_sql = sql_text("""
            INSERT INTO scout.cs_meta
                (case_id, case_title, client, industry, business_function, source_file, total_pages)
            VALUES
                (:cid, :title, '', :industry, :bfunc, :src, 1)
            ON CONFLICT (case_id) DO UPDATE SET
                case_title = EXCLUDED.case_title,
                industry   = EXCLUDED.industry,
                business_function = EXCLUDED.business_function,
                source_file= EXCLUDED.source_file,
                total_pages= EXCLUDED.total_pages
        """)
        conn.execute(upsert_meta_sql, meta_params)
        print(f"Upserted meta rows: {len(meta_params)}")

        # 4b) Delete all existing chunks for these case_ids in one go
        if case_ids:
            conn.execute(
                sql_text("DELETE FROM scout.cs_chunks WHERE case_id = ANY(:ids)"),
                {"ids": case_ids},
            )

        # 4c) Bulk insert chunks with vectors
        insert_chunk_sql = sql_text("""
            INSERT INTO scout.cs_chunks (case_id, page_no, chunk, embedding)
            VALUES (:cid, 1, :chunk, CAST(:emb AS vector))
        """)

        chunk_params = []
        for cid, chunk, emb in zip(case_ids, chunk_texts, vectors):
            chunk_params.append({"cid": cid, "chunk": chunk, "emb": emb})

        # executemany-style when passing a list of dicts
        conn.execute(insert_chunk_sql, chunk_params)
        print(f"Inserted chunk rows: {len(chunk_params)}")

    print("✅ Ingestion complete.")

if __name__ == "__main__":
    run()
