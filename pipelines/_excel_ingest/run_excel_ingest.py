# pipelines/_excel_ingest/run_excel_ingest.py
import os
import json
import hashlib
import asyncio
import nest_asyncio
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from sqlalchemy import text, MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from utils.db import get_engine

# =========================================
# ⚙️ Setup
# =========================================
load_dotenv()
nest_asyncio.apply()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4-vision-preview")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

TABLE_NAME = "scout.impact_repository"
SOURCE_TYPE = "excel"
TOP_N = 10
MIN_CONF = 0.80

# =========================================
# 🧠 Prompt Template
# =========================================
PROMPT_TEMPLATE = """
You are a domain expert specializing in business impact classification for enterprise case studies and consulting use cases.

Your task is to analyze each “Impact” statement and classify it into three clearly defined fields:
1. Industry — macro business domain (CPG, Retail, Manufacturing, BFSI, Pharma, Utilities, etc.)
2. Business Group — function (Marketing, Sales, Supply Chain, Finance, Manufacturing, etc.)
3. Use Case — concise description of the analytical or business initiative.

### Confidence Logic
For each classification, include a numeric confidence score between **0 and 1**, but restrict it to clean 0.05 intervals only
(e.g., 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0).  
If multiple options exist, include up to 3 ranked options per field, sorted by descending confidence.

Return **only valid JSON** in this structure:
{{
  "Industry": [{{"value": "CPG", "confidence": 0.95}}],
  "Business Group": [{{"value": "Marketing", "confidence": 0.93}}],
  "Use Case": [{{"value": "Conjoint Analysis for Product & Pricing Optimization", "confidence": 0.97}}]
}}

Impact statement to classify:
{impact_text}
"""

# =========================================
# 🔧 Helpers
# =========================================
def compute_fingerprint(row: Dict[str, Any]) -> str:
    sig = "|".join([
        row.get("impact", "").strip().lower(),
        row.get("industry", "").strip().lower(),
        row.get("business_group", "").strip().lower(),
        row.get("use_case", "").strip().lower(),
        row.get("source_file", "").strip().lower(),
        row.get("source_type", "").strip().lower(),
    ])
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()

def safe_json_extract(text: str) -> Dict[str, Any]:
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {"Industry": [], "Business Group": [], "Use Case": []}

# =========================================
# 🧠 Async LLM Call
# =========================================
async def classify_impact(impact_text: str, idx: int):
    prompt = PROMPT_TEMPLATE.format(impact_text=impact_text)
    try:
        response = await client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            temperature=0,
            messages=[
                {"role": "system", "content": "Return only valid JSON, nothing else."},
                {"role": "user", "content": prompt},
            ],
        )
        reply = response.choices[0].message.content or ""
        return safe_json_extract(reply)
    except Exception as e:
        print(f"⚠️ Error row {idx}: {e}")
        return {"Industry": [], "Business Group": [], "Use Case": []}

# =========================================
# 🗂️ Expand & Prepare DataFrame
# =========================================
def expand_to_rows(results, impacts, source_file):
    rows = []
    for impact_text, data in zip(impacts, results):
        industries = data.get("Industry", []) or [{"value": "", "confidence": 0}]
        business_groups = data.get("Business Group", []) or [{"value": "", "confidence": 0}]
        use_cases = data.get("Use Case", []) or [{"value": "", "confidence": 0}]

        for ind in industries:
            for bg in business_groups:
                for uc in use_cases:
                    rows.append({
                        "source_file": source_file,
                        "impact": impact_text.strip(),
                        "industry": ind.get("value", ""),
                        "industry_conf": ind.get("confidence", 0),
                        "business_group": bg.get("value", ""),
                        "business_group_conf": bg.get("confidence", 0),
                        "use_case": uc.get("value", ""),
                        "use_case_conf": uc.get("confidence", 0),
                        "source_type": SOURCE_TYPE,
                        "slide_no": None,
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fingerprint"] = df.apply(lambda r: compute_fingerprint(r.to_dict()), axis=1)
    return df

# =========================================
# 🧱 Database Logic
# =========================================
def ensure_table(engine):
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS scout;
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id BIGSERIAL PRIMARY KEY,
        source_file TEXT NOT NULL,
        impact TEXT NOT NULL,
        industry TEXT,
        industry_conf DOUBLE PRECISION,
        business_group TEXT,
        business_group_conf DOUBLE PRECISION,
        use_case TEXT,
        use_case_conf DOUBLE PRECISION,
        source_type TEXT NOT NULL,
        slide_no INTEGER,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        fingerprint TEXT UNIQUE,
        embedding vector(1536)
    );
    """
    with engine.begin() as conn:
        for stmt in ddl.strip().split(";\n"):
            if stmt.strip():
                conn.execute(text(stmt))

def upsert_rows(engine, df):
    metadata = MetaData(schema="scout")
    table = Table("impact_repository", metadata, autoload_with=engine)
    valid_cols = set(table.columns.keys())
    df_filtered = df[[c for c in df.columns if c in valid_cols]].copy()
    records = df_filtered.to_dict(orient="records")

    with engine.begin() as conn:
        stmt = pg_insert(table).values(records)
        stmt = stmt.on_conflict_do_nothing(index_elements=["fingerprint"])
        res = conn.execute(stmt)
    print(f"✅ {res.rowcount} records inserted into scout.impact_repository")
    return res.rowcount

# =========================================
# 🚀 Main Logic
# =========================================
async def main(excel_path: str, source_file: str):
    df = pd.read_excel(excel_path)
    if "Impact" not in df.columns:
        raise ValueError("Excel must contain a column named 'Impact'.")

    impacts = df["Impact"].astype(str).tolist()
    all_results = []

    for i in tqdm(range(0, len(impacts), 5), desc="Classifying impacts"):
        batch = impacts[i:i+5]
        batch_tasks = [classify_impact(text, i+j) for j, text in enumerate(batch)]
        batch_results = await asyncio.gather(*batch_tasks)
        all_results.extend(batch_results)

    df_expanded = expand_to_rows(all_results, impacts, source_file)
    df_final = df_expanded[
        (df_expanded["industry_conf"] >= MIN_CONF) &
        (df_expanded["business_group_conf"] >= MIN_CONF) &
        (df_expanded["use_case_conf"] >= MIN_CONF)
    ].copy()

    if df_final.empty:
        print("ℹ️ No high-confidence rows to insert.")
        return df_final

    engine = get_engine()
    ensure_table(engine)
    upsert_rows(engine, df_final)
    print(f"✅ Excel ingest complete. Inserted={len(df_final)} rows.")
    return df_final

# =========================================
# 🏁 CLI Entrypoint
# =========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Excel impacts into Azure Postgres (scout.impact_repository)")
    parser.add_argument("excel_path", help="Path to Excel file containing 'Impact' column")
    parser.add_argument("--source-file", required=True, help="Source URL or identifier (e.g., SharePoint link)")
    args = parser.parse_args()

    asyncio.run(main(args.excel_path, args.source_file))
