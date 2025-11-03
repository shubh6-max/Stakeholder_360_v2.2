#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-time Excel ingester for scout.impact_repository

- Reads an Excel with an `Impact` column
- Classifies each row via Azure OpenAI into (Industry, Business Group, Use Case) + confidences
- Generates text embeddings
- Deduplicates via SHA-256 fingerprint
- Upserts into Postgres table `scout.impact_repository`

Usage:
  python -m pipelines.run_excel_ingest \
      --input "data\\pranjal_impact_pointers.xlsx" \
      --min-conf 0.80 \
      --top-n 10 \
      --source-url "https://themathcompany.sharepoint.com/..."
"""

import os, re, json, hashlib, argparse, asyncio
from typing import Any, Dict, List, Optional

import nest_asyncio
nest_asyncio.apply()

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from openai import AsyncAzureOpenAI
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.types import UserDefinedType

from utils.db import get_engine
from pipelines.embedding_utils import generate_embeddings, EMBEDDING_DIM

load_dotenv()

# =============================================================================
# Register pgvector type with SQLAlchemy reflection
# =============================================================================
from sqlalchemy.types import UserDefinedType

class Vector(UserDefinedType):
    def __init__(self, dim: int = None):
        self.dim = dim  # accept optional dimension

    def get_col_spec(self) -> str:
        return "vector" if self.dim is None else f"vector({self.dim})"


# =====================================================
# 🧱 Constants / ENV
# =====================================================
TABLE_SCHEMA = "scout"
TABLE_NAME = "impact_repository"
FQN = f"{TABLE_SCHEMA}.{TABLE_NAME}"

DEFAULT_MIN_CONF = float(os.getenv("MIN_CONF", "0.80"))
DEFAULT_TOP_N = int(os.getenv("TOP_N_PER_DECK", "1000"))

AZURE_ENDPOINT   = os.getenv("AZURE_ENDPOINT", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "")
AZURE_API_KEY    = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION= os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

CLASSIFY_PROMPT = """
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
""".strip()

# =====================================================
# 🔧 Helper Functions
# =====================================================
def compute_fingerprint(row: Dict[str, Any]) -> str:
    sig = "|".join([
        (row.get("impact") or "").strip().lower(),
        (row.get("industry") or "").strip().lower(),
        (row.get("business_group") or "").strip().lower(),
        (row.get("use_case") or "").strip().lower(),
        (row.get("source_file") or "").strip().lower(),
        (row.get("source_type") or "").strip().lower(),
    ])
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()

def safe_extract_json(text: str) -> Dict[str, Any]:
    try:
        match = re.search(r"\{.*\}\s*$", text.strip(), re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"Industry":[{"value":"","confidence":0}],
            "Business Group":[{"value":"","confidence":0}],
            "Use Case":[{"value":"","confidence":0}]}

def expand_classification(impact_text:str, parsed:Dict[str,Any], source_file:str)->List[Dict[str,Any]]:
    industries = parsed.get("Industry") or [{"value":"","confidence":0}]
    bgroups   = parsed.get("Business Group") or [{"value":"","confidence":0}]
    usecases  = parsed.get("Use Case") or [{"value":"","confidence":0}]
    rows=[]
    for ind in industries:
        for bg in bgroups:
            for uc in usecases:
                rows.append({
                    "source_file":source_file,
                    "impact":impact_text.strip(),
                    "industry":(ind.get("value") or "").strip(),
                    "industry_conf":float(ind.get("confidence") or 0),
                    "business_group":(bg.get("value") or "").strip(),
                    "business_group_conf":float(bg.get("confidence") or 0),
                    "use_case":(uc.get("value") or "").strip(),
                    "use_case_conf":float(uc.get("confidence") or 0),
                    "source_type":"excel",
                    "slide_no":None,
                    "embedding":None,
                    "fingerprint":""
                })
    return rows

def filter_confident(df:pd.DataFrame, min_conf:float)->pd.DataFrame:
    if df.empty: return df
    return df[
        (pd.to_numeric(df["industry_conf"], errors="coerce")>=min_conf)&
        (pd.to_numeric(df["business_group_conf"], errors="coerce")>=min_conf)&
        (pd.to_numeric(df["use_case_conf"], errors="coerce")>=min_conf)
    ].copy()

# =====================================================
# 🤖 Azure Client + Classifier
# =====================================================
def build_client()->AsyncAzureOpenAI:
    if not (AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_KEY):
        raise RuntimeError("Missing Azure OpenAI credentials.")
    return AsyncAzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        timeout=60.0
    )

async def classify_one(client:AsyncAzureOpenAI, impact:str, retries:int=3)->Dict[str,Any]:
    prompt = CLASSIFY_PROMPT.format(impact_text=impact)
    for i in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                temperature=0,
                messages=[
                    {"role":"system","content":"Return valid JSON only."},
                    {"role":"user","content":prompt}
                ]
            )
            reply=(resp.choices[0].message.content or "").strip()
            return safe_extract_json(reply)
        except Exception as e:
            if i==retries-1:
                return {"Industry":[{"value":"","confidence":0}],
                        "Business Group":[{"value":"","confidence":0}],
                        "Use Case":[{"value":"","confidence":0}]}
            await asyncio.sleep(2**i)

async def classify_all(client:AsyncAzureOpenAI, impacts:List[str], concurrency:int=8)->List[Dict[str,Any]]:
    sem=asyncio.Semaphore(concurrency)
    async def _task(txt:str):
        async with sem:
            return await classify_one(client,txt)
    return await asyncio.gather(*[_task(x) for x in impacts])

# =====================================================
# 🗃️ Database Ops
# =====================================================
DDL_ADD_VECTOR=f"ALTER TABLE {FQN} ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIM});"

def ensure_table(engine)->None:
    with engine.begin() as conn:
        conn.execute(text(DDL_ADD_VECTOR))

def upsert(engine, df:pd.DataFrame)->int:
    if df.empty: return 0
    metadata=MetaData(schema=TABLE_SCHEMA)
    table=Table(TABLE_NAME, metadata, autoload_with=engine)
    valid_cols=set(table.columns.keys())
    df=df[[c for c in df.columns if c in valid_cols]].copy()
    with engine.begin() as conn:
        stmt=pg_insert(table).values(df.to_dict(orient="records"))
        stmt=stmt.on_conflict_do_nothing(index_elements=["fingerprint"])
        res=conn.execute(stmt)
        return res.rowcount or 0

# =====================================================
# 🚀 Main
# =====================================================
async def main(input_excel:str, source_url:str, min_conf:float, top_n:int, sheet:str=None):
    if not os.path.exists(input_excel):
        raise FileNotFoundError(input_excel)
    df_in=pd.read_excel(input_excel, sheet_name=sheet) if sheet else pd.read_excel(input_excel)
    if "Impact" not in df_in.columns:
        raise ValueError("Excel must have 'Impact' column.")

    impacts=[str(x).strip() for x in df_in["Impact"].fillna("")]
    impacts=[x for x in impacts if x]
    if not impacts:
        print("ℹ️ No non-empty rows.")
        return

    client=build_client()
    print(f"🔤 Classifying {len(impacts)} impact statements...")
    results=await classify_all(client, impacts, concurrency=8)

    source_file=source_url or os.path.basename(input_excel)
    rows=[]
    for txt,parsed in zip(impacts,results):
        rows.extend(expand_classification(txt,parsed,source_file))
    df=pd.DataFrame(rows)
    if df.empty:
        print("ℹ️ No classified rows.")
        return

    df=filter_confident(df,min_conf)
    if df.empty:
        print("ℹ️ Nothing meets confidence threshold.")
        return

    df["fingerprint"]=df.apply(lambda r:compute_fingerprint(r.to_dict()),axis=1)
    df=df.drop_duplicates(subset=["fingerprint"]).copy()

    # ---- Embeddings ----
    embed_texts=(df["impact"].fillna("")+" | "+df["use_case"].fillna("")+
                 " | "+df["industry"].fillna("")+" | "+df["business_group"].fillna("")).tolist()
    print(f"🧠 Generating embeddings for {len(embed_texts)} rows (dim={EMBEDDING_DIM})...")
    vectors=await generate_embeddings(embed_texts)
    if len(vectors)!=len(df):
        print("⚠️ Warning: embedding count mismatch; skipping bad rows.")
    df["embedding"]=vectors

    # ---- DB Insert ----
    engine=get_engine()
    ensure_table(engine)
    inserted=upsert(engine,df)

    print("\n================== Summary ==================")
    print(f"Source file   : {source_file}")
    print(f"Rows classified: {len(rows)}")
    print(f"Rows kept      : {len(df)} (after confidence & dedupe)")
    print(f"Inserted       : {inserted}")
    print(f"Target table   : {FQN}")
    print("=============================================\n")

# =====================================================
# CLI
# =====================================================
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--source-url",default=None)
    p.add_argument("--sheet",default=None)
    p.add_argument("--min-conf",type=float,default=DEFAULT_MIN_CONF)
    p.add_argument("--top-n",type=int,default=DEFAULT_TOP_N)
    a=p.parse_args()
    asyncio.run(main(a.input,a.source_url,a.min_conf,a.top_n,a.sheet))
