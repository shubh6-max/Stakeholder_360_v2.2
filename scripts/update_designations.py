import os
import time
import json
import pandas as pd
import psycopg2
import psycopg2.extras
from openai import AzureOpenAI
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from dotenv import load_dotenv

# ====================================================
# 1️⃣ Load environment variables
# ====================================================
# ====================================================
# 1️⃣ Load environment variables (from GitHub secrets)
# ====================================================
import os
from dotenv import load_dotenv

load_dotenv()  # loads local .env if running manually; GitHub injects ENV automatically

PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

if not all([PG_USER, PG_PASSWORD, PG_HOST, PG_DB, AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT]):
    raise EnvironmentError("❌ Missing one or more required environment variables.")


# ====================================================
# 2️⃣ DB Engine + Connection
# ====================================================
def get_engine():
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=int(PG_PORT),
        database=PG_DB,
        query={"sslmode": "require"},
    )
    return create_engine(url, pool_size=3, max_overflow=5, echo=False)

# ====================================================
# 3️⃣ Load records needing classification
# ====================================================
def load_missing_roles():
    cols = ["email_address", "client_name", "client_designation", "working_group", "business_unit", "seniority_level"]
    query = f"""
        SELECT {', '.join(cols)}
        FROM scout.centralize_db
        WHERE (business_unit IS NULL OR business_unit='NaN')
          AND (seniority_level IS NULL OR seniority_level='NaN');
    """
    with get_engine().begin() as conn:
        df = pd.read_sql(query, conn)
    return df

# ====================================================
# 4️⃣ Azure OpenAI setup
# ====================================================
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

import re

def extract_role_type(title: str) -> dict:
    PROMPT = f'''
You are an expert in classifying corporate job titles.

Given the title: "{title}", classify it into the following 3 fields:

1. "Working Group": choose from ["Business", "Business Analytics", "Business IT", "Central Analytics", "IT", "Others"]
2. "Business Unit": choose from ["Central Analytics", "Finance", "HR", "IT", "Manufacturing", "Marketing", "Operations", "R&D", "Sales", "Strategy", "Supply Chain", "Others"]
3. "Seniority Level": choose from ["Associate Director", "Director", "Director -", "Director +", "Senior Manager", "VP", "VP +", "Other"]

Return only a *valid JSON object*. No markdown, no explanation.
Example:
{{
  "working_group": "...",
  "business_unit": "...",
  "seniority_level": "..."
}}
'''

    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0
        )
        raw = resp.choices[0].message.content.strip()

        # --- Step 1: Clean any markdown fences or noise ---
        raw = re.sub(r"```(json)?", "", raw).strip()

        # --- Step 2: Try parsing JSON ---
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # --- Step 3: Try extracting JSON substring if model wrapped it ---
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                # --- Step 4: fallback defaults ---
                print(f"⚠️  Non-JSON output for '{title}': {raw[:100]}...")
                return {
                    "working_group": "Others",
                    "business_unit": "Others",
                    "seniority_level": "Other"
                }

    except Exception as e:
        print(f"❌ API error for '{title}': {e}")
        return {
            "working_group": "Others",
            "business_unit": "Others",
            "seniority_level": "Other"
        }


# ====================================================
# 5️⃣ Process rows with parallel batching (optimized)
# ====================================================
def classify_missing(df: pd.DataFrame):
    results = []
    for _, row in df[:10].iterrows():
        try:
            info = extract_role_type(row["client_designation"])
            info.update({
                "client_name": row["client_name"],
                "client_designation": row["client_designation"],
                "email_address": row["email_address"],
            })
            results.append(info)
            print(f"✅ {row['client_designation']} → {info}")
            time.sleep(1.0)
        except Exception as e:
            print(f"❌ Error on {row['client_designation']} → {e}")
    return results

# ====================================================
# 6️⃣ Batch Upsert to DB (fast version)
# ====================================================
def upsert_records(records):
    if not records:
        print("⚠️ No records to update.")
        return
    conn = psycopg2.connect(
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DB,
        sslmode="require",
    )
    cur = conn.cursor()

    update_sql = """
        UPDATE scout.centralize_db
           SET client_designation = %(client_designation)s,
               client_name        = %(client_name)s,
               working_group      = %(working_group)s,
               business_unit      = %(business_unit)s,
               seniority_level    = %(seniority_level)s
         WHERE email_address = %(email_address)s;
    """
    insert_sql = """
        INSERT INTO scout.centralize_db (
            email_address, client_designation, client_name,
            working_group, business_unit, seniority_level
        )
        SELECT %(email_address)s, %(client_designation)s, %(client_name)s,
               %(working_group)s, %(business_unit)s, %(seniority_level)s
         WHERE NOT EXISTS (
            SELECT 1 FROM scout.centralize_db WHERE email_address = %(email_address)s
        );
    """

    psycopg2.extras.execute_batch(cur, update_sql, records, page_size=100)
    psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=100)
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Upserted {len(records)} records successfully!")

# ====================================================
# 7️⃣ Full pipeline
# ====================================================
if __name__ == "__main__":
    df_missing = load_missing_roles()
    print(f"🔍 Found {len(df_missing)} records missing classification.")
    if not df_missing.empty:
        classified = classify_missing(df_missing)
        upsert_records(classified)
    else:
        print("✅ No missing records found.")
