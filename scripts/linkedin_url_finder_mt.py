"""
linkedin_url_finder.py
-----------------------------------------------------
Finds the most likely LinkedIn URL for each persona
using Tavily API and stores results in a DataFrame.

✅ Sequential (no multithreading) — API-safe and reliable
✅ Adds 1-second delay between requests
✅ Handles Tavily rate limits automatically
✅ Updates PostgreSQL (Azure) safely
✅ Works in Google Colab and GitHub Actions

Author: Shubham Vishwas Puranik
"""

import os
import sys
import time
import random
import pandas as pd
from time import sleep
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from openai import AzureOpenAI

# ======================================================
# 🔧 CONFIGURATION
# ======================================================
load_dotenv()

PG_USER = os.getenv("PGUSER")
PG_PASSWORD = os.getenv("PGPASSWORD")
PG_HOST = os.getenv("PGHOST")
PG_PORT = os.getenv("PGPORT")
PG_DB = os.getenv("PGDATABASE")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# === Azure OpenAI Settings ===
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
AZURE_API_VERSION = "2024-02-15-preview"

aoai_client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

def verify_linkedin_url_with_aoai(url: str) -> bool:
    """
    Uses Azure OpenAI to verify whether a URL is a real LinkedIn profile URL.
    Returns True only if URL points to an actual person profile.
    Otherwise returns False.
    """

    if url is None:
        return False

    prompt = f"""
    You are a strict LinkedIn URL validator.Check if the following URL is a REAL LinkedIn PERSON PROFILE:
    URL: {url}

    example:
    - https://www.linkedin.com/in/marwa-abouawad-m-sc-0b368b57  → "YES"
    - https://www.linkedin.com/in/preethi-gudla  → "YES"
    
    Rules:
  - If it is a company page, article, PDF, generic search page → return "NO".
  - If it is a real person’s LinkedIn profile → return "YES".
  - Output MUST be only YES or NO.
    """

    try:
        response = aoai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )

        answer = response.choices[0].message.content.strip().upper()

        return answer == "YES"

    except Exception as e:
        print(f"⚠️ AOAI verification failed for {url}: {e}")
        return False

# ======================================================
# 🧠 DATABASE CONNECTION
# ======================================================
def get_engine():
    """Create SQLAlchemy engine for Azure PostgreSQL"""
    if not all([PG_USER, PG_PASSWORD, PG_HOST, PG_DB]):
        raise RuntimeError("❌ Database configuration incomplete. Check .env values.")

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=int(PG_PORT),
        database=PG_DB,
        query={"sslmode": "require"},  # Azure Postgres SSL
    )

    return create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        echo=False,
    )

# ======================================================
# 📥 LOAD DATA
# ======================================================
def load_centralize_df() -> pd.DataFrame:
    """Load the 'scout.centralize_db' table into a DataFrame"""
    try:
        engine = get_engine()
        sql = "SELECT * FROM scout.centralize_db"
        with engine.begin() as conn:
            df = pd.read_sql(sql, conn)
        print(f"✅ Loaded data successfully. Rows: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        raise

import requests

# ======================================================
# 🔍 FETCH LINKEDIN URL USING JINA.AI
# ======================================================
def fetch_linkedin_urls(df: pd.DataFrame, limit: int = 1000) -> pd.DataFrame:
    df = df.copy()
    df = df[df["linkedin_url"].isna() | (df["linkedin_url"] == "NaN")]

    if df.empty:
        print("⚠️ No records with missing LinkedIn URLs found.")
        return pd.DataFrame()

    df = df.head(limit)

    print(f"🔍 Fetching LinkedIn URLs for {len(df)} personas sequentially...")
    results = []
    start_time = time.time()

    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Respond-With": "no-content"
    }

    for idx, (company_name, persona_name) in enumerate(zip(df["account"], df["client_name"]), start=1):

        # -------------------
        # 1️⃣ Build Jina query
        # -------------------
        query = f"LinkedIn profile URL for the persona {persona_name} only from {company_name} website linkedin.com."
        jina_url = f"https://s.jina.ai/?q={requests.utils.quote(query)}"

        linkedin_url = "Not found"

        try:
            # -------------------
            # 2️⃣ Call Jina Search
            # -------------------
            response = requests.get(jina_url, headers=headers, timeout=15)

            time.sleep(2)

            if response.status_code == 200:
                data = response.json().get("data", [])
                if len(data) > 0:
                    first_url = data[0].get("url")

                    # -------------------------------
                    # 3️⃣ Verify using Azure OpenAI
                    # -------------------------------
                    is_valid = verify_linkedin_url_with_aoai(first_url)

                    if is_valid:
                        linkedin_url = first_url
                    else:
                        linkedin_url = "Not found"

            else:
                print(f"❌ Jina HTTP error {response.status_code} for {persona_name}")

        except Exception as e:
            print(f"❌ Error fetching Jina result for {persona_name}: {e}")

        print(f"[{idx}/{len(df)}] {persona_name} → {linkedin_url}")

        results.append(
            {
                "company_name": company_name,
                "persona_name": persona_name,
                "linkedin_url": linkedin_url,
            }
        )

        sleep(1)  # safety delay

    elapsed = time.time() - start_time
    print(f"✅ Completed {len(results)} lookups in {elapsed:.2f}s")
    return pd.json_normalize(results)



# ======================================================
# 🧱 UPDATE DATABASE
# ======================================================
def update_linkedin_urls(final_df: pd.DataFrame):
    """Update linkedin_url column in scout.centralize_db"""
    if final_df.empty:
        print("⚠️ No records to update.")
        return

    engine = get_engine()
    updated_count = 0
    print("🧩 Starting database update process...")

    update_sql = text("""
        UPDATE scout.centralize_db
        SET linkedin_url = :linkedin_url
        WHERE client_name = :persona_name
          AND account = :company_name;
    """)

    with engine.begin() as conn:
        for idx, row in final_df.iterrows():
            try:
                params = {
                    "linkedin_url": row["linkedin_url"],
                    "persona_name": row["persona_name"],
                    "company_name": row["company_name"],
                }
                conn.execute(update_sql, params)
                updated_count += 1
                if idx % 50 == 0:
                    print(f"🟢 Updated {idx} records so far...")
            except Exception as e:
                print(f"❌ Failed to update ({row['persona_name']}, {row['company_name']}): {e}")

    print(f"✅ Database update complete. Total updated: {updated_count} records.")

# ======================================================
# ⚡ MAIN
# ======================================================
def main():
    start_time = time.time()
    df = load_centralize_df()
    final_df = fetch_linkedin_urls(df, limit=1000)
    print(f"Final results: {final_df.shape}")

    # 🧱 Update DB
    update_linkedin_urls(final_df)

    elapsed = time.time() - start_time
    print(f"⏱️ Total time: {elapsed:.2f}s")
    return final_df

# ======================================================
# 🧩 ENTRY POINT
# ======================================================
if __name__ == "__main__":
    final_persona_linkedin_url_df = main()
    print("\n✅ Final Output Preview:")
    print(final_persona_linkedin_url_df.head())
