"""
linkedin_url_finder.py
-----------------------------------------------------
Finds the most likely LinkedIn URL for each persona
using Tavily API and Azure OpenAI, and updates Azure PostgreSQL.

✅ Sequential (API-safe)
✅ Tavily key state stored in Postgres (scout_v2.tavily_key_state)
✅ Auto-rotates keys + auto-disables when <= MIN_CREDITS_LEFT
✅ Uses AOAI to extract URL from Tavily "answer text" (not just regex)
✅ Uses AOAI to validate profile URL (YES/NO)
✅ Updates scout_v2.master_ldb by id (safe)

Author: Shubham Vishwas Puranik
"""

import os
import time
from time import sleep
from typing import Optional, Dict, Any, Tuple

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from tavily import TavilyClient
from openai import AzureOpenAI

# ======================================================
# 🔧 CONFIGURATION
# ======================================================
load_dotenv()

# Postgres
PG_USER = os.getenv("PGUSER", "")
PG_PASSWORD = os.getenv("PGPASSWORD", "")
PG_HOST = os.getenv("PGHOST", "")
PG_PORT = os.getenv("PGPORT", "5432")
PG_DB = os.getenv("PG_DB", "stakeholder360_v2")  # change if needed

# Azure OpenAI
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4-vision-preview")
# AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")


# Behavior
MIN_CREDITS_LEFT = int(os.getenv("TAVILY_MIN_CREDITS_LEFT", "10"))
BATCH_LIMIT = int(os.getenv("BATCH_LIMIT", "1000"))
API_DELAY_S = float(os.getenv("API_DELAY_S", "1.0"))  # delay between persons

# Where keys are stored
KEY_TABLE = "scout_v2.tavily_key_state"

# ======================================================
# 🔌 AOAI CLIENT
# ======================================================
aoai_client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

# ======================================================
# 🧠 DATABASE CONNECTION
# ======================================================
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
    return create_engine(url, pool_pre_ping=True)

# ======================================================
# 🔑 DB KEY MANAGER (ROTATION + AUTO-DISABLE)
# ======================================================
class TavilyKeyManagerDB:
    """
    Key selection + credit decrement stored in Postgres table:
      scout_v2.tavily_key_state(key_name, api_key, credits_remaining, is_active, ...)

    Concurrency-safe (best effort):
      - acquires a row lock with FOR UPDATE SKIP LOCKED inside a transaction
      - returns key for use in Tavily request
      - on success: decrements credits_remaining and auto-disables if <= MIN_CREDITS_LEFT
    """

    def __init__(self, engine, min_left: int = 10):
        self.engine = engine
        self.min_left = min_left

    def acquire_key(self) -> Tuple[str, str]:
        """
        Returns (key_name, api_key) for an active key with credits_remaining > min_left.
        Locks the selected row for the duration of the transaction (released immediately after commit).
        """
        pick_sql = text(f"""
            SELECT key_name, api_key
            FROM {KEY_TABLE}
            WHERE is_active = TRUE
              AND credits_remaining > :min_left
            ORDER BY credits_remaining DESC, updated_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1;
        """)

        with self.engine.begin() as conn:
            row = conn.execute(pick_sql, {"min_left": self.min_left}).mappings().first()

        if not row:
            raise RuntimeError(
                f"No active Tavily keys available (all keys have <= {self.min_left} credits remaining or inactive)."
            )

        return row["key_name"], row["api_key"]

    def consume_credit(self, key_name: str) -> Dict[str, Any]:
        """
        Decrement credits_remaining by 1, auto-disable if now <= min_left.
        Returns updated row details (credits_remaining, is_active).
        """
        upd_sql = text(f"""
            UPDATE {KEY_TABLE}
            SET
                credits_remaining = credits_remaining - 1,
                is_active = CASE
                    WHEN (credits_remaining - 1) <= :min_left THEN FALSE
                    ELSE is_active
                END
            WHERE key_name = :key_name
            RETURNING key_name, credits_remaining, is_active;
        """)
        with self.engine.begin() as conn:
            row = conn.execute(upd_sql, {"key_name": key_name, "min_left": self.min_left}).mappings().first()

        if not row:
            raise RuntimeError(f"Failed to consume credit: key not found: {key_name}")

        return dict(row)

    def disable_key(self, key_name: str):
        """Manually disable a key (e.g., invalid/blocked)."""
        dis_sql = text(f"""
            UPDATE {KEY_TABLE}
            SET is_active = FALSE
            WHERE key_name = :key_name;
        """)
        with self.engine.begin() as conn:
            conn.execute(dis_sql, {"key_name": key_name})

    def summary(self) -> str:
        sql = text(f"""
            SELECT
              COUNT(*) FILTER (WHERE is_active = TRUE AND credits_remaining > :min_left) AS active_usable,
              COUNT(*) FILTER (WHERE is_active = TRUE) AS active_total,
              COUNT(*) FILTER (WHERE is_active = FALSE) AS inactive_total,
              COALESCE(MAX(credits_remaining) FILTER (WHERE is_active = TRUE), 0) AS max_active_remaining
            FROM {KEY_TABLE};
        """)
        with self.engine.begin() as conn:
            row = conn.execute(sql, {"min_left": self.min_left}).mappings().first()

        return (
            f"Keys usable(> {self.min_left}): {row['active_usable']} | "
            f"active: {row['active_total']} | inactive: {row['inactive_total']} | "
            f"max_active_remaining: {row['max_active_remaining']}"
        )

# ======================================================
# 🧠 AOAI: Extract URL from Tavily answer TEXT
# ======================================================
def aoai_extract_linkedin_profile_url(text_blob: str) -> Optional[str]:
    """
    Extract ONLY a LinkedIn personal profile URL from noisy text.
    Return the URL or None.
    """
    if not text_blob:
        return None

    prompt = f"""
Extract ONLY the LinkedIn personal profile URL from the text.

Rules:
- Output ONLY ONE URL if present and it must be a profile like: https://www.linkedin.com/in/...
- If none found, output ONLY: NONE
- Do NOT output company pages (/company/), posts (/feed/ /posts/), articles, jobs, etc.

Text:
{text_blob}
""".strip()

    try:
        resp = aoai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        out = (resp.choices[0].message.content or "").strip()

        if out.upper() == "NONE":
            return None

        out = out.strip().rstrip(").,]}")
        if "linkedin.com/in/" not in out:
            return None

        # normalize (optional): ensure https + www
        out = out.replace("http://", "https://")
        if out.startswith("https://linkedin.com/"):
            out = out.replace("https://linkedin.com/", "https://www.linkedin.com/")

        return out
    except Exception as e:
        print(f"⚠️ AOAI extract failed: {e}")
        return None

# ======================================================
# 🧠 AOAI: Validate profile URL (YES/NO)
# ======================================================
def verify_linkedin_url_with_aoai(url: str) -> bool:
    if not url:
        return False

    prompt = f"""
You are a strict LinkedIn URL validator.

Rules:
- If it is a company page, article, PDF, post, job, or generic page → NO
- If it is a real person's LinkedIn profile (linkedin.com/in/...) → YES
Output ONLY YES or NO.

URL: {url}
""".strip()

    try:
        resp = aoai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans == "yes"
    except Exception as e:
        print(f"⚠️ AOAI validation failed for {url}: {e}")
        return False

# ======================================================
# 📥 LOAD DATA (ONLY NULL linkedin_url)
# ======================================================
def load_master_df(engine, limit: int = 1000) -> pd.DataFrame:
    sql = text("""
        SELECT id, client_name, account
        FROM scout_v2.master_ldb
        WHERE linkedin_url IS NULL
        ORDER BY id
        LIMIT :limit;
    """)
    with engine.begin() as conn:
        df = pd.read_sql(sql, conn, params={"limit": limit})

    print(f"✅ Loaded {len(df)} records where linkedin_url IS NULL")
    return df

# ======================================================
# 🧱 UPDATE DATABASE (SAFE BY ID)
# ======================================================
def update_linkedin_urls(engine, final_df: pd.DataFrame):
    if final_df.empty:
        print("⚠️ Nothing to update.")
        return

    update_sql = text("""
        UPDATE scout_v2.master_ldb
        SET linkedin_url = :linkedin_url
        WHERE id = :id;
    """)

    updated = 0
    with engine.begin() as conn:
        for row in final_df.itertuples(index=False):
            conn.execute(update_sql, {"id": row.id, "linkedin_url": row.linkedin_url})
            updated += 1

    print(f"✅ Database updated for {updated} personas (by id)")

# ======================================================
# 🔍 FETCH LINKEDIN URL USING TAVILY + DB KEY ROTATION
# ======================================================
def fetch_linkedin_urls(engine, df: pd.DataFrame, key_mgr: TavilyKeyManagerDB) -> pd.DataFrame:
    results = []
    total = len(df)

    print(f"🔍 Tavily lookup started for {total} personas...")
    print("🔑", key_mgr.summary())

    for i, row in enumerate(df.itertuples(index=False), start=1):
        _id = row.id
        name = (row.client_name or "").strip()
        company = (row.account or "").strip()

        if not name:
            results.append({"id": _id, "persona_name": name, "company_name": company, "linkedin_url": None})
            print(f"[{i}/{total}] id={_id} → skipped (missing client_name)")
            continue

        query = f'Answer only with LinkedIn Profile URL. Person Name: "{name}" Company: "{company}"'
        linkedin_url: Optional[str] = None

        # up to 3 attempts (key rotation happens by re-acquire)
        for attempt in range(1, 4):
            key_name = None
            try:
                key_name, api_key = key_mgr.acquire_key()
                tavily_client = TavilyClient(api_key=api_key)

                resp = tavily_client.search(
                    query=query,
                    search_depth="basic",
                    max_results=3,
                    include_domains=["linkedin.com"],
                    include_answer="advanced",
                )

                # ✅ 1 Tavily call = 1 credit (decrement immediately after successful API call)
                state = key_mgr.consume_credit(key_name)

                # 1) Use AOAI to extract URL from Tavily "answer" text
                answer_text = resp.get("answer") or ""
                candidate_url = aoai_extract_linkedin_profile_url(answer_text)

                # 2) Fallback: sometimes answer is empty, check results urls too
                if not candidate_url:
                    for r in (resp.get("results") or []):
                        u = (r.get("url") or "")
                        candidate_url = aoai_extract_linkedin_profile_url(u)
                        if candidate_url:
                            break

                # 3) Validate via AOAI
                if candidate_url and verify_linkedin_url_with_aoai(candidate_url):
                    linkedin_url = candidate_url
                else:
                    linkedin_url = None

                # log key status
                if state["is_active"] is False:
                    print(f"🟠 Key auto-disabled (<= {key_mgr.min_left} left): {key_name} | remaining={state['credits_remaining']}")

                break  # done for this person

            except Exception as e:
                msg = str(e).lower()
                print(f"❌ Error (attempt {attempt}) id={_id} {name}: {e}")

                # If Tavily key is invalid/blocked/rate-limited, disable it
                if key_name and (("429" in msg) or ("rate" in msg) or ("unauthorized" in msg) or ("invalid" in msg)):
                    print(f"🛑 Disabling key due to error: {key_name}")
                    try:
                        key_mgr.disable_key(key_name)
                    except Exception as ee:
                        print(f"⚠️ Failed to disable key {key_name}: {ee}")

                # backoff before retry
                sleep(min(2.0 * attempt, 6.0))

        print(f"[{i}/{total}] id={_id} {name} → {linkedin_url or 'Not found'}")
        results.append({
            "id": _id,
            "persona_name": name,
            "company_name": company,
            "linkedin_url": linkedin_url
        })

        sleep(API_DELAY_S)

    print("🔑", key_mgr.summary())
    return pd.DataFrame(results)

# ======================================================
# ⚡ MAIN
# ======================================================
def main():
    start = time.time()

    engine = get_engine()
    key_mgr = TavilyKeyManagerDB(engine=engine, min_left=MIN_CREDITS_LEFT)

    print("🔑 Initial:", key_mgr.summary())

    df = load_master_df(engine, limit=BATCH_LIMIT)
    if df.empty:
        print("✅ No pending records found.")
        return df

    final_df = fetch_linkedin_urls(engine, df, key_mgr)
    update_linkedin_urls(engine, final_df)

    print(f"⏱️ Total runtime: {time.time() - start:.2f}s")
    return final_df

if __name__ == "__main__":
    out = main()
    print("\n🔎 Preview:")
    print(out.head(10))
