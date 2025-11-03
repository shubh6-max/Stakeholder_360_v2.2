# services/ingest_excel.py
import pandas as pd
from utils.db import get_engine
from services.impact_repo import ensure_tables, upsert_impact_df

# Expect: df_expanded or filtered_df from your existing Excel classifier
def ingest_excel_df(
    df: pd.DataFrame,
    *,
    source_file_url: str,   # keep your SharePoint URL if you want traceability
    azure_endpoint: str,
    azure_api_key: str,
    embed_deployment: str,
    embed_api_version: str,
) -> dict:
    engine = get_engine()
    ensure_tables(engine)
    return upsert_impact_df(
        engine,
        df,
        source_type="excel",
        source_file=source_file_url,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        embed_deployment=embed_deployment,
        embed_api_version=embed_api_version,
    )
