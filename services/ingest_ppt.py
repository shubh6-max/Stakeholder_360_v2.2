# services/ingest_ppt.py
import pandas as pd
from utils.db import get_engine
from services.impact_repo import ensure_tables, upsert_impact_df

# Expect: df_expanded or filtered_df from your PPT extractor/classifier
def ingest_ppt_df(
    df: pd.DataFrame,
    *,
    source_file_name: str,  # e.g., "Impact Story Boardroom_BNSF_2025.07.02.pptx"
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
        source_type="ppt",
        source_file=source_file_name,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        embed_deployment=embed_deployment,
        embed_api_version=embed_api_version,
    )
