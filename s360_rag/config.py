# s360_rag/config.py
import os
from typing import Dict, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

# ---------- helpers ----------

def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

def _get_float(name: str, default: float, min_v: float | None = None, max_v: float | None = None) -> float:
    raw = os.getenv(name)
    v = float(raw) if raw not in (None, "") else default
    if min_v is not None and v < min_v:
        raise ValueError(f"{name}={v} < min {min_v}")
    if max_v is not None and v > max_v:
        raise ValueError(f"{name}={v} > max {max_v}")
    return v

def _get_int(name: str, default: int, min_v: int | None = None, max_v: int | None = None) -> int:
    raw = os.getenv(name)
    v = int(raw) if raw not in (None, "") else default
    if min_v is not None and v < min_v:
        raise ValueError(f"{name}={v} < min {min_v}")
    if max_v is not None and v > max_v:
        raise ValueError(f"{name}={v} > max {max_v}")
    return v


# ---------- Azure ----------

AZURE_ENDPOINT          = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY           = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION       = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT        = os.getenv("AZURE_DEPLOYMENT", "gpt-4-vision-preview")
AZURE_EMBED_DEPLOYMENT  = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")

# ---------- App/debug ----------

APP_ENV     = os.getenv("APP_ENV", "dev").lower()      # dev | prod
DEBUG_RAG   = _get_bool("DEBUG_RAG", default=(APP_ENV != "prod"))

# ---------- RAG core ----------

RAG_EMBED_DIM      = _get_int("RAG_EMBED_DIM", 1536, min_v=1)
RAG_CHUNK_TOKENS   = _get_int("RAG_CHUNK_TOKENS", 2000, min_v=100)
RAG_CHUNK_OVERLAP  = _get_int("RAG_CHUNK_OVERLAP", 150, min_v=0)
TOP_K              = _get_int("RAG_TOP_K", 4, min_v=1, max_v=50)

# Primary similarity threshold (we treat it as cosine **similarity**)
# If DB returns cosine distance (0=identical), we will convert: sim = 1 - dist
SIM_THRESHOLD      = _get_float("RAG_SIM_THRESHOLD", 0.35, min_v=0.0, max_v=1.0)

# ---------- Strictness gates (optional) ----------

# Require at least N KPI hits in the winning chunk
RAG_REQUIRE_KPI_HIT   = _get_bool("RAG_REQUIRE_KPI_HIT", False)
RAG_KPI_MATCH_MIN     = _get_int("RAG_KPI_MATCH_MIN", 1, min_v=1, max_v=10)

# Require filename/slide-derived function/industry alignment
RAG_REQUIRE_FN_ALIGN      = _get_bool("RAG_REQUIRE_FN_ALIGN", False)
RAG_REQUIRE_INDUSTRY_ALIGN= _get_bool("RAG_REQUIRE_INDUSTRY_ALIGN", False)

# Require at least N impact pointers to accept a match
RAG_MIN_IMPACT_POINTERS   = _get_int("RAG_MIN_IMPACT_POINTERS", 2, min_v=1, max_v=10)

# Optional LLM re-rank confidence margin
RAG_RERANK_TOP_M      = _get_int("RAG_RERANK_TOP_M", 3, min_v=1, max_v=10)
RAG_RERANK_MIN_GAP    = _get_float("RAG_RERANK_MIN_GAP", 0.08, min_v=0.0, max_v=1.0)

# Output JSON strictness (reject if schema invalid)
RAG_STRICT_JSON       = _get_bool("RAG_STRICT_JSON", True)

# ---------- PPT source ----------

PPT_FOLDER = os.getenv("PPT_FOLDER", "./data/case_studies")

# ---------- DB (Postgres) ----------

PGHOST      = os.getenv("PGHOST", "")
PGDATABASE  = os.getenv("PGDATABASE", "")
PGUSER      = os.getenv("PGUSER", "")
PGPASSWORD  = os.getenv("PGPASSWORD", "")
PGPORT      = os.getenv("PGPORT", "5432")
PGSSL       = os.getenv("PGSSL", "require")  # Azure requires SSL by default

# Guard: common Azure mistake (username accidentally in host)
if "@" in PGHOST:
    raise RuntimeError(
        "PGHOST contains '@'. Use plain host like 'servername.postgres.database.azure.com'. "
        "Put the username only in PGUSER (no '@servername' needed for Flexible Server)."
    )

PG_CONN_STR = (
    f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}?sslmode={PGSSL}"
    if all([PGHOST, PGDATABASE, PGUSER, PGPASSWORD, PGPORT])
    else ""
)

# ---------- Schema / Table names (single source of truth) ----------

SCHEMA              = os.getenv("RAG_SCHEMA", "scout")
TBL_CS_META         = os.getenv("RAG_TBL_CS_META", "cs_meta")
TBL_CS_CHUNKS       = os.getenv("RAG_TBL_CS_CHUNKS", "cs_chunks")
TBL_RAG_QUERY_LOGS  = os.getenv("RAG_TBL_QUERY_LOGS", "rag_query_logs")
TBL_PERSONA_KPIS    = os.getenv("RAG_TBL_PERSONA_KPIS", "persona_kpis_store")

# Fully qualified names
FQN_CS_META         = f"{SCHEMA}.{TBL_CS_META}"
FQN_CS_CHUNKS       = f"{SCHEMA}.{TBL_CS_CHUNKS}"
FQN_RAG_QUERY_LOGS  = f"{SCHEMA}.{TBL_RAG_QUERY_LOGS}"
FQN_PERSONA_KPIS    = f"{SCHEMA}.{TBL_PERSONA_KPIS}"


# ---------- utilities ----------

def assert_required() -> None:
    """Fail fast for required secrets and infra."""
    missing = []
    required: Dict[str, Any] = dict(
        AZURE_ENDPOINT=AZURE_ENDPOINT,
        AZURE_API_KEY=AZURE_API_KEY,
        AZURE_API_VERSION=AZURE_API_VERSION,
        AZURE_DEPLOYMENT=AZURE_DEPLOYMENT,
        AZURE_EMBED_DEPLOYMENT=AZURE_EMBED_DEPLOYMENT,
        PG_CONN_STR=PG_CONN_STR,
    )
    for k, v in required.items():
        if not v:
            missing.append(k)
    if missing:
        raise RuntimeError(f"Missing required env: {', '.join(missing)}")

def is_dev() -> bool:
    return APP_ENV != "prod"

@dataclass
class _SafeConfig:
    app_env: str
    debug_rag: bool
    rag_embed_dim: int
    rag_chunk_tokens: int
    rag_chunk_overlap: int
    top_k: int
    sim_threshold: float
    require_kpi_hit: bool
    kpi_match_min: int
    require_fn_align: bool
    require_industry_align: bool
    min_impact_pointers: int
    rerank_top_m: int
    rerank_min_gap: float
    strict_json: bool
    ppt_folder: str
    schema: str
    tbl_cs_meta: str
    tbl_cs_chunks: str
    tbl_rag_query_logs: str
    tbl_persona_kpis: str

def safe_asdict() -> Dict[str, Any]:
    """Return a redacted snapshot for logging/debug (no secrets)."""
    cfg = _SafeConfig(
        app_env=APP_ENV,
        debug_rag=DEBUG_RAG,
        rag_embed_dim=RAG_EMBED_DIM,
        rag_chunk_tokens=RAG_CHUNK_TOKENS,
        rag_chunk_overlap=RAG_CHUNK_OVERLAP,
        top_k=TOP_K,
        sim_threshold=SIM_THRESHOLD,
        require_kpi_hit=RAG_REQUIRE_KPI_HIT,
        kpi_match_min=RAG_KPI_MATCH_MIN,
        require_fn_align=RAG_REQUIRE_FN_ALIGN,
        require_industry_align=RAG_REQUIRE_INDUSTRY_ALIGN,
        min_impact_pointers=RAG_MIN_IMPACT_POINTERS,
        rerank_top_m=RAG_RERANK_TOP_M,
        rerank_min_gap=RAG_RERANK_MIN_GAP,
        strict_json=RAG_STRICT_JSON,
        ppt_folder=PPT_FOLDER,
        schema=SCHEMA,
        tbl_cs_meta=TBL_CS_META,
        tbl_cs_chunks=TBL_CS_CHUNKS,
        tbl_rag_query_logs=TBL_RAG_QUERY_LOGS,
        tbl_persona_kpis=TBL_PERSONA_KPIS,
    )
    return asdict(cfg)
