# features/insights/vectorstore.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import inspect

from sqlalchemy import text
from sqlalchemy.engine import Engine

# Prefer split package; fallback to community if needed
try:
    from langchain_postgres import PGVector
    _pgvector_src = "langchain_postgres"
except Exception:
    from langchain_community.vectorstores import PGVector
    _pgvector_src = "langchain_community"

# Cross-version DistanceStrategy import
try:
    from langchain_community.vectorstores.utils import DistanceStrategy
except Exception:
    try:
        from langchain_core.vectorstores import DistanceStrategy  # type: ignore
    except Exception:
        from enum import Enum
        class DistanceStrategy(str, Enum):  # shim for type hints
            EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
            MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
            COSINE = "COSINE"

from .config import (
    get_settings,
    make_embeddings,
    pg_conn_str_from_engine,
    company_collection_name,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _opclass_for(distance: str) -> str:
    d = (distance or "cosine").lower()
    if d == "l2":
        return "vector_l2_ops"
    if d in ("ip", "inner", "inner_product"):
        return "vector_ip_ops"
    return "vector_cosine_ops"

def _distance_str(distance: str) -> str:
    """Normalize to the string values PGVector expects in some versions."""
    d = (distance or "cosine").lower()
    if d in ("l2", "euclidean", "euclid"):
        return "l2"
    if d in ("ip", "inner", "inner_product", "max_inner_product"):
        return "inner"
    return "cosine"

def _distance_enum(distance: str) -> DistanceStrategy:
    """Enum for newer builds that accept DistanceStrategy."""
    d = (distance or "cosine").lower()
    if d == "l2":
        return DistanceStrategy.EUCLIDEAN_DISTANCE  # type: ignore[attr-defined]
    if d in ("ip", "inner", "inner_product"):
        return DistanceStrategy.MAX_INNER_PRODUCT  # type: ignore[attr-defined]
    return DistanceStrategy.COSINE  # type: ignore[attr-defined]

def ensure_pgvector_bootstrap(engine: Engine, schema: str) -> None:
    """CREATE EXTENSION vector; CREATE SCHEMA <schema> if missing."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))

def ensure_ivfflat_index(engine: Engine, schema: str, distance: str, table_name: str) -> None:
    """Add an IVFFLAT index once the embedding table exists."""
    opclass = _opclass_for(distance)
    idx_name = f"idx_{schema}_{table_name}_ivf"
    ddl = f"""
    DO $$
    BEGIN
      IF EXISTS (
        SELECT 1
          FROM information_schema.tables
         WHERE table_schema='{schema}' AND table_name='{table_name}'
      ) THEN
        IF NOT EXISTS (
          SELECT 1 FROM pg_indexes
           WHERE schemaname='{schema}' AND indexname='{idx_name}'
        ) THEN
          EXECUTE 'CREATE INDEX {idx_name}
                     ON {schema}.{table_name} USING ivfflat (embedding {opclass})
                     WITH (lists = 100)';
        END IF;
      END IF;
    END $$;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

# ──────────────────────────────────────────────────────────────────────────────
# Version-agnostic PGVector builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_pgvector(
    conn_str: str,
    embeddings_obj,
    collection_name: str,
    distance_cfg: str,
    prefer_schema: str = "rag",
) -> Tuple[PGVector, str, str]:
    """
    Construct PGVector across many versions by inspecting __init__ and
    trying distance as string first (distance="cosine"), then fallbacks.
    Returns (store, schema_used, emb_table_name_without_schema).
    """
    sig = inspect.signature(PGVector.__init__)
    params = set(sig.parameters.keys())
    params.discard("self")

    # Base kwargs
    kwargs_base: Dict[str, Any] = {}

    if "connection" in params:
        kwargs_base["connection"] = conn_str
    elif "connection_string" in params:
        kwargs_base["connection_string"] = conn_str
    else:
        raise RuntimeError("PGVector.__init__ has no 'connection' or 'connection_string'")

    # Embeddings param variants
    if "embeddings" in params:
        kwargs_base["embeddings"] = embeddings_obj
    elif "embedding_function" in params:
        kwargs_base["embedding_function"] = embeddings_obj
    elif "embedding" in params:
        kwargs_base["embedding"] = embeddings_obj
    else:
        raise RuntimeError("PGVector.__init__ has neither 'embeddings' nor 'embedding_function'")

    if "collection_name" in params:
        kwargs_base["collection_name"] = collection_name
    if "use_jsonb" in params:
        kwargs_base["use_jsonb"] = True
    if "pre_delete_collection" in params:
        kwargs_base["pre_delete_collection"] = False

    # Schema / table handling
    schema_used = "public"
    emb_table = "langchain_pg_embedding"
    coll_table = "langchain_pg_collection"

    if "schema_name" in params:
        schema_used = prefer_schema
        schema_kwargs = {"schema_name": prefer_schema}
    elif "table_name" in params and "collection_table_name" in params:
        schema_used = prefer_schema
        schema_kwargs = {
            "table_name": f"{prefer_schema}.{emb_table}",
            "collection_table_name": f"{prefer_schema}.{coll_table}",
        }
    else:
        schema_kwargs = {}  # package controls schema (usually 'public')

    # Distance: try string first (the error you saw wants this)
    dist_str = _distance_str(distance_cfg)
    attempts: list[Dict[str, Any]] = []

    if "distance" in params:
        attempts.append({**kwargs_base, **schema_kwargs, "distance": dist_str})
    if "distance_strategy" in params:
        # Try string variant for distance_strategy (many builds accept this)
        attempts.append({**kwargs_base, **schema_kwargs, "distance_strategy": dist_str})
        # Finally try enum variant for truly new builds
        attempts.append({**kwargs_base, **schema_kwargs, "distance_strategy": _distance_enum(distance_cfg)})

    # If neither param exists, just one attempt without any distance kw
    if not any(k in params for k in ("distance", "distance_strategy")):
        attempts.append({**kwargs_base, **schema_kwargs})

    last_err: Optional[Exception] = None
    for kw in attempts:
        try:
            store = PGVector(**kw)  # type: ignore[arg-type]
            return store, schema_used, emb_table
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Unable to construct PGVector from {_pgvector_src}; last error: {last_err}"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VectorStoreHandles:
    store: PGVector  # Use: handles.store.as_retriever(search_kwargs={"k": 8})

def get_vectorstore_for_company(
    company: str,
    *,
    engine: Optional[Engine] = None,
) -> VectorStoreHandles:
    """
    Returns a PGVector store bound to the company's dedicated collection.
    Prefers schema 'rag' when supported; otherwise falls back to package defaults.
    """
    stg = get_settings()
    eng = engine or __import__("utils.db", fromlist=["get_engine"]).get_engine()

    prefer_schema = "rag"

    # Bootstrap preferred schema (ignore if perms missing)
    try:
        ensure_pgvector_bootstrap(eng, prefer_schema)
    except Exception:
        pass

    embeddings = make_embeddings()
    conn_str = pg_conn_str_from_engine(eng)
    coll_name = company_collection_name(company)

    store, schema_used, emb_table = _build_pgvector(
        conn_str=conn_str,
        embeddings_obj=embeddings,
        collection_name=coll_name,
        distance_cfg=stg.distance,
        prefer_schema=prefer_schema,
    )

    # Best-effort IVFFLAT index
    try:
        ensure_ivfflat_index(eng, schema=schema_used, distance=stg.distance, table_name=emb_table)
    except Exception:
        pass

    return VectorStoreHandles(store=store)
