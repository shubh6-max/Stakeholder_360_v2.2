# utils/rag_env.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# ✅ Use Azure-specific LangChain wrappers
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

EMBED_DIM = 1536
DEFAULT_SECTION_CHUNK_CHARS = int(os.getenv("CASE_CHUNK_CHARS", "4200"))
DEFAULT_SECTION_CHUNK_OVERLAP_CHARS = int(os.getenv("CASE_CHUNK_OVERLAP_CHARS", "500"))

def _env(name: str, required: bool = True, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if required and (v is None or str(v).strip() == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v

def assert_env() -> None:
    _env("AZURE_API_KEY")
    _env("AZURE_ENDPOINT")
    _env("AZURE_API_VERSION")
    _env("AZURE_DEPLOYMENT")         # chat deployment name (e.g., gpt-4o, gpt-4o-mini)
    _env("AZURE_EMBED_DEPLOYMENT")   # embeddings deployment name

    _env("PGHOST"); _env("PGDATABASE"); _env("PGUSER"); _env("PGPASSWORD")
    os.getenv("PGPORT", "5432")

def get_chat_llm(
    *,
    temperature: float = 0.0,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> AzureChatOpenAI:
    """
    Azure OpenAI chat LLM for LangChain.
    Use a **non-vision** deployment that supports tool/function calling
    (e.g., gpt-4o, gpt-4o-mini). We'll use structured output via function_calling.
    """
    assert_env()
    return AzureChatOpenAI(
        azure_deployment=model or _env("AZURE_DEPLOYMENT"),
        api_version=_env("AZURE_API_VERSION"),
        azure_endpoint=_env("AZURE_ENDPOINT"),
        openai_api_key=_env("AZURE_API_KEY"),
        temperature=temperature,
        timeout=timeout,
        **(extra_kwargs or {}),
    )

def get_embeddings(*, model: Optional[str] = None, chunk_size: int = 32) -> AzureOpenAIEmbeddings:
    assert_env()
    return AzureOpenAIEmbeddings(
        azure_deployment=_env("AZURE_EMBED_DEPLOYMENT"),
        api_version=_env("AZURE_API_VERSION"),
        azure_endpoint=_env("AZURE_ENDPOINT"),
        openai_api_key=_env("AZURE_API_KEY"),
        chunk_size=chunk_size,
        # model name is optional for Azure wrappers; deployment controls it.
    )

@dataclass(frozen=True)
class PgConn:
    host: str; db: str; user: str; password: str; port: int = 5432
    @classmethod
    def from_env(cls) -> "PgConn":
        return cls(
            host=_env("PGHOST"),
            db=_env("PGDATABASE"),
            user=_env("PGUSER"),
            password=_env("PGPASSWORD"),
            port=int(os.getenv("PGPORT", "5432")),
        )
    def sqlalchemy_url(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

def chunking_defaults() -> Dict[str, int]:
    return {"chunk_chars": DEFAULT_SECTION_CHUNK_CHARS, "overlap_chars": DEFAULT_SECTION_CHUNK_OVERLAP_CHARS}

def case_collections() -> Dict[str, str]:
    return {
        "schema": "insights",
        "docs_table": "insights.case_docs",
        "facts_table": "insights.case_facts",
        "chunks_table": "insights.case_chunks",
        "vector_collection": "case_study_collection",
    }
