# features/insights/clients.py
from __future__ import annotations

import os
from typing import Iterable, List, Optional

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI


# Keep this in sync with utils.rag_db.EMBED_DIM
EMBED_DIM = 1536  # text-embedding-3-small → 1536 dims


def _get_env(name: str, *, required: bool = True, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if required and not (val and str(val).strip()):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return str(val).strip() if val is not None else ""


def get_azure_embeddings() -> AzureOpenAIEmbeddings:
    """
    Returns a LangChain embeddings client for Azure OpenAI.
    Expects these env vars:
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_ENDPOINT               (e.g. https://openai-scout-001.openai.azure.com/)
      - AZURE_OPENAI_API_VERSION            (e.g. 2023-05-15)
      - AZURE_EMBED_DEPLOYMENT              (your deployment name for text-embedding-3-small)
    """
    api_key = _get_env("AZURE_OPENAI_API_KEY")
    endpoint = _get_env("AZURE_OPENAI_ENDPOINT")
    api_version = _get_env("AZURE_OPENAI_API_VERSION", required=True)  # you said "2023-05-15"
    deploy = _get_env("AZURE_EMBED_DEPLOYMENT")  # deployment for text-embedding-3-small

    # Note: model name is inferred by Azure via the deployment; we don’t pass "model=" here.
    emb = AzureOpenAIEmbeddings(
        api_key=api_key,
        azure_endpoint=endpoint,
        openai_api_version=api_version,
        azure_deployment=deploy,
    )
    # Sanity note: if you switch to a different deployment (e.g., 3072 dims), update EMBED_DIM.
    return emb


def get_chat_llm(temperature: float = 0.2) -> AzureChatOpenAI:
    """
    Returns an Azure Chat LLM for summarization/extraction.
    Expects:
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_VERSION           (e.g. 2024-02-15-preview is common for chat)
      - AZURE_CHAT_DEPLOYMENT              (your GPT-4/4o/4.1 deployment name)
    """
    api_key = _get_env("AZURE_API_KEY")
    endpoint = _get_env("AZURE_ENDPOINT")
    api_version = _get_env("AZURE_API_VERSION")  # can be shared with embeddings if same
    deploy = _get_env("AZURE_DEPLOYMENT")

    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        openai_api_version=api_version,
        azure_deployment=deploy,
        temperature=temperature,
    )
    return llm


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """
    Convenience helper: embed an iterable of strings with the configured Azure embeddings client.
    """
    emb = get_azure_embeddings()
    # LangChain’s .embed_documents returns List[List[float]]
    vectors = emb.embed_documents(list(texts))
    # Optional: enforce expected dimension to avoid pgvector/IVFFLAT surprises
    if vectors and len(vectors[0]) != EMBED_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: got {len(vectors[0])}, expected {EMBED_DIM}. "
            "Update utils.rag_db.EMBED_DIM and the IVFFLAT index if you changed deployments."
        )
    return vectors