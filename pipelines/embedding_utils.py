"""
pipelines/embedding_utils.py
Production-grade async embedding utilities for Azure OpenAI.

Key Highlights:
- Reads config from .env (no hard-coded secrets)
- Async + concurrency-safe + exponential backoff
- Order-preserving outputs
- Includes safe sync wrapper for non-async contexts
"""

from __future__ import annotations

import os
import math
import time
import random
import asyncio
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field, ValidationError

# ==========================================================
# ✅ Logging setup
# ==========================================================
logger = logging.getLogger("embedding_utils")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

EMBEDDING_DIM = 1536


# ==========================================================
# ✅ Environment helpers
# ==========================================================
def _req_env(name: str) -> str:
    """Fetch required env var or raise clean error."""
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return val


# ==========================================================
# ✅ Config model (Pydantic validated)
# ==========================================================
class EmbeddingConfig(BaseModel):
    azure_endpoint: str = Field(default_factory=lambda: _req_env("AZURE_ENDPOINT"))
    azure_api_key: str = Field(default_factory=lambda: _req_env("AZURE_API_KEY"))
    azure_api_version: str = Field(default_factory=lambda: os.getenv("AZURE_API_VERSION", "2024-02-15-preview"))
    deployment: str = Field(default_factory=lambda: _req_env("AZURE_EMBED_DEPLOYMENT"))

    max_concurrency: int = Field(default_factory=lambda: int(os.getenv("EMBED_MAX_CONCURRENCY", "4")))
    batch_size: int = Field(default_factory=lambda: int(os.getenv("EMBED_BATCH_SIZE", "64")))
    max_retries: int = Field(default_factory=lambda: int(os.getenv("EMBED_MAX_RETRIES", "5")))
    truncate_tokens: int = Field(default_factory=lambda: int(os.getenv("EMBED_TRUNCATE_TOKENS", "8192")))

    def model_post_init(self, __context: object) -> None:
        self.max_concurrency = max(1, self.max_concurrency)
        self.batch_size = max(1, self.batch_size)
        self.max_retries = max(0, self.max_retries)
        self.truncate_tokens = max(256, self.truncate_tokens)


# ==========================================================
# ✅ Singleton Azure client
# ==========================================================
_client_lock = asyncio.Lock()
_client_singleton: Optional[AsyncAzureOpenAI] = None


async def get_client(cfg: Optional[EmbeddingConfig] = None) -> AsyncAzureOpenAI:
    """Return a cached AsyncAzureOpenAI client (singleton per process)."""
    global _client_singleton
    if _client_singleton:
        return _client_singleton

    async with _client_lock:
        if _client_singleton:
            return _client_singleton

        cfg = cfg or EmbeddingConfig()
        _client_singleton = AsyncAzureOpenAI(
            azure_endpoint=cfg.azure_endpoint,
            api_key=cfg.azure_api_key,
            api_version=cfg.azure_api_version,
        )
        logger.info("✅ AsyncAzureOpenAI embedding client initialized.")
        return _client_singleton


# ==========================================================
# ✅ Helper utilities
# ==========================================================
def _sanitize(s: Optional[str]) -> str:
    return str(s or "").strip()


def _chunks(seq: Sequence[str], n: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _backoff_seconds(attempt: int, base: float = 0.8, cap: float = 20.0) -> float:
    return min(cap, base * (2 ** (attempt - 1))) + random.random() * 0.5


# ==========================================================
# ✅ Core embedding batch worker
# ==========================================================
async def _embed_batch(
    texts: Sequence[str],
    cfg: EmbeddingConfig,
    semaphore: asyncio.Semaphore,
) -> List[Optional[List[float]]]:
    """Embed a single batch with retries and order preservation."""
    client = await get_client(cfg)
    attempt = 0
    last_err = None

    while attempt <= cfg.max_retries:
        try:
            async with semaphore:
                resp = await client.embeddings.create(
                    model=cfg.deployment,
                    input=list(texts),
                )
            return [d.embedding for d in resp.data]
        except Exception as e:
            last_err = e
            attempt += 1
            if attempt > cfg.max_retries:
                logger.error(f"❌ Embedding batch failed after {attempt} attempts: {e}")
                break
            sleep_s = _backoff_seconds(attempt)
            logger.warning(f"⚠️ Embedding batch error (attempt {attempt}/{cfg.max_retries}): {e} | retrying in {sleep_s:.2f}s")
            await asyncio.sleep(sleep_s)

    return [None] * len(texts)


# ==========================================================
# ✅ Public async embedding API
# ==========================================================
async def generate_embeddings(
    texts: Sequence[Optional[str]],
    cfg: Optional[EmbeddingConfig] = None,
) -> List[Optional[List[float]]]:
    """Generate embeddings asynchronously with concurrency + retry handling."""
    try:
        cfg = cfg or EmbeddingConfig()
    except (EnvironmentError, ValidationError) as e:
        logger.error(f"Embedding configuration error: {e}")
        return [None] * len(texts)

    sanitized: List[str] = []
    max_chars = cfg.truncate_tokens * 4
    for t in texts:
        s = _sanitize(t)
        if len(s) > max_chars:
            s = s[:max_chars]
        sanitized.append(s)

    if not sanitized:
        return []

    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    results: List[Optional[List[float]]] = [None] * len(sanitized)
    bs = cfg.batch_size

    batches: List[Tuple[List[int], List[str]]] = []
    for start in range(0, len(sanitized), bs):
        idxs = list(range(start, min(start + bs, len(sanitized))))
        texts_batch = [sanitized[i] for i in idxs]
        batches.append((idxs, texts_batch))

    async def worker(idxs: List[int], btxts: List[str]):
        vecs = await _embed_batch(btxts, cfg, semaphore)
        if len(vecs) != len(btxts):
            logger.error("❌ Embedding length mismatch, skipping batch.")
            return
        for lp, gi in enumerate(idxs):
            results[gi] = vecs[lp]

    await asyncio.gather(*(worker(idxs, btxts) for idxs, btxts in batches))
    return results


# ==========================================================
# ✅ Sync wrapper (for normal scripts)
# ==========================================================
def generate_embeddings_sync(
    texts: Sequence[Optional[str]],
    cfg: Optional[EmbeddingConfig] = None,
) -> List[Optional[List[float]]]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(generate_embeddings(texts, cfg), loop).result()
    else:
        return asyncio.run(generate_embeddings(texts, cfg))


# ==========================================================
# ✅ CLI smoke test
# ==========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick embedding smoke test")
    parser.add_argument("--text", action="append", help="Text to embed (repeatable)", required=True)
    args = parser.parse_args()

    async def _main():
        vecs = await generate_embeddings(args.text)
        dims = None
        for i, v in enumerate(vecs):
            if v is None:
                print(f"[{i}] EMBEDDING: None (failed)")
            else:
                dims = len(v)
                print(f"[{i}] EMBEDDING: dim={len(v)}  sample={v[:8]}...")
        if dims:
            print(f"✓ Completed. Dimension: {dims}")

    asyncio.run(_main())
