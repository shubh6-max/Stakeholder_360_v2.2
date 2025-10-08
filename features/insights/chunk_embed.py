# features/insights/chunk_embed.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Dict

import numpy as np
import streamlit as st
import faiss
import tiktoken

from .config import get_settings
from .clients import get_embed_client, get_embed_deployment_name


# ---------------------------
# Datamodels
# ---------------------------

@dataclass(frozen=True)
class EmbedIndex:
    """A lightweight container holding the FAISS index and the mapping to text."""
    index: faiss.IndexFlatL2
    mapping: List[str]          # chunk text by row id
    dim: int                    # embedding dimensionality


# ---------------------------
# Tokenization / Chunking
# ---------------------------

def _encoder():
    # cl100k_base matches Azure OpenAI text embedding tokenization well
    # If you later change the embed model, you can swap this if needed.
    return tiktoken.get_encoding("cl100k_base")


def _split_tokens(tokens: List[int], max_tokens: int) -> List[List[int]]:
    if max_tokens <= 0:
        return [tokens]
    return [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]


@st.cache_data(ttl=3600, show_spinner=False)
def chunk_text(text: str, max_tokens: Optional[int] = None) -> List[str]:
    """
    Split the text into token-budgeted chunks using tiktoken.
    Cached by (text, max_tokens).
    """
    if not text:
        return []
    stg = get_settings()
    max_tok = int(max_tokens or stg.chunk_tokens or 900)

    enc = _encoder()
    toks = enc.encode(text)
    pieces = _split_tokens(toks, max_tok)

    # decode back to strings
    return [enc.decode(p) for p in pieces if p]


# ---------------------------
# Embeddings (Azure OpenAI)
# ---------------------------

def _hash_chunks(chunks: Sequence[str]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(b"\x00")          # separator to avoid collisions
        h.update(c.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


@st.cache_resource(show_spinner=True)
def embed_chunks(chunks: Tuple[str, ...]) -> EmbedIndex:
    """
    Embed chunks with Azure OpenAI and build a FAISS L2 index.
    Cached by the *content* of chunks (hash).
    """
    if not chunks:
        # Build an empty 1D index to keep typing simpler downstream
        empty = faiss.IndexFlatL2(1)
        return EmbedIndex(index=empty, mapping=[], dim=1)

    client = get_embed_client()
    model = get_embed_deployment_name()

    vectors: List[List[float]] = []
    for ch in chunks:
        # Single-input call to keep memory steady for long corpora
        resp = client.embeddings.create(model=model, input=ch)
        emb = resp.data[0].embedding
        vectors.append(emb)

    dim = len(vectors[0])
    mat = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatL2(dim)
    index.add(mat)

    return EmbedIndex(index=index, mapping=list(chunks), dim=dim)


# ---------------------------
# Retrieval
# ---------------------------

def _embed_query(q: str) -> np.ndarray:
    client = get_embed_client()
    model = get_embed_deployment_name()
    resp = client.embeddings.create(model=model, input=q)
    vec = np.array(resp.data[0].embedding, dtype="float32")
    # FAISS expects shape (1, dim)
    return vec.reshape(1, -1)


def retrieve(
    query: str,
    emb_index: EmbedIndex,
    k: int | None = None,
) -> List[Tuple[str, float]]:
    """
    Search the FAISS index and return a list of (chunk_text, distance).
    Lower distance = closer (L2).
    """
    if not query or emb_index.index.ntotal == 0:
        return []

    stg = get_settings()
    topk = int(k or stg.top_k or 12)

    qv = _embed_query(query)
    D, I = emb_index.index.search(qv, topk)

    out: List[Tuple[str, float]] = []
    for idx, dist in zip(I[0].tolist(), D[0].tolist()):
        if 0 <= idx < len(emb_index.mapping):
            out.append((emb_index.mapping[idx], float(dist)))
    return out


# ---------------------------
# Convenience: one-call build + search
# ---------------------------

def build_and_retrieve(
    text: str,
    query: str,
    *,
    max_tokens: Optional[int] = None,
    k: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Utility for quick experiments:
      - chunk text
      - embed + index
      - retrieve
    Returns (chunks, matches[(chunk, dist), ...])
    """
    chunks = chunk_text(text, max_tokens=max_tokens)
    idx = embed_chunks(tuple(chunks))
    matches = retrieve(query, idx, k=k)
    return chunks, matches
