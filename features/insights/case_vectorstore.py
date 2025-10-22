# features/insights/case_vectorstore.py
from __future__ import annotations

from typing import Iterable, List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.rag_env import get_embeddings, chunking_defaults


def chunk_strings(
    text: str,
    chunk_chars: int | None = None,
    overlap_chars: int | None = None,
) -> List[str]:
    cfg = chunking_defaults()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars or cfg["chunk_chars"],
        chunk_overlap=overlap_chars or cfg["overlap_chars"],
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text or "")


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    emb = get_embeddings()
    # LangChain batching is internal; we keep it simple
    return emb.embed_documents(list(texts))


def embed_query(text: str) -> List[float]:
    emb = get_embeddings()
    return emb.embed_query(text or "")
