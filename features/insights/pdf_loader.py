# features/insights/case_parsers/pdf_loader.py
from __future__ import annotations

import os
import re
from typing import List, Tuple, Dict, Any

from langchain_core.documents import Document
import fitz  # PyMuPDF


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\u00AD", "", s)  # soft hyphen
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    return s.strip()


def load_pdf_as_documents(
    path: str,
    *,
    clip_headers_footers: bool = False,
) -> Tuple[List[Document], str, List[Dict[str, Any]]]:
    """
    Load a PDF into LangChain Documents (one per page).

    Returns:
        docs: List[Document] with page_content per page
        full_text: concatenated text of all pages
        slide_page_text: [{"index": i, "text": "..."}]  # keeps naming consistent with PPT
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    doc = fitz.open(path)
    docs: List[Document] = []
    slide_page_text: List[Dict[str, Any]] = []

    file_name = os.path.basename(path)
    source_path = os.path.abspath(path)

    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text")  # raw text layout
        txt = _clean_text(txt)

        if clip_headers_footers:
            # Heuristic: drop very first/last line if it repeats across pages (implement later if needed)
            pass

        meta = {
            "file_name": file_name,
            "source_path": source_path,
            "source_type": "pdf",
            "page_index": i,
        }
        docs.append(Document(page_content=txt, metadata=meta))
        slide_page_text.append({"index": i, "text": txt})

    full_text = _clean_text("\n\n".join(x["text"] for x in slide_page_text))
    return docs, full_text, slide_page_text