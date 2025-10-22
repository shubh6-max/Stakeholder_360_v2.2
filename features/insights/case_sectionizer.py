# features/insights/case_sectionizer.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SectionedText:
    sections: Dict[str, str]  # {"problem": "...", "solution": "...", ... , "general": "..."}

# Heuristic heading patterns (extend freely)
_PATS = {
    "problem":  r"(?:^|\n)\s*(?:problem|challenge|context|pain\s*points?)\s*[:\-]?\s*$",
    "solution": r"(?:^|\n)\s*(?:solution|approach|implementation|methodology)\s*[:\-]?\s*$",
    "impact":   r"(?:^|\n)\s*(?:impact|outcomes?|results?|benefits?|business\s*value)\s*[:\-]?\s*$",
    "kpi":      r"(?:^|\n)\s*(?:kpis?|metrics|by\s*the\s*numbers)\s*[:\-]?\s*$",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE | re.MULTILINE) for k, v in _PATS.items()}


def _normalize(text: str) -> str:
    text = re.sub(r"\x00", " ", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def sectionize(full_text: str, slide_page_text: List[Dict]) -> SectionedText:
    """
    Very robust but simple: use headings in the concatenated text; if none found,
    fallback to slide/page-level cues and finally put everything in 'general'.
    """
    s = _normalize(full_text)
    if not s:
        return SectionedText(sections={"general": ""})

    hits: List[Tuple[int, str]] = []
    for name, pat in _COMPILED.items():
        for m in pat.finditer(s):
            hits.append((m.start(), name))
    hits.sort(key=lambda x: x[0])

    if not hits:
        # Try slide/page cues: look for lines beginning with typical titles per page
        # (kept minimal; you can expand as needed)
        pass

    buckets: Dict[str, List[str]] = {"problem": [], "solution": [], "impact": [], "kpi": [], "general": []}

    if not hits:
        buckets["general"].append(s)
    else:
        # Slice between markers
        for idx, (pos, name) in enumerate(hits):
            end = hits[idx + 1][0] if idx + 1 < len(hits) else len(s)
            chunk = s[pos:end].strip()
            # remove heading line itself (first line) if present
            chunk = re.sub(_COMPILED[name], "", chunk, count=1).strip()
            buckets[name].append(chunk)

        # Anything before first heading?
        first_pos = hits[0][0]
        pre = s[:first_pos].strip()
        if pre:
            buckets["general"].append(pre)

    merged = {k: "\n\n".join(v).strip() for k, v in buckets.items() if v}
    # Ensure keys exist
    for k in ("problem", "solution", "impact", "kpi", "general"):
        merged.setdefault(k, "")
    return SectionedText(sections=merged)
