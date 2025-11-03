# features/insights/case_prompts.py
from __future__ import annotations

# System & user prompt parts kept separate for easy iteration/versioning.
SYSTEM_PROMPT = """\
You are an expert analyst extracting structured facts from case-study documents.
You must return ONLY valid JSON that conforms to the provided schema.
Do not include any explanations or commentary outside the JSON.
Do not invent information. If a field is not present, use null or an empty list.
Copy metrics verbatim (e.g., percentages, hours/week, currency with symbols)."""

# This goes into the user message together with full_text and the JSON schema.
RULES_BLOCK = """\
Rules and notes:
- Prefer exact phrases found in the document.
- Summarize problem and solution into one crisp paragraph each (no bullets).
- For impact_pointers and kpi, return concise bullets; include numbers/symbols verbatim when present.
- If a URL appears in the document for the case, set case_study_link; otherwise, null.
- Never output anything except a single JSON object matching the schema."""