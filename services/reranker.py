import json
from langchain_openai import AzureChatOpenAI
from config.settings import (
    AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT, MAX_RERANK_INPUT_CHARS
)

_llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT,
    openai_api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version=AZURE_API_VERSION,
    temperature=0.0,
)

RERANK_PROMPT = """
You are an expert business relevance scorer.

Given the persona profile, the list of generated KPIs, and candidate impact statements:
1) Rank the impacts from most to least relevant.
2) Prefer quantitative statements (%, $, hours, accuracy, lift, savings, reduction).
3) Enforce diversity—avoid selecting near-duplicates (same numbers/phrases).
4) Output exactly 3 unique impacts.

Return only a valid JSON array (no markdown, no commentary):

[
  {{
    "Impact": "<verbatim impact text as seen in context>",
    "Industry": "<industry if present, else empty>",
    "BusinessGroup": "<business group if present, else empty>",
    "UseCase": "<use case if present, else empty>"
  }}
]

Persona Context:
{persona_info}

Persona KPIs:
{persona_kpis}

Candidate Impacts:
{candidates}
""".strip()

def _truncate(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit] + " ..."

def rerank_impacts(persona_info: str, persona_kpis: list, candidates: list) -> list:
    # keep only needed fields for candidates
    canon = []
    for c in candidates:
        canon.append({
            "Impact": c.get("Impact",""),
            "Industry": c.get("Industry",""),
            "BusinessGroup": c.get("BusinessGroup",""),
            "UseCase": c.get("UseCase",""),
        })

    payload = RERANK_PROMPT.format(
        persona_info=_truncate(persona_info, 3000),
        persona_kpis=json.dumps(persona_kpis, ensure_ascii=False),
        candidates=_truncate(json.dumps(canon, ensure_ascii=False), MAX_RERANK_INPUT_CHARS),
    )
    resp = _llm.invoke(payload).content.strip().replace("```json","").replace("```","")
    try:
        data = json.loads(resp)
        if not isinstance(data, list): raise ValueError
        # enforce 3 unique impacts
        seen = set()
        out = []
        for d in data:
            imp = (d.get("Impact") or "").strip()
            if imp and imp not in seen:
                seen.add(imp)
                out.append({
                    "Impact": imp,
                    "Industry": d.get("Industry",""),
                    "BusinessGroup": d.get("BusinessGroup",""),
                    "UseCase": d.get("UseCase",""),
                })
            if len(out) == 3:
                break
        return out
    except Exception:
        # fallback: take first 3 non-empty unique
        uniq = []
        seen = set()
        for c in canon:
            t = c["Impact"].strip()
            if t and t not in seen:
                uniq.append(c); seen.add(t)
            if len(uniq) == 3: break
        return uniq
