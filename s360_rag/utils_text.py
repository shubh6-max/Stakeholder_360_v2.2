import re, json

def strip_md_fences(s: str) -> str:
    return re.sub(r"```(?:json)?|```", "", s or "").strip()

def safe_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None
