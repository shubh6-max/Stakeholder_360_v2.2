import json
from langchain_openai import AzureChatOpenAI
from config.settings import (
    AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT
)

_llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT,
    openai_api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version=AZURE_API_VERSION,
    temperature=0.0,
)

KPI_PROMPT = """
You are a B2B customer success analyst.

Analyze the stakeholder's role and generate the following:
- Business Functions (from the list below)
- Top 5–7 strategic KPIs for each
- Likely industry focus 
- Use short form names for Industries (e.g. like Consumer Packaged Goods use CPG etc.)

Respond in this JSON structure:
{{
  "Business Function": {{
    "strategic_kpis": ["KPI1", "KPI2", "..."],
    "Industry": ["Sector1", "Sector2"]
  }}
}}

Input:
{input_sentence}
""".strip()

def generate_kpis(persona_info: str) -> dict:
    msg = KPI_PROMPT.format(input_sentence=persona_info)
    out = _llm.invoke(msg).content.strip().replace("```json","").replace("```","")
    try:
        data = json.loads(out)
        if not isinstance(data, dict): raise ValueError
        print("Generated KPIs:", data)
        return data
    except Exception:
        # safe fallback
        return {
            "IT": {
                "strategic_kpis": [
                    "System Uptime %","MTTR","Automation Coverage %",
                    "Change Failure Rate","Incident SLA Compliance %"
                ],
                "Industry": []
            }
        }
