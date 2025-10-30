# s360_rag/prompt_templates.py
from langchain_core.prompts import PromptTemplate

STRICT_JSON_PROMPT = PromptTemplate.from_template("""
You are an expert in analytics impact storytelling.

TASK:
1) From persona details, list the top 3–5 Persona KPIs (standard wording).
2) Using ONLY the retrieved case study context, decide if a clear match exists.
3) If NO clear match: set Best_Matching_Case_Study = "None" and Impact_Pointers = [].
4) If a match exists: return the best case study title and  impact pointers as it is grounded in the context.

Persona Details:
{question}

Retrieved Case Study Context:
{context}

Return ONLY valid JSON with this structure:
{{
  "Persona_KPIs": ["..."],
  "Best_Matching_Case_Study": "...",
  "Impact_Pointers": ["...", "..."],
  "Reason_for_Match": "..."
}}
""")
