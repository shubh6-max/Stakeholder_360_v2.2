# features/insights/case_schema.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class CaseFacts(BaseModel):
    """
    Strict schema for LLM-extracted case-study facts.
    Keep field names stable—they map 1:1 to DB columns in insights.case_facts.
    """
    case_study_name: str = Field(..., description="The official or slide title of the case study.")
    industry: Optional[str] = Field(None, description="Client industry or sector, e.g., 'FMCG', 'Transportation'.")
    business_function: Optional[str] = Field(None, description="Primary function, e.g., 'Marketing', 'Supply Chain'.")
    problem_statement: Optional[str] = Field(None, description="Single coherent paragraph describing the problem/challenge.")
    impact_pointers: List[str] = Field(default_factory=list, description="Concise impact bullets with metrics where possible.")
    solution_approach: Optional[str] = Field(None, description="One paragraph summarizing solution and approach.")
    case_study_link: Optional[str] = Field(None, description="URL if present on the deck/PDF; otherwise null.")
    kpi: List[str] = Field(default_factory=list, description="Key metric statements, e.g., '97% accuracy', '$1M+ saved'.")

    @classmethod
    def json_schema_str(cls) -> str:
        """String form of the JSON schema to embed in prompts."""
        # Pydantic v2
        import json
        return json.dumps(cls.model_json_schema(), ensure_ascii=False, indent=2)
