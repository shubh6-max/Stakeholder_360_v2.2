from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class PersonaInput(BaseModel):
    client_name: Optional[str] = None
    email_id: Optional[str] = None
    client_designation: Optional[str] = None
    seniority_level: Optional[str] = None
    working_group: Optional[str] = None
    business_unit: Optional[str] = None
    business_functions: Optional[str] = None
    service_line: Optional[str] = None
    industry_hint: Optional[str] = None
    linkedin_title: Optional[str] = None
    linkedin_about: Optional[str] = None
    linkedin_desc_html: Optional[str] = None

class KPIBlock(BaseModel):
    Business_Function: Dict[str, List[str]] = Field(
        description='{"strategic_kpis": [...], "Industry": [...]}'
    )

class MatchOutput(BaseModel):
    Persona_KPIs: List[str]
    Best_Matching_Case_Study: str
    Impact_Pointers: List[str]
    Reason_for_Match: str