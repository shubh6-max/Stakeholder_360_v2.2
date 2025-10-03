import os
import io
import json
import base64
from PIL import Image
import json
import pandas as pd
import time
from dotenv import load_dotenv
from snapshot_to_df import extract_current_profiles_from_json


# ===================================================================
# Load environment variables (for secure credentials)
load_dotenv()

# Set your folder path
folder_path = r"C:\Users\ShubhamVishwasPurani\Downloads\marcketing\snapshots"
# Load the Excel files
use_cases_df = pd.read_excel(r"C:\Users\ShubhamVishwasPurani\Downloads\marcketing\Copy of Use Cases with Pitch - vF1.xlsx")


# ===================================================================

import pandas as pd

def get_manager_enriched_details(df: pd.DataFrame, client_name: str) -> pd.DataFrame:
    # === Base columns to retain ===
    base_columns = [
        'Client Name', 'Designation', '1st degree Manager',
        'Business Segment', 'Working Group',
        'Business Functions', 'Designation Seniority', 'LinkedIn URL'
    ]

    # === Filter relevant columns ===
    leaddatabase_filtered = df[base_columns]

    # === Create manager mapping DataFrame without duplicate key column ===
    manager_info_df = leaddatabase_filtered[[  
        'Client Name', 'Designation', 'Business Segment', 'Working Group',
        'Business Functions', 'Designation Seniority', 'LinkedIn URL'
    ]].rename(columns={
        'Client Name': '1st degree Manager',
        'Designation': 'Manager Designation',
        'Business Segment': 'Manager Business Segment',
        'Working Group': 'Manager Working Group',
        'Business Functions': 'Manager Business Functions',
        'Designation Seniority': 'Manager Designation Seniority',
        'LinkedIn URL': 'Manager LinkedIn URL'
    })

    # === Merge to enrich with manager details ===
    leaddatabase_filtered = leaddatabase_filtered.merge(
        manager_info_df,
        on='1st degree Manager',
        how='left'
    )

    # === Reorder columns for final output ===
    final_columns = base_columns + [
        'Manager Designation', 'Manager Business Segment',
        'Manager Working Group', 'Manager Business Functions',
        'Manager Designation Seniority', 'Manager LinkedIn URL'
    ]
    leaddatabase_filtered = leaddatabase_filtered[final_columns]

    # === Filter by client name ===
    leaddatabase_filtered = leaddatabase_filtered[leaddatabase_filtered["Client Name"] == client_name]

    return leaddatabase_filtered


# Call the enrichment function
# leaddatabase_filtered = get_manager_enriched_details(leaddatabase, "Ingrid Knies")

# Call the function
# df = extract_current_profiles_from_json(folder_path=r"C:\Users\ShubhamVishwasPurani\Downloads\marcketing\MARS_pet_snapshots_location", verbose=True)


# # Merge the two DataFrames on LinkedIn URLs
# merged_df = leaddatabase_filtered.merge(
#     df.rename(columns={"input_url": "LinkedIn URL"}),  # match column name for join
#     on="LinkedIn URL",
#     how="left",
# )

# # View the resulting DataFrame
# merged_df.drop(columns=["name"],inplace=True)

# =====================================================

# === Helper: Build sentence only if value exists ===
def build_input_sentence(row):
    def safe_get(key):
        value = row.get(key, "")
        return value if pd.notna(value) and str(value).strip() else None

    parts = []

    name = safe_get("Client Name")
    designation = safe_get("Designation")
    business_segment = safe_get("Business Segment")
    working_group = safe_get("Working Group")
    business_functions = safe_get("Business Functions")
    seniority = safe_get("Designation Seniority")
    manager = safe_get("1st degree Manager")
    manager_designation = safe_get("Manager Designation")
    current_title = safe_get("current_position_title")
    # linkedin = safe_get("LinkedIn URL")
    description = safe_get("description_html")

    if name and designation:
        parts.append(f"{name} is currently working as {designation}")
    if business_segment:
        parts.append(f"in the {business_segment} segment")
    if working_group:
        parts.append(f"within the {working_group} group")
    if business_functions:
        parts.append(f"focused on {business_functions} function")
    if seniority:
        parts.append(f"This role holds a seniority level of {seniority}")
    if name and manager:
        mgr = f"{name} reports to {manager}"
        if manager_designation:
            mgr += f", who is designated as {manager_designation}"
        parts.append(mgr + ".")
    if current_title:
        parts.append(f"Their current title is '{current_title}'")
    # if linkedin:
    #     parts.append(f"and their professional profile can be found here: {linkedin}.")
    if description:
        parts.append(f"The role involves the following responsibilities or background: {description}")
    else:
        parts.append("No role-specific description is provided.")

    return " ".join(parts)

# === Prompt Builder Function ===
def build_llm_prompt(row):
    input_sentence = build_input_sentence(row)

    return f"""
    You are a B2B customer success manager, focused on strategic account expansion and solution mapping.

    Your organization offers the following services under key business groups:

    1. **Data and Analytics**
    - Data Diagnostics
    - Predictive Modeling
    - Statistical Modeling
    - AI/ML Blueprint
    - Decision Culture
    - Strategy & Roadmap

    2. **Engineering**
    - Cloud Engineering
    - Data Platform Implementation
    - MLOps
    - Data as a Service

    3. **Information Technology**
    - Data & Enterprise BI Migration
    - Data Operations & Governance
    - Master Data Management

    4. **Digital Transformation**
    - Strategy – GenAI Roadmap
    - Business Agent Programs & Prompt Engineering
    - Foundation – LLM & Data

    5. **Innovation and Growth**
    - Business Solutions & Insights
    - Custom GenAI Accelerators
    - NucliOS Integration

    6. **Product Management**
    - Tools – Business Agent Programs & Prompt Engineering
    - Strategy – GenAI Roadmap

    You may also recommend additional services from these categories that align with the stakeholder's role and function.

    Please following as Business Function:
    - Central Analytics  
    - Commercial  
    - Consumer Insights  
    - Digital  
    - Ecommerce  
    - Finance And Procurement  
    - HR  
    - IT 
    - Manufacturing  
    - Marketing  
    - Operations  
    - Platform  
    - R&D  
    - Revenue  
    - Sales  
    - Shopper Insights    
    - Strategy And Planning  
    - Supply Chain
    - Others
    ---
    Based on the stakeholder profile provided below, analyze their scope and strategic responsibilities. Return only the **relevant business functions** from the above list, along with **strategic KPIs** for each and Industry .

    Respond only in valid JSON format with the following structure:

    ```json
    {{
    "Business Function": {{
        "strategic_kpis": [
        "KPI1",
        "KPI2",
        "KPI3"
        ],
        Industry:[]
    }}
    }}
    Only include business functions that apply.
    Let the KPI list for each function be specific and tailored based on the stakeholder role in concise, metrics-driven style give atleast 5 KPIs but most relevent. 
    but do not include numbers use "(%)" and no explaination.

    <Client_Record>
    {input_sentence}

    Respond ONLY in valid JSON format.
    """.strip()


# ===========================================================================

def get_kpi_alignment_response(client_kpi, strategic_kpi):
    prompt = f"""
You are a B2B customer success manager, focused on strategic account expansion and solution mapping.

Given the following:
- Client KPIs
- Strategic KPIs

Do two things:
1.  Understand the context for both KPIs and Assess alignment on a scale from 1 (very high match) to 5 (very low match).
2. Generate a one-line impact pointer with measurable business benefit (e.g. savings, ROI, % improvement).

Respond ONLY in this strict JSON format:
{{
  "score": <number from 1 to 5>,
  "impact": "<short business-style headline>"
}}

Client KPI:
{client_kpi}

Strategic KPI:
{strategic_kpi}
"""
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content.replace("```json","").replace("```","").strip()  # type: ignore

        if not content or not content.startswith("{"):
            print("⚠️ Invalid or empty response:", content)
            return {"score": None, "impact": None}

        data = json.loads(content)
        return {
            "score": int(data.get("score", 0)),
            "impact": data.get("impact", "")
        }

    except Exception as e:
        print("❌ LLM Exception:", e)
        return {"score": None, "impact": None}
    





