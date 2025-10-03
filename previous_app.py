import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import time
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")


# ============================================================
load_dotenv()
snapshot_folder = r"C:\Users\ShubhamVishwasPurani\Downloads\marcketing\MARS_pet_snapshots_location"
kpi_source_truth=r"C:\Users\ShubhamVishwasPurani\Downloads\marcketing\Copy of Use Cases with Pitch - vF1.xlsx"

# === Read Azure settings from environment variables ===
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

# Initialize Azure OpenAI client
client_api = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT, # type: ignore
)
# ============================================================
# === Step 1: Clean Stakeholder Data and Enrich with Manager Info ===

def enrich_with_manager_metadata(filtered_stakeholders_df: pd.DataFrame, selected_client) -> pd.DataFrame:
    """
    Given filtered stakeholders, enrich with manager metadata by performing a self-merge.
    """
    # Ensure only relevant base columns are used
    base_cols = [
        'Client Name', 'Designation', '1st degree Manager',
        'Business Segment', 'Working Group',
        'Business Functions', 'Designation Seniority', 'LinkedIn URL'
    ]
    df = filtered_stakeholders_df[base_cols].copy()

    # Prepare manager metadata (excluding '1st degree Manager' to avoid duplication)
    manager_data = df.drop(columns=['1st degree Manager']).rename(columns={
        'Client Name': '1st degree Manager',
        'Designation': 'Manager Designation',
        'Business Segment': 'Manager Business Segment',
        'Working Group': 'Manager Working Group',
        'Business Functions': 'Manager Business Functions',
        'Designation Seniority': 'Manager Designation Seniority',
        'LinkedIn URL': 'Manager LinkedIn URL'
    })

    # Perform the self-merge
    enriched_df = df.merge(manager_data, on='1st degree Manager', how='left')

    enriched_df=enriched_df[enriched_df["Client Name"]==selected_client]

    return enriched_df

# =====================================================================================
def merge_with_json_snapshots(df: pd.DataFrame, folder_path: str) -> pd.DataFrame:
    """
    Merge stakeholder Excel data with persona JSON snapshot info.

    Args:
        df (pd.DataFrame): Stakeholder DataFrame containing LinkedIn URLs.
        folder_path (str): Path to the folder containing JSON snapshots.

    Returns:
        pd.DataFrame: Merged DataFrame with enriched fields.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Snapshot folder does not exist: {folder_path}")

    records = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        raise ValueError("Empty file")

                    profiles = json.loads(content)

                    if not isinstance(profiles, list):
                        raise ValueError("Expected list of profiles")

                    for profile in profiles:
                        if not isinstance(profile, dict):
                            continue
                        name = profile.get("name", "")
                        input_url = profile.get("input", {}).get("url", "")
                        experiences = profile.get("experience", [])

                        title, description_html = "", ""
                        if isinstance(experiences, list):
                            for exp in experiences:
                                if exp.get("end_date", "") == "Present":
                                    title = exp.get("title", "")
                                    description_html = exp.get("description_html", "")
                                    break

                        records.append({
                            "input_url": input_url,
                            "current_position_title": title,
                            "description_html": description_html
                        })

            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")

    json_df = pd.DataFrame(records).rename(columns={"input_url": "LinkedIn URL"})
    merged_df = df.merge(json_df, on="LinkedIn URL", how="left")

    return merged_df

# ==============================================================
# === Sentence Builder ===
def build_input_sentence(row):
    def safe_get(key):
        val = row.get(key, "")
        return val if pd.notna(val) and str(val).strip() else None

    parts = []

    if safe_get("Client Name") and safe_get("Designation"):
        parts.append(f"{row['Client Name']} is currently working as {row['Designation']}")
    if safe_get("Business Segment"):
        parts.append(f"in the {row['Business Segment']} segment")
    if safe_get("Working Group"):
        parts.append(f"within the {row['Working Group']} group")
    if safe_get("Business Functions"):
        parts.append(f"focused on {row['Business Functions']} function")
    if safe_get("Designation Seniority"):
        parts.append(f"This role holds a seniority level of {row['Designation Seniority']}")
    if safe_get("Client Name") and safe_get("1st degree Manager"):
        mgr = f"{row['Client Name']} reports to {row['1st degree Manager']}"
        if safe_get("Manager Designation"):
            mgr += f", who is designated as {row['Manager Designation']}"
        parts.append(mgr + ".")
    if safe_get("current_position_title"):
        parts.append(f"Their current title is '{row['current_position_title']}'")
    if safe_get("description_html"):
        parts.append(f"The role involves the following responsibilities: {row['description_html']}")
    else:
        parts.append("No role-specific description is provided.")

    return " ".join(parts)

# ==============================================================
# === Prompt Generator ===
def build_llm_prompt(row):
    input_sentence = build_input_sentence(row)
    return f"""
You are a B2B customer success analyst.

Analyze the stakeholder's role and generate the following:
- Business Functions (from the list below)
- Top 5–7 strategic KPIs for each
- Likely industry focus

Allowed Business Functions:
Central Analytics, Commercial, Consumer Insights, Digital, Ecommerce, Finance And Procurement,
HR, IT, Manufacturing, Marketing, Operations, Platform, R&D, Revenue, Sales, Shopper Insights,
Strategy And Planning, Supply Chain, Others

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
# ==============================================================

# === Main KPI Extraction Runner ===
def run_kpi_analysis(df, sample_limit=None):
    results = []

    for i, row in df.iterrows():
        if sample_limit and i >= sample_limit:
            break

        prompt = build_llm_prompt(row)
        content = ""
        try:
            response = client_api.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {"role": "system", "content": "You are a B2B insights analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip() # type: ignore

            parsed_json = json.loads(content)
        except Exception as e:
            print(f"❌ Error for {row['Client Name']}: {e}")
            parsed_json = {"raw_response": content} # type: ignore

        results.append({
            "Client Name": row.get("Client Name"),
            "LinkedIn URL": row.get("LinkedIn URL"),
            "Response": parsed_json
        })

        time.sleep(1)

    return results

# ==============================================================
def extract_persona_info(profiles):
    """
    Given a list of valid profile dictionaries (from JSON), extract key fields
    and return a structured pandas DataFrame.
    """
    extracted_data = []

    for profile in profiles:
        try:
            name = profile.get("name", "").strip()
            input_url = profile.get("input", {}).get("url", "")
            experience = profile.get("experience", [])
            avatar = str(profile.get("avatar", ""))
            designation = ""
            description_html = ""

            # Pick the most recent "Present" role
            if isinstance(experience, list):
                for role in experience:
                    if role.get("end_date", "") == "Present":
                        designation = role.get("title", "").strip()
                        description_html = role.get("description_html", "").strip()
                        break

            extracted_data.append({
                "name": name,
                "designation": designation,
                "location": profile.get("location", ""),
                "email": profile.get("email", ""),
                "linkedin_url": input_url,
                "description_html": description_html,
                "avatar": avatar
            })

        except Exception as e:
            print(f"⚠️ Skipping profile due to error: {e}")

    return pd.DataFrame(extracted_data)

# ==============================================================
def flatten_kpi_results(results: list) -> pd.DataFrame:
    """
    Flattens the KPI analysis results into a structured DataFrame.

    Args:
        results (list): List of dictionaries returned from run_kpi_analysis().

    Returns:
        pd.DataFrame: Flattened DataFrame with columns -
                      'Client Name', 'LinkedIn URL', 'Business Function', 'Strategic KPIs'
    """
    flat_rows = []

    for item in results:
        client_name = item.get("Client Name")
        linkedin_url = item.get("LinkedIn URL")
        response = item.get("Response", {})

        for function, details in response.items():
            kpis = details.get("strategic_kpis", [])
            flat_rows.append({
                "Client Name": client_name,
                "LinkedIn URL": linkedin_url,
                "Business Function": function,
                "Strategic KPIs": ", ".join(kpis) if isinstance(kpis, list) else kpis
            })

    return pd.DataFrame(flat_rows)
# ==============================================================
def map_kpis_to_use_cases(kpi_grouped_df: pd.DataFrame, use_case_file_path: str) -> pd.DataFrame:
    """
    Matches stakeholder strategic KPIs to use cases from a reference Excel sheet.

    Args:
        kpi_grouped_df (pd.DataFrame): Output from flatten_kpi_results().
        use_case_file_path (str): Path to the use case Excel file (e.g., "Use Cases with Pitch.xlsx").

    Returns:
        pd.DataFrame: Final matched DataFrame with stakeholder KPIs and aligned use cases.
    """

    # Load the use case Excel
    use_cases_df = pd.read_excel(use_case_file_path)

    # Normalize both columns for comparison
    use_cases_df['Business Group'] = use_cases_df['Business Group'].str.lower().str.strip()
    kpi_grouped_df['Business Function'] = kpi_grouped_df['Business Function'].str.lower().str.strip()

    # Prepare reference functions
    reference_functions = kpi_grouped_df['Business Function'].unique().tolist()

    # Step 1: Filter use cases with partial match
    filtered_use_cases = use_cases_df[
        use_cases_df['Business Group'].apply(lambda x: any(ref in x for ref in reference_functions))
    ].copy()

    # Step 2: Join based on partial match logic
    matched_rows = []

    for _, use_row in filtered_use_cases.iterrows():
        bg = use_row['Business Group']
        for _, ref_row in kpi_grouped_df.iterrows():
            bf = ref_row['Business Function']
            if bf in bg:
                matched_rows.append({
                    "Client Name": ref_row.get("Client Name"),
                    "LinkedIn URL": ref_row.get("LinkedIn URL"),
                    "Business Group": ref_row.get("Business Function"),
                    "Use Case": use_row.get("Use Case"),
                    "Client KPI": use_row.get("KPI"),
                    "Strategic KPIs": ref_row.get("Strategic KPIs"),
                    "MathCo Case Study": use_row.get("MathCo Case Study")
                })

    return pd.DataFrame(matched_rows)
# ========================================================================
def get_kpi_alignment_response(client_kpi, strategic_kpi):
    prompt = f"""
    You are a B2B customer success manager, focused on strategic account expansion and solution mapping.

    Given the following:
    - Client KPIs
    - Strategic KPIs

    Do two things:
    1.  Understand the context for both KPIs and Assess alignment on a scale from 1 (very high match) to 5 (very low match).

    Respond ONLY in this strict JSON format:
    {{
    "score": <number from 1 to 5>,
    }}

    Client KPI:
    {client_kpi}

    Strategic KPI:
    {strategic_kpi}
    """
    try:
        response = client_api.chat.completions.create(
            model="gpt-4-vision-preview",
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
    

# ==============================================================
def extract_top_kpis(grouped_df: pd.DataFrame, top_n: int = 3) -> Tuple[pd.DataFrame, str]:
    """
    Sorts the DataFrame by Client Name and KPI Match Rank, keeps top N rows,
    and extracts all unique KPIs from Strategic and Client KPIs.

    Args:
        grouped_df (pd.DataFrame): Input DataFrame with KPI alignment.
        top_n (int): Number of top rows to retain after sorting.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Filtered DataFrame and combined KPI list.
    """
    # Step 1: Sort and keep top rows
    grouped_df_sorted = grouped_df.sort_values(
        by=["Client Name", "KPI Match Rank"],
        ascending=[True, True]
    ).head(top_n)

    # Step 2: Extract KPIs
    strategic_kpis = []
    client_kpis = []

    for kpi in grouped_df_sorted["Strategic KPIs"]:
        if pd.notna(kpi):
            strategic_kpis.extend([k.strip() for k in kpi.split(",•")])

    for j in grouped_df_sorted["Client KPI"]:
        if pd.notna(j):
            client_kpis.extend([k.strip() for k in j.split(",•")])

    combined_kpis = strategic_kpis + client_kpis

    combined_kpis=list(set(combined_kpis))

    combined_kpis_str = "<br>".join(f"• {k.strip()}" for k in sorted(set(combined_kpis)) if k.strip())

    return grouped_df_sorted, combined_kpis_str

# ==============================================================

def render_info_table(title, data_dict, color="#cce5ff",name_for_link=None):
    """Render a colored section title + 2-column HTML table with LinkedIn URL as clickable"""
    st.markdown(f"""
        <div style='background-color:{color};padding:8px;border-radius:5px;
        margin-top:20px;margin-bottom:5px;font-weight:bold;font-size:16px'>
            {title}
        </div>
    """, unsafe_allow_html=True)

    table_html = """
        <table style='width:100%;border-collapse:collapse;font-size:15px;'>
    """

    for k, v in data_dict.items():
        if k.lower().startswith("linkedin") and pd.notna(v):
            label = name_for_link or v
            v = f"<a href='{v}' target='_blank' style='color:#1a0dab;text-decoration:underline'>linkedin/{label}</a>"
        elif not pd.notna(v):
            v = "-"

            
        table_html += f"""
        <tr style='border-bottom:1px solid #ddd'>
            <td style='padding:6px 10px;font-weight:600;width:40%;background-color:#f2f2f2'>{k}</td>
            <td style='padding:6px 10px;background-color:#f2f2f2'>{v}</td>
        </tr>
        """

    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)



# Page layout
st.set_page_config(page_title="Stakeholder Org Viewer", layout="wide",page_icon="https://media.licdn.com/dms/image/v2/D4D0BAQFSLuRei6pVZA/company-logo_200_200/B4DZfJuCNfGgAI-/0/1751435978200/themathcompany_logo?e=1756944000&v=beta&t=CkeqG4ihtOep-IGUMLTLMItiVdFJ4-TroEeSoXs1Jxw")

st.markdown("""
    <style>
    /* Remove default Streamlit padding from top */
    .block-container {
        padding-top: 0rem !important;
    }

    .stApp {
        background-color: #ebe3d6;
    }

    .logo-container {
        display: flex;
        align-items: center;
        padding: 5px 0px 0px 10px; /* Reduced top padding */
        margin-top: 20px;  /* Pull logo upward */
        margin-left: -0px;
        margin-bottom: 40px;
    }

    .logo-container img {
        width: 150px;
        background-color: white;
        padding: 5px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    </style>

    <div class="logo-container">
        <img src="https://upload.wikimedia.org/wikipedia/commons/8/88/MathCo_Logo.png" />
    </div>
""", unsafe_allow_html=True)

st.markdown("### **Stakeholder 360**",)

uploaded_file = st.file_uploader("**📂 Upload Excel File**", type=["xlsx"])

col_filter1, col_filter2,col_filter3, col_filter4= st.columns(4)

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    with col_filter1:
        sheet = st.selectbox("**Select Sheet**", ["-- Select Sheet --"] + xls.sheet_names)
        # Block until valid selection
        if sheet == "-- Select Sheet --":
            st.warning("⚠️ Please select the updated sheet to continue.")
            st.stop()  # 🚫 HALTS further app execution
        else:
            df = pd.read_excel(xls, sheet_name=sheet)
    
    df["Vendor CompanyName"].fillna("-")
    
    df = df.dropna(subset=["Client Name"])

    with col_filter2:
        selected_working_group = st.selectbox(
            "**Working Group**", ["All"] + sorted(df["Working Group"].dropna().unique().tolist())
        )

    with col_filter3:
        selected_business_function = st.selectbox(
            "**Business Functions**", ["All"] + sorted(df["Business Functions"].dropna().unique().tolist())
        )

    # --- Apply filtering logic ---
    filtered_df = df.copy()
    if selected_working_group != "All":
        filtered_df = filtered_df[filtered_df["Working Group"] == selected_working_group]

    if selected_business_function != "All":
        filtered_df = filtered_df[filtered_df["Business Functions"] == selected_business_function]

    # --- Stakeholder dropdown ---
    client_names = filtered_df["Client Name"].dropna().unique()

    if len(client_names) == 0:
        st.warning("⚠️ No stakeholders match the selected filters.")
        st.stop()

    with col_filter4:
        selected_client = st.selectbox("**Select Stakeholder**", sorted(client_names))

    # Work only on filtered row
    row = filtered_df[filtered_df["Client Name"] == selected_client].iloc[0]


    if selected_client:
        # Get selected row
        row = df[df["Client Name"] == selected_client].iloc[0]
        
        # Extract hierarchy
        client = row["Client Name"]
        mgr_1 = row["1st degree Manager"]
        mgr_2 = row["2nd Degree Manager"]
        target_client=row["avatar"]
        target_client_link_url=row["LinkedIn URL"]

        # --- Build Mini Org Chart with reportees ---
        nodes = []
        edges = []
        added = set()

        def add_node(name, title=None, color="lightblue", shape="box"):
            if pd.notna(name) and name not in added:
                # Combine name + title (designation)
                label = f"{name}\n{title}" if title else name
                nodes.append(Node(
                    id=name,
                    label=label,
                    color=color,
                    shape=shape,
                    font={"size": 25}
                ))
                added.add(name)


        # Managers
        # For 2nd Degree Manager
        add_node(mgr_2, title=df[df["Client Name"] == mgr_2]["Designation"].values[0] if mgr_2 in df["Client Name"].values else None, color="lightgray")

        # For 1st Degree Manager
        add_node(mgr_1, title=df[df["Client Name"] == mgr_1]["Designation"].values[0] if mgr_1 in df["Client Name"].values else None, color="#4A90E2")

        # For selected stakeholder
        add_node(client, title=row["Designation"], color="#6AA84F")

        # Edges upward
        if pd.notna(mgr_2) and pd.notna(mgr_1):
            edges.append(Edge(source=mgr_2, target=mgr_1))
        if pd.notna(mgr_1):
            edges.append(Edge(source=mgr_1, target=client))

        # For reportees
        reportees_df = df[df["1st degree Manager"] == client]
        for _, rep in reportees_df.iterrows():
            rep_name = rep["Client Name"]
            rep_title = rep["Designation"]
            add_node(rep_name, title=rep_title, color="#FFF2CC")
            edges.append(Edge(source=client, target=rep_name))
        # Config for AGraph
        # MAIN
        config = Config(
        width="100%", # type: ignore
        height=500,
        font={"size":50},
        directed=True,
        physics=False,
        hierarchical=True,
        hierarchicalOption={
            "direction": "UD",  # Top-down
            "sortMethod": "directed",
            "nodeSpacing":6000,
            "levelSeparation":600,
            "parentCentralization":True,
        }
    )

        st.subheader(f"🙎🏻‍♂️ Org Chart for {selected_client}")
        agraph(nodes=nodes, edges=edges, config=config)


        st.markdown("---")
        # ==== DEFINE COLUMNS ====
        col1, col2 = st.columns(2)

        # ==== LEFT COLUMN ====
        with col1:
            st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src="{target_client}" width="250" height="250" style="border-radius:10px;">
        </div>
        """,
        unsafe_allow_html=True
    )

            render_info_table("👤 Lead Identification & Contact Details", {
        "Business Group":row["Business Group"],
        "Lead Priority":row["Lead Priority"],
        "Client Name":row["Client Name"],
        "Designation": row["Designation"],
        "Location (from teams)": row["Location (from teams)"],
        "Email address": row["Email address"],
        "LinkedIn URL": row["LinkedIn URL"]
    }, name_for_link=row["Client Name"])

            render_info_table("📬 Engagement & Outreach Strategy", {
                "Scope of work/Priorities (internal research)": row["Scope of work/Priorities (internal research)"],
                "Additional Research (External)": row["Additional Research (External)"],
                "MathCo LinkedIn Connects": row["MathCo LinkedIn Connects"],
                "Introduction Path": row["Introduction Path"],
                "Pursured in past": row["Pursured in past"],
                "Relationship Strength": row["Relationship Strength"],
                # "Lead Status": row["Lead Status"],
                "Lead Potential ESS": row["Lead Potential ESS (func. Of designation & Vendor Count)"],
                "Lead Potential DAC": row["Lead Potential DAC (func. Of designation & Vendor Count)"],
                # "Scope of work/Priorities": row["Scope of work/Priorities (internal research)"],
                "If Yes, background/context ?": row["If Yes, background/context ?"],
                "Comments": row["Comments"],
                
            })

        # ==== RIGHT COLUMN ====
        with col2:
            render_info_table("🏢 Company & Department Info", {
                "Business Segment": row["Business Segment"],
                "Working Group": row["Working Group"],
                "Business Functions": row["Business Functions"],

            })

            render_info_table("🧑‍🤝‍🧑 Organizational Hierarchy", {
                "1st Degree Manager": row["1st degree Manager"],
                "2nd Degree Manager": row["2nd Degree Manager"]
            })

            render_info_table("📊 Lead Status & Tracking", {
                "Who will reach out ?": row["Who will reach out ?"],
                "Lever for Reach out(s) ready (Cold email/LinkedIn Message/Demos/PoVs etc.) ?": row["Lever for Reach out(s) ready (Cold email/LinkedIn Message/Demos/PoVs etc.) ?"],
                "Lead Status": row["Lead Status"]
            })
            render_info_table("🧠 Expertise & Experience", {
                "Designation Seniority": row["Designation Seniority"],
                "Location (From LinkedIn)": row["Location (from LinkedIn)"]
            })

            render_info_table("📦 Contractor Information", {
                "Contractor count": (row["Contractor Count"]),
                "Vendor Company Name": (row["Vendor CompanyName"])
            })

    st.markdown("---")
    # === Generate KPIs Button ===
    if st.button(f"**Generate KPIs {selected_client}**"):
        with st.spinner("Generating persona KPIs..."):
            # Assuming df_filtered is already filtered based on user selection
            enriched_stakeholders_df = enrich_with_manager_metadata(filtered_df,selected_client)

            final_df = merge_with_json_snapshots(enriched_stakeholders_df, folder_path=snapshot_folder)
            
            final_df=final_df.fillna("-").replace("", "-")

            description=final_df["description_html"].iloc[0]
                
            kpi_results = run_kpi_analysis(final_df, sample_limit=3)


            kpi_grouped_df = flatten_kpi_results(kpi_results)


            final_df = map_kpis_to_use_cases(kpi_grouped_df, kpi_source_truth)


            # Group by 'Client Name' and 'Use Case'
            grouped_df = final_df.groupby(['Client Name', 'Use Case'], as_index=False).agg({
                'LinkedIn URL': 'first',
                'Business Group': 'first',
                'Client KPI': lambda x: "; ".join(x.dropna().unique()),
                'Strategic KPIs': lambda x: "; ".join(x.dropna().unique()),
                'MathCo Case Study': 'first'
            })

            # Sample subset for demo (you can remove [:2] to run full batch)
            results = []
            for _, row in grouped_df.iterrows():
                result = get_kpi_alignment_response(row["Client KPI"], row["Strategic KPIs"])
                results.append(result)

            # Create DataFrame from LLM results
            results_df = pd.DataFrame(results)

            # Combine back with original grouped_df
            grouped_df = grouped_df.reset_index(drop=True)
            grouped_df["KPI Match Rank"] = results_df["score"]

            # Final result
            grouped_df=grouped_df[["Client Name", "Use Case","Client KPI","Strategic KPIs", "KPI Match Rank","MathCo Case Study"]]

            top_df, all_top_kpis = extract_top_kpis(grouped_df)

            # ==== DEFINE COLUMNS ====
            col3, col4 = st.columns(2)

            with col3:
                render_info_table("🔗 LinkeDin Information", {
                        "Description": (description)
                    })

            with col4:
                render_info_table("🔗 LinkeDin Information", {
                        "Persona KPIs": (all_top_kpis),
                        "MathCO Case study":(top_df['MathCo Case Study'].iloc[0])
                    })














