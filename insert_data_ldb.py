import psycopg2
import pandas as pd
import numpy as np   # needed for type checks

# ==== Database Connection Details ====
DB_HOST = "psql-scout.postgres.database.azure.com"
DB_NAME = "stakeholder360"
DB_USER = "mathcoadmin"
DB_PASS = "Shubham@123"
DB_PORT = 5432

# ==== Load Excel Data ====
excel_file = r"C:\Users\ShubhamVishwasPurani\OneDrive - TheMathCompany Private Limited\Desktop\Stakeholder_360\Kellenova Standardize LDB.xlsx"
df = pd.read_excel(excel_file)

# Rename Excel → snake_case
df = df.rename(columns={
    "Account": "account",
    "File Name": "file_name",
    "Sr. No.": "sr_no",
    "Input Date": "input_date",
    "Last Update Date": "last_update_date",
    "Subsidiary": "subsidiary",
    "Working Group": "working_group",
    "Business Unit": "business_unit",
    "Lead Priority": "lead_priority",
    "Client Name": "client_name",
    "Client Designation": "client_designation",
    "Seniority Level": "seniority_level",
    "Service Line": "service_line",
    "CSL Owner": "csl_owner",
    "Reporting Manager": "reporting_manager",
    "Reporting Manager Designation": "reporting_manager_designation",
    "Email Address": "email_address",
    "LinkedIn URL": "linkedin_url",
    "Location": "location",
    "Internal Research": "internal_research",
    "External Research": "external_research",
    "Personalization Notes": "personalization_notes",
    "Vendor Name": "vendor_name",
    "Contractor Count": "contractor_count",
    "Reachout Lever": "reachout_lever",
    "Reachout Channel": "reachout_channel",
    "Pursued In Past": "pursued_in_past",
    "Context": "context",
    "Introduction Path": "introduction_path",
    "MathCo SPOC 1": "mathco_spoc_1",
    "MathCo SPOC 2": "mathco_spoc_2",
    "MathCo SPOC 3": "mathco_spoc_3",
    "Scout Linkedin Connected Flag": "scout_linkedin_connected_flag",
    "First Outreach Date": "first_outreach_date",
    "Last Outreach Date": "last_outreach_date",
    "Recent Role Change (Last 3 months)": "recent_role_change_last_3_months",
    "Intel Link": "intel_link",
    "KPI": "kpi",
    "Intel Summary": "intel_summary",
    "Email Template": "email_template",
    "Impact Pointers": "impact_pointers",
    "Status": "status",
    "Comments": "comments"
})

# ==== Clean DataFrame ====
df = df.where(pd.notnull(df), None)  # NaN → None (NULL in DB)

# Force Python-native ints but keep NULL
if "sr_no" in df.columns:
    df["sr_no"] = df["sr_no"].apply(lambda x: int(x) if pd.notnull(x) else None)

if "contractor_count" in df.columns:
    df["contractor_count"] = df["contractor_count"].apply(lambda x: int(x) if pd.notnull(x) else None)

# Convert rows into Python tuples
rows = [tuple(x if x is not pd.NaT else None for x in row) for row in df.to_numpy()]

# ==== Insert into PostgreSQL ====
conn = psycopg2.connect(
    host=DB_HOST,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    port=DB_PORT,
    sslmode="require"
)
cur = conn.cursor()

insert_sql = """
INSERT INTO scout.centralize_db
(account, file_name, sr_no, input_date, last_update_date, subsidiary, working_group, business_unit,
 lead_priority, client_name, client_designation, seniority_level, service_line, csl_owner, reporting_manager,
 reporting_manager_designation, email_address, linkedin_url, location, internal_research, external_research,
 personalization_notes, vendor_name, contractor_count, reachout_lever, reachout_channel, pursued_in_past,
 context, introduction_path, mathco_spoc_1, mathco_spoc_2, mathco_spoc_3, scout_linkedin_connected_flag,
 first_outreach_date, last_outreach_date, recent_role_change_last_3_months, intel_link, kpi, intel_summary,
 email_template, impact_pointers, status, comments)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

cur.executemany(insert_sql, rows)

conn.commit()
cur.close()
conn.close()

print("✅ Data inserted successfully into scout.centralize_db!")
