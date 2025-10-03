import psycopg2
import pandas as pd

# ==== Database Connection Details ====
DB_HOST = "psql-scout.postgres.database.azure.com"   # e.g. mydb.postgres.database.azure.com
DB_NAME = "stakeholder360"
DB_USER = "mathcoadmin"   # For Azure use: user@servername
DB_PASS = "Shubham@123"
DB_PORT = 5432

# ==== Load Excel Data ====
excel_file = r"C:\Users\ShubhamVishwasPurani\OneDrive - TheMathCompany Private Limited\Desktop\SCOUT Automation Codes\kellanova_linkedin_data.xlsx"   # <-- change to your actual file
df = pd.read_excel(excel_file)  # adjust sheet name if needed

# Keep only required columns (must match table schema)
df = df[[
    "client_id",
    "client_name",
    "client_city",
    "client_about",
    "client_present_title",
    "client_present_description_html",
    "client_url",
    "client_avatar",
    "client_linkedin_id"
]]

# Convert dataframe to list of tuples
rows = list(df.itertuples(index=False, name=None))

# ==== Insert into PostgreSQL ====
conn = psycopg2.connect(
    host=DB_HOST,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    port=DB_PORT,
    sslmode="require"   # Azure PostgreSQL requires SSL
)

cur = conn.cursor()

insert_sql = """
INSERT INTO scout.linkedin_clients_data
(client_id, client_name, client_city, client_about, client_present_title,
 client_present_description_html, client_url, client_avatar, client_linkedin_id)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

cur.executemany(insert_sql, rows)

conn.commit()
cur.close()
conn.close()

print("✅ Data inserted successfully into scout.linkedin_clients_data!")
