from pathlib import Path
from sqlalchemy import text
from utils.db import get_engine

def run_sql(path: Path):
    sql = path.read_text(encoding="utf-8")
    with get_engine().begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))

if __name__ == "__main__":
    base = Path(__file__).parent / "sql"
    run_sql(base / "01_create_schema.sql")
    run_sql(base / "02_create_tables.sql")
    print("✅ Schema & tables ensured.")
