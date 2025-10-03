# utils/db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL
from dotenv import load_dotenv

load_dotenv()

def get_engine() -> Engine:
    pg_user = os.getenv("PGUSER")
    pg_password = os.getenv("PGPASSWORD")
    pg_host = os.getenv("PGHOST")
    pg_port = os.getenv("PGPORT", "5432")
    pg_db = os.getenv("PGDATABASE")

    if not all([pg_user, pg_password, pg_host, pg_db]):
        raise RuntimeError("❌ Database configuration is incomplete. Check .env values.")

    # Build a properly quoted URL (handles special chars like '@' in password)
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=pg_user,
        password=pg_password,   # safely quoted by SQLAlchemy
        host=pg_host,
        port=int(pg_port),
        database=pg_db,
        query={"sslmode": "require"},  # Azure Postgres SSL
    )

    engine = create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        echo=False,
    )
    return engine
