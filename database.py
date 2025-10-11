import psycopg2
from psycopg2.extras import RealDictCursor
import os

# Constants
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres")

def connect_db():
    """Establishes a connection to the database."""
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)

def fetch_latest_applicant():
    """Fetches the most recent applicant with non-null ai_data from the database."""
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, email, score, ai_data, created_at
                FROM applicants
                WHERE ai_data IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1;
            """)
            applicant = cur.fetchone()
            return applicant