import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging

logger = logging.getLogger(__name__)

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

def save_interview_summary(applicant_id: str, summary: str) -> bool:
    """
    Saves the interview evaluation summary to the applicants table.
    
    Args:
        applicant_id: The ID of the applicant
        summary: The generated interview evaluation text
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE applicants 
                    SET calls_summary = %s 
                    WHERE id = %s;
                """, (summary, applicant_id))
                conn.commit()
                logger.info(f"✅ Saved interview summary for applicant {applicant_id}")
                return True
    except Exception as e:
        logger.error(f"❌ Error saving interview summary: {e}")
        return False