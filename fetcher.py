import psycopg2
from psycopg2.extras import RealDictCursor
import textwrap

DB_URL = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"

def connect_db():
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)

def print_summary(text, width=100):
    """Shortens long text blocks for console readability."""
    if not text:
        return "—"
    text = str(text).replace("\n", " ")
    if len(text) > width:
        text = text[:width] + "..."
    return text

def fetch_applicants(limit=5):
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, email, score, ai_data, created_at
                FROM applicants
                ORDER BY created_at DESC
                LIMIT %s;
            """, (limit,))
            return cur.fetchall()

def main():
    try:
        applicants = fetch_applicants(limit=5)
        print("\n=== Recent Applicants ===\n")

        for i, app in enumerate(applicants, 1):
            print(f"{i}. {app['name']} ({app['email']})")
            print(f"   Score: {app.get('score', 'N/A')}")
            print(f"   Created: {app['created_at']}")
            print(f"   AI Summary: {print_summary(app.get('ai_data'))}")
            print("-" * 120)

    except Exception as e:
        print("❌ Error fetching applicants:", e)

if __name__ == "__main__":
    main()
