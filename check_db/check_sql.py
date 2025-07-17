import pyodbc
import random

# --- Configuration ---
# Copied from your main.py for consistency
DB_CONFIG = {
    "DRIVER": "ODBC Driver 17 for SQL Server",
    "SERVER": "183.82.108.211",
    "DATABASE": "SuperDB",
    "UID": "Connectly",
    "PWD": "LT@connect25",
}

CONNECTION_STRING = (
    f"DRIVER={{{DB_CONFIG['DRIVER']}}};"
    f"SERVER={DB_CONFIG['SERVER']};"
    f"DATABASE={DB_CONFIG['DATABASE']};"
    f"UID={DB_CONFIG['UID']};"
    f"PWD={DB_CONFIG['PWD']}"
)

def check_sql_server_connection():
    """
    Connects to the SQL Server, fetches a few student records, and prints the status.
    """
    print("--- SQL Server Connection Test ---")
    
    try:
        # 1. Attempt to connect to the database
        print(f"Attempting to connect to {DB_CONFIG['SERVER']}...")
        conn = pyodbc.connect(CONNECTION_STRING, timeout=5)
        print("✅ Connection Successful!")
        
        # 2. Attempt to fetch data
        print("\nFetching a few student records...")
        cursor = conn.cursor()
        
        # Execute a query to get the top 5 valid student records
        cursor.execute("""
            SELECT TOP 5 ID, First_Name, Last_Name 
            FROM tbl_Student 
            WHERE ID IS NOT NULL AND First_Name IS NOT NULL AND Last_Name IS NOT NULL
        """)
        
        records = cursor.fetchall()
        
        if not records:
            print("⚠️  Warning: Connection was successful, but no student records were found.")
        else:
            print(f"✅ Data Fetched Successfully! Found {len(records)} records.")
            for row in records:
                print(f"  - ID: {row.ID}, Name: {row.First_Name} {row.Last_Name}")

    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"❌ DATABASE ERROR: {sqlstate}")
        print(f"   Details: {ex}")
    except Exception as e:
        print(f"❌ AN UNEXPECTED ERROR OCCURRED: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("\nConnection closed.")
        print("--- Test Complete ---")

if __name__ == "__main__":
    check_sql_server_connection()
