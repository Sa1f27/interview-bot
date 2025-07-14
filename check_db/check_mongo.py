from pymongo import MongoClient, DESCENDING
from urllib.parse import quote_plus

# Credentials
user = quote_plus("LanTech")
password = quote_plus("L@nc^ere@0012")
host = "192.168.48.201:27017"
auth_source = "admin"
db_name = "Api-1"
collection_name = "original-1"

# Connection URI
uri = f"mongodb://{user}:{password}@{host}/{db_name}?authSource={auth_source}"

# Connect and check
try:
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]

    print(f"🔍 Checking latest summary in {db_name}.{collection_name} ...")

    doc = collection.find_one(
        {"summary": {"$exists": True, "$ne": ""}},
        sort=[("_id", DESCENDING)]
    )

    if doc:
        print("✅ Found summary:")
        print("-" * 50)
        print(doc["summary"])
        print("-" * 50)
    else:
        print("⚠️ No valid summary documents found.")

except Exception as e:
    print(f"❌ ERROR: {e}")
