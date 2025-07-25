from pymongo import MongoClient, DESCENDING

# Connection URI
uri = "mongodb://connectly:LT%40connect25@192.168.48.201:27017/connectlydb?authSource=connectlydb"

# The name of the collection you want to query.
collection_name = "original-1"  # TODO: Update this to your target collection name

# Connect and check
try:
    client = MongoClient(uri)
    client.admin.command('ping') # Check if the connection is successful

    # The database is inferred from the URI
    db = client.get_default_database()
    collection = db[collection_name]

    print(f"✅ Connection to {db.name} successful.")
    print(f"🔍 Checking latest summary in collection: '{collection_name}' ...")

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
finally:
    if 'client' in locals():
        client.close()
        print("\nConnection closed.")
