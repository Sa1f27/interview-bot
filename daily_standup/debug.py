#!/usr/bin/env python3
"""
Debug script to check MongoDB collection structure and data
Run this to understand what's in your database
"""

import pymongo
from urllib.parse import quote_plus
import json
from datetime import datetime

# MongoDB credentials
MONGO_USER = "LanTech"
MONGO_PASS = "L@nc^ere@0012"
MONGO_HOST = "192.168.48.201:27017"
MONGO_DB_NAME = "Api-1"
MONGO_AUTH_SOURCE = "admin"

def debug_mongodb():
    """Debug MongoDB collections and structure"""
    
    connection_string = f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASS)}@{MONGO_HOST}/{MONGO_DB_NAME}?authSource={MONGO_AUTH_SOURCE}"
    
    try:
        print("🔌 Connecting to MongoDB...")
        client = pymongo.MongoClient(connection_string)
        db = client[MONGO_DB_NAME]
        
        # Test connection
        client.admin.command('ismaster')
        print("✅ Connection successful!")
        
        # List all collections
        collections = db.list_collection_names()
        print(f"\n📚 Available collections: {collections}")
        
        # Focus on transcripts collection
        transcripts = db["original-1"]
        
        print(f"\n🔍 Analyzing 'original-1' collection:")
        
        # Count documents
        total_docs = transcripts.count_documents({})
        print(f"  📊 Total documents: {total_docs}")
        
        if total_docs == 0:
            print("  ❌ No documents found in collection!")
            
            # Check if there are any collections with data
            print("\n🔍 Checking other collections for data...")
            for collection_name in collections:
                coll = db[collection_name]
                count = coll.count_documents({})
                print(f"  - {collection_name}: {count} documents")
                
                if count > 0:
                    sample = coll.find_one({})
                    if sample:
                        print(f"    Sample fields: {list(sample.keys())}")
            
            return
        
        # Get sample documents
        print(f"\n📄 Sample documents:")
        sample_docs = list(transcripts.find({}).limit(3))
        
        for i, doc in enumerate(sample_docs, 1):
            print(f"\n  Document {i}:")
            print(f"    ID: {doc.get('_id', 'No ID')}")
            
            # Show all fields and their types
            for key, value in doc.items():
                if key == '_id':
                    continue
                    
                value_type = type(value).__name__
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    print(f"    {key}: {value_type} ({len(value)} chars) = {preview}")
                elif isinstance(value, (int, float)):
                    print(f"    {key}: {value_type} = {value}")
                elif isinstance(value, dict):
                    print(f"    {key}: {value_type} with keys: {list(value.keys())}")
                elif isinstance(value, list):
                    print(f"    {key}: {value_type} with {len(value)} items")
                else:
                    print(f"    {key}: {value_type} = {str(value)[:50]}...")
        
        # Check for summary field specifically
        print(f"\n🔍 Checking for 'summary' field:")
        summary_docs = transcripts.count_documents({"summary": {"$exists": True}})
        print(f"  📊 Documents with 'summary' field: {summary_docs}")
        
        if summary_docs > 0:
            # Get the latest document with summary
            latest_summary_doc = transcripts.find_one(
                {"summary": {"$exists": True}}, 
                sort=[("timestamp", -1)]
            )
            if latest_summary_doc:
                summary = latest_summary_doc.get("summary", "")
                print(f"  ✅ Latest summary found ({len(summary)} chars)")
                print(f"  Preview: {summary[:200]}...")
            else:
                print("  ❌ Could not retrieve summary document")
        
        # Check for alternative fields
        print(f"\n🔍 Checking for alternative content fields:")
        alternative_fields = ["content", "text", "transcript", "lecture_content", "data"]
        
        for field in alternative_fields:
            count = transcripts.count_documents({field: {"$exists": True}})
            if count > 0:
                print(f"  ✅ Found {count} documents with '{field}' field")
                sample_doc = transcripts.find_one({field: {"$exists": True}})
                if sample_doc and field in sample_doc:
                    content = sample_doc[field]
                    if isinstance(content, str):
                        print(f"    Sample: {content[:100]}...")
            else:
                print(f"  ❌ No documents with '{field}' field")
        
        # Check timestamps
        print(f"\n📅 Checking timestamps:")
        timestamp_docs = transcripts.count_documents({"timestamp": {"$exists": True}})
        print(f"  📊 Documents with 'timestamp' field: {timestamp_docs}")
        
        if timestamp_docs > 0:
            # Get latest and oldest timestamps
            latest = transcripts.find_one({}, sort=[("timestamp", -1)])
            oldest = transcripts.find_one({}, sort=[("timestamp", 1)])
            
            if latest and "timestamp" in latest:
                latest_time = datetime.fromtimestamp(latest["timestamp"]) if isinstance(latest["timestamp"], (int, float)) else latest["timestamp"]
                print(f"  📅 Latest: {latest_time}")
                
            if oldest and "timestamp" in oldest:
                oldest_time = datetime.fromtimestamp(oldest["timestamp"]) if isinstance(oldest["timestamp"], (int, float)) else oldest["timestamp"]
                print(f"  📅 Oldest: {oldest_time}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mongodb()