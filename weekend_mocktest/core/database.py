# weekend_mocktest/core/database.py
import logging
import time
import pymongo
import pyodbc
from typing import List, Dict, Any, Optional
from .config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager with real MongoDB integration and SQL Server fallback"""
    
    def __init__(self):
        """Initialize database connections"""
        logger.info("?? Initializing Database Manager")
        
        self.use_dummy_sql = config.USE_DUMMY_DATA  # Only SQL Server uses dummy mode now
        
        # MongoDB is always live now (server is up)
        self.mongo_client = None
        self.db = None
        self.summaries_collection = None
        self.test_results_collection = None
        
        # SQL Server connection string
        self.sql_connection_string = config.SQL_CONNECTION_STRING
        
        # Initialize MongoDB connection (real connection)
        self._init_mongodb()
        
        if self.use_dummy_sql:
            logger.info("?? SQL Server in dummy mode (server unavailable)")
            self._init_dummy_sql_data()
        else:
            logger.info("?? SQL Server in live mode")
    
    def _init_mongodb(self):
        """Initialize MongoDB connection - REAL CONNECTION"""
        try:
            # Use the actual MongoDB connection from your working script
            self.mongo_client = pymongo.MongoClient(
                config.MONGO_CONNECTION_STRING,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=10,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000
            )
            
            # Test connection
            self.mongo_client.admin.command('ping')
            
            # Initialize database and collections
            self.db = self.mongo_client[config.MONGO_DB_NAME]
            self.summaries_collection = self.db[config.SUMMARIES_COLLECTION]
            self.test_results_collection = self.db[config.TEST_RESULTS_COLLECTION]
            
            # Create indexes for performance
            try:
                self.test_results_collection.create_index("test_id")
                self.test_results_collection.create_index("timestamp")
                self.summaries_collection.create_index("timestamp")
                logger.info("? Database indexes created")
            except Exception as idx_error:
                logger.warning(f"?? Index creation failed: {idx_error}")
            
            logger.info("? MongoDB connection established (LIVE MODE)")
            
            # Test collection access
            summary_count = self.summaries_collection.count_documents(
                {"summary": {"$exists": True, "$ne": ""}}
            )
            logger.info(f"?? Found {summary_count} summaries in collection '{config.SUMMARIES_COLLECTION}'")
            
        except Exception as e:
            logger.error(f"? MongoDB connection failed: {e}")
            raise Exception(f"MongoDB connection failure: {e}")
    
    def _init_dummy_sql_data(self):
        """Initialize dummy SQL data for when SQL Server is down"""
        self.dummy_students = [
            {"ID": 1001, "First_Name": "John", "Last_Name": "Doe"},
            {"ID": 1002, "First_Name": "Jane", "Last_Name": "Smith"}, 
            {"ID": 1003, "First_Name": "Alice", "Last_Name": "Johnson"},
            {"ID": 1004, "First_Name": "Bob", "Last_Name": "Wilson"},
            {"ID": 1005, "First_Name": "Carol", "Last_Name": "Brown"}
        ]
        
        self.dummy_sessions = [
            {"Session_ID": "session_001"},
            {"Session_ID": "session_002"},
            {"Session_ID": "session_003"},
            {"Session_ID": "session_004"},
            {"Session_ID": "session_005"}
        ]
        
        logger.info("?? Dummy SQL data initialized")
    
    def get_recent_summaries(self, limit: int = None) -> List[Dict[str, Any]]:
        """Fetch recent summaries from MongoDB - REAL DATA"""
        if limit is None:
            limit = config.RECENT_SUMMARIES_COUNT
        
        try:
            logger.info(f"?? Fetching {limit} recent summaries from MongoDB")
            
            # Query using the same pattern as your working script
            cursor = self.summaries_collection.find(
                {"summary": {"$exists": True, "$ne": ""}},
                {"summary": 1, "timestamp": 1, "date": 1, "session_id": 1, "_id": 1}
            ).sort("_id", pymongo.DESCENDING).limit(limit)
            
            summaries = list(cursor)
            
            if not summaries:
                raise Exception("No summaries found in database")
            
            logger.info(f"? Retrieved {len(summaries)} recent summaries from MongoDB")
            
            # Log a sample of the first summary for verification
            if summaries:
                first_summary = summaries[0]["summary"]
                sample_text = first_summary[:200] + "..." if len(first_summary) > 200 else first_summary
                logger.info(f"?? Sample summary: {sample_text}")
            
            return summaries
            
        except Exception as e:
            logger.error(f"? Failed to fetch summaries from MongoDB: {e}")
            raise Exception(f"Summary retrieval failed: {e}")
    
    def save_test_results(self, test_id: str, test_data: Dict[str, Any], 
                         evaluation_result: Dict[str, Any]) -> bool:
        """Save test results to MongoDB"""
        logger.info(f"?? Saving test results to MongoDB: {test_id}")
        
        try:
            # Get student info (SQL Server or dummy)
            student_id, first_name, last_name, session_id = self.fetch_student_info()
            name = f"{first_name} {last_name}"
            
            # Create conversation pairs from answers
            conversation_pairs = []
            for i, answer_data in enumerate(test_data.get("answers", []), 1):
                conversation_pairs.append({
                    "question_number": i,
                    "question": answer_data.get("question", ""),
                    "answer": answer_data.get("answer", ""),
                    "correct": answer_data.get("correct", False),
                    "feedback": answer_data.get("feedback", "")
                })
            
            # Calculate score percentage
            score_percentage = round(
                (evaluation_result["total_correct"] / test_data["total_questions"]) * 100, 1
            )
            
            # Create clean document structure for MongoDB
            document = {
                "test_id": test_id,
                "timestamp": time.time(),
                "Student_ID": student_id,
                "name": name,
                "session_id": session_id,
                "user_type": test_data["user_type"],
                "score": evaluation_result["total_correct"],
                "total_questions": test_data["total_questions"],
                "score_percentage": score_percentage,
                "evaluation_report": evaluation_result["evaluation_report"],
                "conversation_pairs": conversation_pairs,
                "test_completed": True
            }
            
            # Save to MongoDB
            result = self.test_results_collection.insert_one(document)
            
            if not result.inserted_id:
                raise Exception("MongoDB save operation failed")
            
            logger.info(f"? Test saved successfully to MongoDB: {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"? Save failed: {e}")
            raise Exception(f"Results save failed: {e}")
    
    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get test results by test ID from MongoDB"""
        try:
            logger.info(f"?? Fetching test results from MongoDB: {test_id}")
            
            doc = self.test_results_collection.find_one(
                {"test_id": test_id}, 
                {"_id": 0}
            )
            
            if not doc:
                logger.warning(f"No test results found for test_id: {test_id}")
                return None
            
            result = {
                "test_id": test_id,
                "score": doc.get("score", 0),
                "total_questions": doc.get("total_questions", 0),
                "score_percentage": doc.get("score_percentage", 0),
                "analytics": doc.get("evaluation_report", "Report not available"),
                "timestamp": doc.get("timestamp", 0),
                "pdf_available": True
            }
            
            logger.info(f"? Test results retrieved from MongoDB: {test_id}")
            return result
            
        except Exception as e:
            logger.error(f"? Failed to get test results from MongoDB: {e}")
            raise Exception(f"Test results retrieval failed: {e}")
    
    def get_all_test_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all test results from MongoDB with pagination"""
        try:
            logger.info(f"?? Fetching all test results from MongoDB (limit: {limit})")
            
            results = list(self.test_results_collection.find(
                {},
                {"_id": 0, "test_id": 1, "name": 1, "score": 1, "total_questions": 1, 
                 "score_percentage": 1, "timestamp": 1, "user_type": 1}
            ).sort("timestamp", pymongo.DESCENDING).limit(limit))
            
            logger.info(f"? Retrieved {len(results)} test results from MongoDB")
            return results
            
        except Exception as e:
            logger.error(f"? Failed to get all test results from MongoDB: {e}")
            raise Exception(f"Test results retrieval failed: {e}")
    
    def get_student_list(self) -> List[Dict[str, Any]]:
        """Get unique students from test results in MongoDB"""
        try:
            logger.info("?? Fetching student list from MongoDB")
            
            pipeline = [
                {"$group": {"_id": "$Student_ID", "name": {"$first": "$name"}}},
                {"$project": {"_id": 0, "Student_ID": "$_id", "name": 1}}
            ]
            students = list(self.test_results_collection.aggregate(pipeline))
            
            logger.info(f"? Retrieved {len(students)} students from MongoDB")
            return students
            
        except Exception as e:
            logger.error(f"? Failed to get student list from MongoDB: {e}")
            raise Exception(f"Student list retrieval failed: {e}")
    
    def get_student_tests(self, student_id: str) -> List[Dict[str, Any]]:
        """Get tests for specific student from MongoDB"""
        try:
            logger.info(f"?? Fetching tests for student {student_id} from MongoDB")
            
            results = list(self.test_results_collection.find(
                {"Student_ID": int(student_id)},
                {"_id": 0, "qa_details": 0, "question_types": 0}
            ).sort("timestamp", pymongo.DESCENDING))
            
            logger.info(f"? Retrieved {len(results)} tests for student {student_id}")
            return results
            
        except Exception as e:
            logger.error(f"? Failed to get student tests from MongoDB: {e}")
            raise Exception(f"Student tests retrieval failed: {e}")
    
    def fetch_student_info(self):
        """Fetch student info from SQL Server or dummy data"""
        if self.use_dummy_sql:
            return self._get_dummy_student_info()
        
        try:
            logger.info("?? Fetching student info from SQL Server")
            
            conn = pyodbc.connect(self.sql_connection_string, timeout=10)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT TOP 1 s.ID, s.First_Name, s.Last_Name, ses.Session_ID
                FROM tbl_Student s 
                CROSS JOIN (SELECT TOP 1 Session_ID FROM tbl_Session ORDER BY NEWID()) ses
                WHERE s.ID IS NOT NULL AND s.First_Name IS NOT NULL AND s.Last_Name IS NOT NULL
                ORDER BY NEWID()
            """)
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not result:
                raise Exception("No valid student data found in SQL Server")
            
            student_id, first_name, last_name, session_id = result
            
            # Validate data quality
            if not all([student_id, first_name, last_name, session_id]):
                raise Exception("Incomplete student data retrieved from SQL Server")
            
            logger.info(f"? Student info retrieved from SQL Server: {student_id}")
            return student_id, first_name, last_name, session_id
            
        except Exception as e:
            logger.error(f"? SQL Server fetch failed, using dummy data: {e}")
            return self._get_dummy_student_info()
    
    def _get_dummy_student_info(self):
        """Get dummy student info when SQL Server is unavailable"""
        import random
        
        student = random.choice(self.dummy_students)
        session = random.choice(self.dummy_sessions)
        
        return (
            student["ID"],
            student["First_Name"], 
            student["Last_Name"],
            session["Session_ID"]
        )
    
    def validate_connection(self) -> Dict[str, Any]:
        """Validate database connections"""
        status = {
            "mongodb": False,
            "sql_server": False,
            "collections_accessible": False,
            "overall": False,
            "mode": "hybrid"  # MongoDB live, SQL may be dummy
        }
        
        try:
            # Test MongoDB (always live now)
            self.mongo_client.admin.command('ping')
            status["mongodb"] = True
            
            # Test MongoDB collections
            summary_count = self.summaries_collection.count_documents({}, limit=1)
            status["collections_accessible"] = True
            logger.info(f"? MongoDB accessible with {summary_count} documents")
            
        except Exception as e:
            logger.error(f"? MongoDB validation failed: {e}")
        
        try:
            # Test SQL Server if not in dummy mode
            if not self.use_dummy_sql:
                conn = pyodbc.connect(self.sql_connection_string, timeout=5)
                conn.close()
                status["sql_server"] = True
                logger.info("? SQL Server accessible")
            else:
                status["sql_server"] = True  # Dummy mode is considered "working"
                logger.info("? SQL Server in dummy mode")
                
        except Exception as e:
            logger.error(f"? SQL Server validation failed: {e}")
        
        # Overall status - MongoDB is critical, SQL can be dummy
        status["overall"] = status["mongodb"] and status["collections_accessible"]
        
        return status
    
    def close(self):
        """Close database connections"""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("? Database connections closed")

# Singleton pattern for database manager
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get database manager instance (singleton)"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def close_db_manager():
    """Close database manager instance"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None