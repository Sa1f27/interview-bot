# weekend_mocktest/core/database.py
import logging
import time
import pymongo
import pyodbc
from typing import List, Dict, Any, Optional
from .config import config
from .dummy_data import get_dummy_data_service

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager with dummy data fallback"""
    
    def __init__(self):
        """Initialize database connections with fallback to dummy data"""
        logger.info("🚀 Initializing Database Manager")
        
        self.use_dummy = config.USE_DUMMY_DATA
        
        if self.use_dummy:
            logger.info("🔧 Using dummy data (server unavailable)")
            self.dummy_service = get_dummy_data_service()
            self._validate_dummy_data()
        else:
            logger.info("🔗 Connecting to live databases")
            self._init_live_connections()
    
    def _validate_dummy_data(self):
        """Validate dummy data availability"""
        if not self.dummy_service.validate_dummy_data():
            raise Exception("Dummy data validation failed")
        logger.info("✅ Dummy data validated successfully")
    
    def _init_live_connections(self):
        """Initialize live database connections"""
        # MongoDB connection
        self.mongo_client = None
        self.db = None
        self.summaries_collection = None
        self.test_results_collection = None
        
        # SQL Server connection string
        self.sql_connection_string = config.SQL_CONNECTION_STRING
        
        self._init_mongodb()
    
    def _init_mongodb(self):
        """Initialize MongoDB connection with validation"""
        try:
            self.mongo_client = pymongo.MongoClient(
                config.MONGO_CONNECTION_STRING,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=10,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000
            )
            
            # Test connection
            self.mongo_client.server_info()
            
            # Initialize database and collections
            self.db = self.mongo_client[config.MONGO_DB_NAME]
            self.summaries_collection = self.db[config.SUMMARIES_COLLECTION]
            self.test_results_collection = self.db[config.TEST_RESULTS_COLLECTION]
            
            # Create indexes for performance
            try:
                self.test_results_collection.create_index("test_id")
                self.test_results_collection.create_index("timestamp")
                self.summaries_collection.create_index("timestamp")
                logger.info("✅ Database indexes created")
            except Exception as idx_error:
                logger.warning(f"⚠️ Index creation failed: {idx_error}")
            
            logger.info("✅ MongoDB connection established")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise Exception(f"MongoDB connection failure: {e}")
    
    def get_recent_summaries(self, limit: int = None) -> List[Dict[str, Any]]:
        """Fetch recent summaries from database or dummy data"""
        if limit is None:
            limit = config.RECENT_SUMMARIES_COUNT
        
        if self.use_dummy:
            logger.info(f"📄 Using dummy summaries (limit: {limit})")
            return self.dummy_service.get_recent_summaries(limit)
        
        try:
            cursor = self.summaries_collection.find(
                {"summary": {"$exists": True, "$ne": ""}},
                {"summary": 1, "timestamp": 1, "date": 1, "session_id": 1}
            ).sort("timestamp", -1).limit(limit)
            
            summaries = list(cursor)
            
            if not summaries:
                raise Exception("No summaries found in database")
            
            logger.info(f"✅ Retrieved {len(summaries)} recent summaries from MongoDB")
            return summaries
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch summaries: {e}")
            raise Exception(f"Summary retrieval failed: {e}")
    
    def save_test_results(self, test_id: str, test_data: Dict[str, Any], 
                         evaluation_result: Dict[str, Any]) -> bool:
        """Save test results to database or simulate save for dummy data"""
        logger.info(f"💾 Saving test results: {test_id}")
        
        if self.use_dummy:
            logger.info(f"🔧 Simulating save for dummy data: {test_id}")
            # Simulate successful save for dummy data
            time.sleep(0.1)  # Simulate network delay
            return True
        
        try:
            # Get authentic student info
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
            
            # Create clean document structure
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
            
            # Save with validation
            result = self.test_results_collection.insert_one(document)
            
            if not result.inserted_id:
                raise Exception("Database save operation failed")
            
            logger.info(f"✅ Test saved successfully: {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
            raise Exception(f"Results save failed: {e}")
    
    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get test results by test ID"""
        if self.use_dummy:
            # For dummy data, return simulated results
            return {
                "test_id": test_id,
                "score": 7,
                "total_questions": 10,
                "score_percentage": 70.0,
                "analytics": "This is a simulated test result for dummy data mode.",
                "timestamp": time.time(),
                "pdf_available": True
            }
        
        try:
            doc = self.test_results_collection.find_one(
                {"test_id": test_id}, 
                {"_id": 0}
            )
            
            if not doc:
                return None
            
            return {
                "test_id": test_id,
                "score": doc.get("score", 0),
                "total_questions": doc.get("total_questions", 0),
                "score_percentage": doc.get("score_percentage", 0),
                "analytics": doc.get("evaluation_report", "Report not available"),
                "timestamp": doc.get("timestamp", 0),
                "pdf_available": True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get test results: {e}")
            raise Exception(f"Test results retrieval failed: {e}")
    
    def get_all_test_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all test results with pagination"""
        if self.use_dummy:
            # Return simulated test results for dummy data
            return [
                {
                    "test_id": f"dummy_test_{i}",
                    "name": f"Student {i}",
                    "score": 7 + (i % 4),
                    "total_questions": 10,
                    "score_percentage": (7 + (i % 4)) * 10,
                    "timestamp": time.time() - (i * 3600),
                    "user_type": "dev" if i % 2 == 0 else "non_dev"
                }
                for i in range(1, min(limit + 1, 11))
            ]
        
        try:
            results = list(self.test_results_collection.find(
                {},
                {"_id": 0, "test_id": 1, "name": 1, "score": 1, "total_questions": 1, 
                 "score_percentage": 1, "timestamp": 1, "user_type": 1}
            ).sort("timestamp", -1).limit(limit))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get all test results: {e}")
            raise Exception(f"Test results retrieval failed: {e}")
    
    def get_student_list(self) -> List[Dict[str, Any]]:
        """Get unique students from test results"""
        if self.use_dummy:
            return self.dummy_service.get_all_students()
        
        try:
            pipeline = [
                {"$group": {"_id": "$Student_ID", "name": {"$first": "$name"}}},
                {"$project": {"_id": 0, "Student_ID": "$_id", "name": 1}}
            ]
            students = list(self.test_results_collection.aggregate(pipeline))
            return students
            
        except Exception as e:
            logger.error(f"❌ Failed to get student list: {e}")
            raise Exception(f"Student list retrieval failed: {e}")
    
    def get_student_tests(self, student_id: str) -> List[Dict[str, Any]]:
        """Get tests for specific student"""
        if self.use_dummy:
            # Return simulated student tests
            return [
                {
                    "test_id": f"student_{student_id}_test_{i}",
                    "score": 6 + i,
                    "total_questions": 10,
                    "score_percentage": (6 + i) * 10,
                    "timestamp": time.time() - (i * 7200),
                    "user_type": "dev" if i % 2 == 0 else "non_dev"
                }
                for i in range(1, 4)
            ]
        
        try:
            results = list(self.test_results_collection.find(
                {"Student_ID": int(student_id)},
                {"_id": 0, "qa_details": 0, "question_types": 0}
            ))
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get student tests: {e}")
            raise Exception(f"Student tests retrieval failed: {e}")
    
    def fetch_student_info(self):
        """Fetch student info from SQL Server or dummy data"""
        if self.use_dummy:
            return self.dummy_service.get_random_student_info()
        
        try:
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
                raise Exception("No valid student data found in database")
            
            student_id, first_name, last_name, session_id = result
            
            # Validate data quality
            if not all([student_id, first_name, last_name, session_id]):
                raise Exception("Incomplete student data retrieved")
            
            logger.info(f"✅ Student info retrieved: {student_id}")
            return student_id, first_name, last_name, session_id
            
        except Exception as e:
            logger.error(f"❌ Student info fetch failed: {e}")
            raise Exception(f"Student data unavailable: {e}")
    
    def validate_connection(self) -> Dict[str, Any]:
        """Validate database connections"""
        if self.use_dummy:
            return {
                "mongodb": True,
                "sql_server": True,
                "collections_accessible": True,
                "overall": True,
                "mode": "dummy_data"
            }
        
        status = {
            "mongodb": False,
            "sql_server": False,
            "collections_accessible": False,
            "overall": False,
            "mode": "live_data"
        }
        
        try:
            # Test MongoDB
            self.mongo_client.server_info()
            status["mongodb"] = True
            
            # Test collections
            self.summaries_collection.count_documents({}, limit=1)
            status["collections_accessible"] = True
            
            # Test SQL Server
            conn = pyodbc.connect(self.sql_connection_string, timeout=5)
            conn.close()
            status["sql_server"] = True
            
            status["overall"] = all([
                status["mongodb"],
                status["sql_server"],
                status["collections_accessible"]
            ])
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
        
        return status
    
    def close(self):
        """Close database connections"""
        if not self.use_dummy and hasattr(self, 'mongo_client') and self.mongo_client:
            self.mongo_client.close()
            logger.info("✅ Database connections closed")

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
    
    def _init_mongodb(self):
        """Initialize MongoDB connection with validation"""
        try:
            self.mongo_client = pymongo.MongoClient(
                config.MONGO_CONNECTION_STRING,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=10,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000
            )
            
            # Test connection
            self.mongo_client.server_info()
            
            # Initialize database and collections
            self.db = self.mongo_client[config.MONGO_DB_NAME]
            self.summaries_collection = self.db[config.SUMMARIES_COLLECTION]
            self.test_results_collection = self.db[config.TEST_RESULTS_COLLECTION]
            
            # Create indexes for performance
            try:
                self.test_results_collection.create_index("test_id")
                self.test_results_collection.create_index("timestamp")
                self.summaries_collection.create_index("timestamp")
                logger.info("✅ Database indexes created")
            except Exception as idx_error:
                logger.warning(f"⚠️ Index creation failed: {idx_error}")
            
            logger.info("✅ MongoDB connection established")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise Exception(f"MongoDB connection failure: {e}")
    
    def get_recent_summaries(self, limit: int = None) -> List[Dict[str, Any]]:
        """Fetch recent summaries from database"""
        if limit is None:
            limit = config.RECENT_SUMMARIES_COUNT
        
        try:
            cursor = self.summaries_collection.find(
                {"summary": {"$exists": True, "$ne": ""}},
                {"summary": 1, "timestamp": 1, "date": 1, "session_id": 1}
            ).sort("timestamp", -1).limit(limit)
            
            summaries = list(cursor)
            
            if not summaries:
                raise Exception("No summaries found in database")
            
            logger.info(f"✅ Retrieved {len(summaries)} recent summaries")
            return summaries
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch summaries: {e}")
            raise Exception(f"Summary retrieval failed: {e}")
    
    def save_test_results(self, test_id: str, test_data: Dict[str, Any], 
                         evaluation_result: Dict[str, Any]) -> bool:
        """Save test results to database"""
        logger.info(f"💾 Saving test results: {test_id}")
        
        try:
            # Get authentic student info
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
            
            # Create clean document structure
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
            
            # Save with validation
            result = self.test_results_collection.insert_one(document)
            
            if not result.inserted_id:
                raise Exception("Database save operation failed")
            
            logger.info(f"✅ Test saved successfully: {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
            raise Exception(f"Results save failed: {e}")
    
    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get test results by test ID"""
        try:
            doc = self.test_results_collection.find_one(
                {"test_id": test_id}, 
                {"_id": 0}
            )
            
            if not doc:
                return None
            
            return {
                "test_id": test_id,
                "score": doc.get("score", 0),
                "total_questions": doc.get("total_questions", 0),
                "score_percentage": doc.get("score_percentage", 0),
                "analytics": doc.get("evaluation_report", "Report not available"),
                "timestamp": doc.get("timestamp", 0),
                "pdf_available": True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get test results: {e}")
            raise Exception(f"Test results retrieval failed: {e}")
    
    def get_all_test_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all test results with pagination"""
        try:
            results = list(self.test_results_collection.find(
                {},
                {"_id": 0, "test_id": 1, "name": 1, "score": 1, "total_questions": 1, 
                 "score_percentage": 1, "timestamp": 1, "user_type": 1}
            ).sort("timestamp", -1).limit(limit))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get all test results: {e}")
            raise Exception(f"Test results retrieval failed: {e}")
    
    def get_student_list(self) -> List[Dict[str, Any]]:
        """Get unique students from test results"""
        try:
            pipeline = [
                {"$group": {"_id": "$Student_ID", "name": {"$first": "$name"}}},
                {"$project": {"_id": 0, "Student_ID": "$_id", "name": 1}}
            ]
            students = list(self.test_results_collection.aggregate(pipeline))
            return students
            
        except Exception as e:
            logger.error(f"❌ Failed to get student list: {e}")
            raise Exception(f"Student list retrieval failed: {e}")
    
    def get_student_tests(self, student_id: str) -> List[Dict[str, Any]]:
        """Get tests for specific student"""
        try:
            results = list(self.test_results_collection.find(
                {"Student_ID": int(student_id)},
                {"_id": 0, "qa_details": 0, "question_types": 0}
            ))
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get student tests: {e}")
            raise Exception(f"Student tests retrieval failed: {e}")
    
    def fetch_student_info(self):
        """Fetch student info from SQL Server"""
        try:
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
                raise Exception("No valid student data found in database")
            
            student_id, first_name, last_name, session_id = result
            
            # Validate data quality
            if not all([student_id, first_name, last_name, session_id]):
                raise Exception("Incomplete student data retrieved")
            
            logger.info(f"✅ Student info retrieved: {student_id}")
            return student_id, first_name, last_name, session_id
            
        except Exception as e:
            logger.error(f"❌ Student info fetch failed: {e}")
            raise Exception(f"Student data unavailable: {e}")
    
    def validate_connection(self) -> Dict[str, Any]:
        """Validate database connections"""
        status = {
            "mongodb": False,
            "sql_server": False,
            "collections_accessible": False,
            "overall": False
        }
        
        try:
            # Test MongoDB
            self.mongo_client.server_info()
            status["mongodb"] = True
            
            # Test collections
            self.summaries_collection.count_documents({}, limit=1)
            status["collections_accessible"] = True
            
            # Test SQL Server
            conn = pyodbc.connect(self.sql_connection_string, timeout=5)
            conn.close()
            status["sql_server"] = True
            
            status["overall"] = all([
                status["mongodb"],
                status["sql_server"],
                status["collections_accessible"]
            ])
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
        
        return status
    
    def close(self):
        """Close database connections"""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("✅ Database connections closed")

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