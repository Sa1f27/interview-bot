# weekly_interview/core/database.py
"""
Database management module for Enhanced Mock Interview System
Handles MongoDB and MySQL connections with connection pooling and optimization
"""

import logging
import asyncio
import time
import mysql.connector
from mysql.connector import pooling, errorcode
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime, timedelta
from .config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Enhanced database management with connection pooling and optimization"""
    
    def __init__(self, client_manager):
        self.client_manager = client_manager
        self._mongo_client = None
        self._mongo_db = None
        self._mysql_pool = None
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            await self._init_mongodb()
            self._init_mysql_pool()
            logger.info("✅ Database connections initialized successfully")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise Exception(f"Database initialization failed: {e}")
    
    async def close_connections(self):
        """Cleanup method for graceful shutdown"""
        try:
            if self._mongo_client:
                self._mongo_client.close()
            
            if self._mysql_pool:
                # MySQL pool doesn't have explicit close, connections auto-close
                self._mysql_pool = None
            
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.warning(f"⚠️ Connection cleanup warning: {e}")

# Singleton pattern for database manager
_db_manager = None

def get_db_manager(client_manager=None):
    """Get database manager singleton"""
    global _db_manager
    if _db_manager is None and client_manager is not None:
        _db_manager = DatabaseManager(client_manager)
    return _db_manager
    
    async def _init_mongodb(self):
        """Initialize MongoDB connection with connection pooling"""
        try:
            self._mongo_client = AsyncIOMotorClient(
                config.mongodb_connection_string,
                maxPoolSize=config.MONGODB_POOL_SIZE,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test connection
            await self._mongo_client.admin.command('ping')
            
            # Initialize database and collections
            self._mongo_db = self._mongo_client[config.MONGODB_DATABASE]
            self.summaries_collection = self._mongo_db[config.SUMMARIES_COLLECTION]
            self.interview_results_collection = self._mongo_db[config.INTERVIEW_RESULTS_COLLECTION]
            
            # Create indexes for performance
            await self._create_mongodb_indexes()
            
            logger.info("✅ MongoDB connection established")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise Exception(f"MongoDB connection failed: {e}")
    
    def _init_mysql_pool(self):
        """Initialize MySQL connection pool"""
        try:
            mysql_config = config.mysql_connection_config
            mysql_config.update({
                'pool_name': 'interview_pool',
                'pool_size': config.MYSQL_POOL_SIZE,
                'pool_reset_session': True,
                'autocommit': True
            })
            
            self._mysql_pool = pooling.MySQLConnectionPool(**mysql_config)
            
            # Test connection
            conn = self._mysql_pool.get_connection()
            conn.close()
            
            logger.info("✅ MySQL connection pool established")
            
        except mysql.connector.Error as e:
            logger.error(f"❌ MySQL connection failed: {e}")
            raise Exception(f"MySQL connection failed: {e}")
    
    async def _create_mongodb_indexes(self):
        """Create MongoDB indexes for performance"""
        try:
            # Summaries collection indexes
            await self.summaries_collection.create_index([("timestamp", -1)])
            await self.summaries_collection.create_index([("date", -1)])
            await self.summaries_collection.create_index([
                ("summary", "text"),
                ("content", "text")
            ])
            
            # Interview results indexes
            await self.interview_results_collection.create_index([("test_id", 1)], unique=True)
            await self.interview_results_collection.create_index([("timestamp", -1)])
            await self.interview_results_collection.create_index([("Student_ID", 1)])
            await self.interview_results_collection.create_index([("session_id", 1)])
            
            logger.info("📊 MongoDB indexes created")
            
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning: {e}")
    
    def get_mysql_connection(self):
        """Get MySQL connection from pool"""
        try:
            return self._mysql_pool.get_connection()
        except mysql.connector.Error as e:
            logger.error(f"❌ MySQL pool connection failed: {e}")
            raise Exception(f"MySQL connection failed: {e}")
    
    async def get_student_info_fast(self) -> Tuple[int, str, str, str]:
        """Fast student info retrieval from MySQL with connection pooling"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.client_manager.executor,
            self._sync_get_student_info
        )
    
    def _sync_get_student_info(self) -> Tuple[int, str, str, str]:
        """Synchronous student info retrieval"""
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get random student with proper data validation
            cursor.execute("""
                SELECT ID, First_Name, Last_Name 
                FROM tbl_Student 
                WHERE ID IS NOT NULL 
                  AND First_Name IS NOT NULL 
                  AND First_Name != ''
                  AND Last_Name IS NOT NULL 
                  AND Last_Name != ''
                ORDER BY RAND()
                LIMIT 1
            """)
            
            student = cursor.fetchone()
            
            if not student:
                raise Exception("No valid student records found")
            
            # Get random session ID
            cursor.execute("""
                SELECT Session_ID 
                FROM tbl_Session 
                WHERE Session_ID IS NOT NULL 
                ORDER BY RAND() 
                LIMIT 1
            """)
            
            session_row = cursor.fetchone()
            session_id = session_row['Session_ID'] if session_row else f"SESS_{int(time.time())}"
            
            cursor.close()
            conn.close()
            
            student_id = student['ID']
            first_name = student['First_Name']
            last_name = student['Last_Name']
            
            logger.info(f"✅ Retrieved student: {first_name} {last_name} (ID: {student_id})")
            return (student_id, first_name, last_name, session_id)
            
        except Exception as e:
            logger.error(f"❌ Error fetching student info: {e}")
            raise Exception(f"Student info retrieval failed: {e}")
    
    async def get_recent_summaries_fast(self, days: int = None, limit: int = None) -> List[Dict[str, Any]]:
        """Enhanced summary retrieval with 7-day window and intelligent processing"""
        if days is None:
            days = config.RECENT_SUMMARIES_DAYS
        if limit is None:
            limit = config.SUMMARIES_LIMIT
            
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_get_recent_summaries,
                days, limit
            )
        except Exception as e:
            logger.error(f"❌ Error fetching summaries: {e}")
            raise Exception(f"Summary retrieval failed: {e}")
    
    def _sync_get_recent_summaries(self, days: int, limit: int) -> List[Dict[str, Any]]:
        """Synchronous recent summaries retrieval with smart filtering"""
        try:
            from pymongo import MongoClient
            
            # Use sync MongoDB client for thread execution
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db[config.SUMMARIES_COLLECTION]
            
            # Calculate date range for last N days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            start_timestamp = start_date.timestamp()
            
            # Enhanced query with multiple fallback strategies
            query_strategies = [
                # Strategy 1: Recent summaries with timestamp
                {
                    "filter": {
                        "summary": {"$exists": True, "$ne": "", "$type": "string"},
                        "timestamp": {"$gte": start_timestamp},
                        "$expr": {"$gt": [{"$strLenCP": "$summary"}, config.MIN_CONTENT_LENGTH]}
                    },
                    "sort": [("timestamp", -1)]
                },
                # Strategy 2: Recent summaries without timestamp filter (fallback)
                {
                    "filter": {
                        "summary": {"$exists": True, "$ne": "", "$type": "string"},
                        "$expr": {"$gt": [{"$strLenCP": "$summary"}, config.MIN_CONTENT_LENGTH]}
                    },
                    "sort": [("_id", -1)]
                },
                # Strategy 3: Any valid summaries (last resort)
                {
                    "filter": {
                        "summary": {"$exists": True, "$ne": "", "$type": "string"}
                    },
                    "sort": [("_id", -1)]
                }
            ]
            
            summaries = []
            
            for strategy in query_strategies:
                try:
                    cursor = collection.find(
                        strategy["filter"],
                        {
                            "summary": 1,
                            "timestamp": 1,
                            "date": 1,
                            "session_id": 1,
                            "_id": 1
                        }
                    ).sort(strategy["sort"]).limit(limit)
                    
                    summaries = list(cursor)
                    
                    if summaries:
                        logger.info(f"✅ Retrieved {len(summaries)} summaries using strategy")
                        break
                        
                except Exception as e:
                    logger.warning(f"⚠️ Strategy failed: {e}")
                    continue
            
            client.close()
            
            if not summaries:
                raise Exception("No valid summaries found in database")
            
            # Log sample for verification
            if summaries:
                first_summary = summaries[0]["summary"]
                sample_length = min(len(first_summary), 150)
                logger.info(f"📄 Sample summary ({sample_length} chars): {first_summary[:sample_length]}...")
            
            return summaries
            
        except Exception as e:
            logger.error(f"❌ Sync summary retrieval error: {e}")
            raise Exception(f"MongoDB summary retrieval failed: {e}")
    
    async def save_interview_result_fast(self, interview_data: Dict[str, Any]) -> bool:
        """Enhanced interview result saving with comprehensive analytics"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_save_interview_result,
                interview_data
            )
        except Exception as e:
            logger.error(f"❌ Error saving interview result: {e}")
            raise Exception(f"Interview save failed: {e}")
    
    def _sync_save_interview_result(self, interview_data: Dict[str, Any]) -> bool:
        """Synchronous interview result saving"""
        try:
            from pymongo import MongoClient
            
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db[config.INTERVIEW_RESULTS_COLLECTION]
            
            # Enhanced document structure
            document = {
                "test_id": interview_data["test_id"],
                "session_id": interview_data["session_id"],
                "student_id": interview_data["student_id"],
                "student_name": interview_data["student_name"],
                "timestamp": time.time(),
                "created_at": interview_data.get("created_at", time.time()),
                
                # Interview content
                "conversation_log": interview_data.get("conversation_log", []),
                "evaluation": interview_data.get("evaluation", ""),
                "scores": interview_data.get("scores", {}),
                
                # Enhanced analytics
                "interview_analytics": {
                    "total_duration_minutes": interview_data.get("duration_minutes", 0),
                    "total_questions": len(interview_data.get("conversation_log", [])),
                    "questions_per_round": interview_data.get("questions_per_round", {}),
                    "round_completion_times": interview_data.get("round_times", {}),
                    "technical_concepts_covered": interview_data.get("concepts_covered", 0),
                    "followup_questions": interview_data.get("followup_questions", 0),
                    "average_response_time": interview_data.get("avg_response_time", 0),
                    "audio_quality_metrics": interview_data.get("audio_metrics", {}),
                    "interview_flow_analytics": interview_data.get("flow_analytics", {})
                },
                
                # System metadata
                "system_info": {
                    "version": config.APP_VERSION,
                    "processing_time": interview_data.get("processing_time", 0),
                    "websocket_used": interview_data.get("websocket_used", True),
                    "audio_format": interview_data.get("audio_format", "webm"),
                    "tts_voice": interview_data.get("tts_voice", config.TTS_VOICE)
                }
            }
            
            result = collection.insert_one(document)
            client.close()
            
            if result.inserted_id:
                logger.info(f"✅ Interview saved: {interview_data['test_id']}")
                return True
            else:
                raise Exception("Database insert failed")
                
        except Exception as e:
            logger.error(f"❌ Sync save error: {e}")
            raise Exception(f"MongoDB save failed: {e}")
    
    async def get_interview_result_fast(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Fast interview result retrieval"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_get_interview_result,
                test_id
            )
        except Exception as e:
            logger.error(f"❌ Error fetching interview result: {e}")
            raise Exception(f"Interview result retrieval failed: {e}")
    
    def _sync_get_interview_result(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous interview result retrieval"""
        try:
            from pymongo import MongoClient
            
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db[config.INTERVIEW_RESULTS_COLLECTION]
            
            result = collection.find_one({"test_id": test_id})
            client.close()
            
            if result:
                result['_id'] = str(result['_id'])
                logger.info(f"✅ Retrieved interview result for {test_id}")
                return result
            
            logger.warning(f"⚠️ No interview result found for {test_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Sync interview result error: {e}")
            raise Exception(f"Interview result retrieval failed: {e}")
    
    async def get_all_interview_results_fast(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all interview results with pagination and analytics"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_get_all_interview_results,
                limit
            )
        except Exception as e:
            logger.error(f"❌ Error fetching all interview results: {e}")
            raise Exception(f"All interview results retrieval failed: {e}")
    
    def _sync_get_all_interview_results(self, limit: int) -> List[Dict[str, Any]]:
        """Synchronous all interview results retrieval"""
        try:
            from pymongo import MongoClient
            
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db[config.INTERVIEW_RESULTS_COLLECTION]
            
            results = list(collection.find(
                {},
                {
                    "_id": 0,
                    "test_id": 1,
                    "student_name": 1,
                    "student_id": 1,
                    "timestamp": 1,
                    "scores": 1,
                    "interview_analytics": 1
                }
            ).sort("timestamp", -1).limit(limit))
            
            client.close()
            
            logger.info(f"✅ Retrieved {len(results)} interview results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Sync all results error: {e}")
            raise Exception(f"All interview results retrieval failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive database health check"""
        health_status = {
            "mongodb": {"status": "unknown", "details": {}},
            "mysql": {"status": "unknown", "details": {}},
            "overall": False
        }
        
        # Test MongoDB
        try:
            await self._mongo_client.admin.command('ping')
            
            # Test collections
            summaries_count = await self.summaries_collection.count_documents({}, limit=1)
            results_count = await self.interview_results_collection.count_documents({}, limit=1)
            
            health_status["mongodb"] = {
                "status": "healthy",
                "details": {
                    "connection": "active",
                    "summaries_available": summaries_count > 0,
                    "results_collection_accessible": True
                }
            }
        except Exception as e:
            health_status["mysql"] = {
                "status": "error", 
                "details": {"error": str(e)}
            }
        
        # Overall status
        health_status["overall"] = (
            health_status["mongodb"]["status"] == "healthy" and
            health_status["mysql"]["status"] == "healthy"
        )
        
        return health_status
            health_status["mongodb"] = {
                "status": "error",
                "details": {"error": str(e)}
            }
        
        # Test MySQL
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            
            health_status["mysql"] = {
                "status": "healthy",
                "details": {
                    "connection": "active",
                    "pool_size": config.MYSQL_POOL_SIZE
                }
            }
        except Exception as e: