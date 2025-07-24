"""
Database management module for Daily Standup application
Handles MongoDB and SQL Server connections and operations
"""

import os
import time
import logging
import asyncio
import pyodbc
from typing import Tuple, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from urllib.parse import quote_plus
from .config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Optimized database management with connection pooling"""
    
    def __init__(self, client_manager):
        self.client_manager = client_manager
        self._mongo_client = None
        self._mongo_db = None
        
    @property
    def mongo_config(self) -> Dict[str, Any]:
        """Get MongoDB configuration from environment"""
        return {
            "host": os.getenv("MONGODB_HOST"),
            "port": int(os.getenv("MONGODB_PORT", "27017")),
            "username": os.getenv("MONGODB_USERNAME"),
            "password": os.getenv("MONGODB_PASSWORD"),
            "database": os.getenv("MONGODB_DATABASE")
        }
    
    @property
    def sql_config(self) -> Dict[str, Any]:
        """Get SQL Server configuration from environment"""
        return {
            "DRIVER": os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server"),
            "SERVER": os.getenv("SQL_SERVER"),
            "DATABASE": os.getenv("SQL_DATABASE"),
            "UID": os.getenv("SQL_USERNAME"),
            "PWD": os.getenv("SQL_PASSWORD"),
            "timeout": int(os.getenv("SQL_TIMEOUT", "5"))
        }
    
    async def get_mongo_client(self) -> AsyncIOMotorClient:
        """Get MongoDB client with connection pooling"""
        if self._mongo_client is None:
            mongo_cfg = self.mongo_config
            mongo_url = (
                f"mongodb://{quote_plus(mongo_cfg['username'])}:"
                f"{quote_plus(mongo_cfg['password'])}@"
                f"{mongo_cfg['host']}:{mongo_cfg['port']}/"
                f"{mongo_cfg['database']}?authSource=admin"
            )
            
            self._mongo_client = AsyncIOMotorClient(
                mongo_url, 
                maxPoolSize=config.MONGO_MAX_POOL_SIZE,
                serverSelectionTimeoutMS=config.MONGO_SERVER_SELECTION_TIMEOUT
            )
            
            try:
                await self._mongo_client.admin.command('ping')
                logger.info("✅ MongoDB client initialized")
            except Exception as e:
                logger.error(f"❌ MongoDB connection failed: {e}")
                raise Exception(f"MongoDB connection failed: {e}")
                
        return self._mongo_client
    
    async def get_mongo_db(self):
        """Get MongoDB database instance"""
        if self._mongo_db is None:
            client = await self.get_mongo_client()
            self._mongo_db = client[self.mongo_config["database"]]
        return self._mongo_db
    
    def get_sql_connection(self):
        """Get SQL Server connection with optimized configuration"""
        try:
            sql_cfg = self.sql_config
            conn_str = (
                f"DRIVER={{{sql_cfg['DRIVER']}}};"
                f"SERVER={sql_cfg['SERVER']};"
                f"DATABASE={sql_cfg['DATABASE']};"
                f"UID={sql_cfg['UID']};"
                f"PWD={sql_cfg['PWD']};"
            )
            
            conn = pyodbc.connect(conn_str, timeout=sql_cfg['timeout'])
            return conn
        except Exception as e:
            logger.error(f"❌ SQL connection failed: {e}")
            raise Exception(f"SQL Server connection failed: {e}")
    
    async def get_student_info_fast(self) -> Tuple[int, str, str, str]:
        """Fast student info retrieval with dummy data support"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.client_manager.executor,
            self._sync_get_student_info
        )
    
    def _sync_get_student_info(self) -> Tuple[int, str, str, str]:
        """Synchronous student info retrieval for thread pool"""
        if config.USE_DUMMY_DATA:
            logger.warning("⚠️ Using dummy student info (SQL Server is DOWN)")
            student_id = 99999
            first_name = "Dummy"
            last_name = "User"
            session_key = f"SESSION_{int(time.time())}"
            return (student_id, first_name, last_name, session_key)
        
        try:
            conn = self.get_sql_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT TOP 1 ID, First_Name, Last_Name FROM tbl_Student ORDER BY NEWID()")
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not row:
                raise Exception("No student records found in tbl_Student")
                
            return (row[0], row[1], row[2], f"SESSION_{int(time.time())}")
            
        except Exception as e:
            logger.error(f"❌ Error fetching student info: {e}")
            raise Exception(f"Student info retrieval failed: {e}")
    
    async def get_summary_fast(self) -> str:
        """Fast summary retrieval from MongoDB with dummy data support"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_get_summary
            )
        except Exception as e:
            logger.error(f"❌ Error fetching summary: {e}")
            raise Exception(f"Summary retrieval failed: {e}")
    
    def _sync_get_summary(self) -> str:
        """Synchronous summary retrieval for thread pool"""
        if config.USE_DUMMY_DATA:
            logger.warning("⚠️ Using dummy summary (MongoDB is DOWN)")
            return ("""
   1. MLOps (Machine Learning Operations) is an advanced discipline that combines best practices from Machine Learning, DevOps, and Data Engineering, aiming to automate, streamline, and manage the entire machine learning lifecycle within production environments for increased reliability, scalability, and business impact,

2. covering all key stages such as data acquisition, data validation, exploratory data analysis, feature extraction and engineering, dataset versioning, model design, training, hyperparameter tuning, model evaluation and validation, packaging, deployment to staging and production, as well as ongoing monitoring and retraining of models to maintain performance over time,

3. facilitating effective collaboration and workflow alignment among cross-functional teams including data scientists, machine learning engineers, data engineers, software developers, IT operations, and compliance officers, ensuring clear roles, efficient handoffs, and faster iteration cycles from model research to business deployment,

4. enforcing strict reproducibility and traceability by maintaining version control of not only code but also datasets, features, configurations, environments, and model artifacts, thus enabling any ML experiment, model training, or production deployment to be exactly reproduced and audited when necessary,

5. enabling robust scalability and reliability by leveraging distributed computing frameworks, Kubernetes-based orchestration, containerization, and cloud-native architectures, allowing seamless expansion of workloads, efficient resource utilization, and the ability to serve both batch and real-time inference at massive scale,

6. establishing continuous monitoring, logging, and observability of live models for metrics such as prediction accuracy, latency, resource utilization, data drift, and model drift, with real-time alerts and automated triggers for retraining, rollback, or human-in-the-loop review to minimize risks and ensure sustained model quality in changing environments,

7. automating not only model training and deployment, but also data validation, quality checks, integration tests, model validation gates, and deployment approvals, using robust CI/CD pipelines to reduce manual intervention, accelerate safe releases, and streamline rollback or hotfix procedures,

8. supporting model governance and regulatory compliance requirements by providing comprehensive audit trails of data lineage, experiment runs, model versions, deployment events, and user access logs, along with fine-grained access control and workflow approval mechanisms that are essential for regulated industries such as finance, healthcare, and government,

9. addressing critical operational challenges such as pipeline orchestration, dependency management, security hardening for both data and models, environment reproducibility, scaling of experimentation, resource allocation, cost optimization, and seamless integration with existing IT infrastructure,

10. leveraging a mature ecosystem of open-source and commercial tools including MLflow for experiment tracking, reproducibility, and model registry; Kubeflow for designing, deploying, and managing end-to-end ML pipelines in Kubernetes environments; TFX (TensorFlow Extended) for building robust and scalable ML pipelines using TensorFlow; Apache Airflow for complex workflow orchestration and scheduling; DVC (Data Version Control) for versioning data and model artifacts; Seldon Core and KFServing for flexible model serving; and Prometheus and Grafana for real-time system and model monitoring,

11. ultimately empowering organizations to deliver robust, scalable, transparent, and production-grade machine learning systems, accelerating innovation, reducing technical debt, mitigating operational risks, and maximizing business value through the reliable deployment and continuous improvement of ML-powered solutions.""")
        
        try:
            import asyncio
            db = asyncio.run(self.get_mongo_db())
            collection = db[config.TRANSCRIPTS_COLLECTION]
            
            doc = asyncio.run(collection.find_one(
                {"summary": {"$exists": True, "$ne": None, "$ne": ""}},
                sort=[("timestamp", -1)]
            ))
            
            if not doc or not doc.get("summary"):
                raise Exception("No valid summary found in MongoDB transcripts collection")
                
            summary = doc["summary"].strip()
            if len(summary) < 100:
                raise Exception(f"Summary too short ({len(summary)} chars): {summary}")
                
            return summary
            
        except Exception as e:
            logger.error(f"❌ Sync summary retrieval error: {e}")
            raise Exception(f"MongoDB summary retrieval failed: {e}")
    
    async def save_session_result_fast(self, session_data, evaluation: str, score: float) -> bool:
        """Fast session result saving"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_save_result,
                session_data, evaluation, score
            )
        except Exception as e:
            logger.error(f"❌ Error saving session result: {e}")
            raise Exception(f"Session save failed: {e}")
    
    def _sync_save_result(self, session_data, evaluation: str, score: float) -> bool:
        """Synchronous save for thread pool with enhanced fragment analytics"""
        if config.USE_DUMMY_DATA:
            logger.warning(f"⚠️ DUMMY SAVE: Session {session_data.session_id} result not saved to DB.")
            return True
            
        try:
            import asyncio
            db = asyncio.run(self.get_mongo_db())
            collection = db[config.RESULTS_COLLECTION]
            
            # Enhanced fragment analytics
            fragment_manager = session_data.summary_manager
            progress_info = fragment_manager.get_progress_info() if fragment_manager else {}
            
            document = {
                "test_id": session_data.test_id,
                "session_id": session_data.session_id,
                "student_id": session_data.student_id,
                "student_name": session_data.student_name,
                "session_key": session_data.session_key,
                "timestamp": time.time(),
                "created_at": session_data.created_at,
                "conversation_log": [
                    {
                        "timestamp": exchange.timestamp,
                        "stage": exchange.stage.value,
                        "ai_message": exchange.ai_message,
                        "user_response": exchange.user_response,
                        "transcript_quality": exchange.transcript_quality,
                        "concept": exchange.concept,
                        "is_followup": exchange.is_followup
                    }
                    for exchange in session_data.exchanges
                ],
                "evaluation": evaluation,
                "score": score,
                "total_exchanges": len(session_data.exchanges),
                "greeting_exchanges": session_data.greeting_count,
                
                # Enhanced fragment analytics
                "fragment_analytics": {
                    "total_concepts": len(session_data.fragment_keys),
                    "concepts_covered": list(session_data.concept_question_counts.keys()),
                    "questions_per_concept": dict(session_data.concept_question_counts),
                    "followup_questions": session_data.followup_questions,
                    "main_questions": session_data.question_index,
                    "target_questions_per_concept": session_data.questions_per_concept,
                    "coverage_percentage": round(
                        (len([c for c, count in session_data.concept_question_counts.items() if count > 0]) 
                         / len(session_data.fragment_keys) * 100) 
                        if session_data.fragment_keys else 0, 1
                    )
                },
                
                "duration": time.time() - session_data.created_at
            }
            
            result = asyncio.run(collection.insert_one(document))
            logger.info(f"Session {session_data.session_id} saved with fragment analytics")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sync save error: {e}")
            raise Exception(f"MongoDB save failed: {e}")
    
    async def get_session_result_fast(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Fast session result retrieval"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_get_session_result,
                session_id
            )
        except Exception as e:
            logger.error(f"❌ Error fetching session result: {e}")
            raise Exception(f"Session result retrieval failed: {e}")
    
    def _sync_get_session_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous session result retrieval"""
        if config.USE_DUMMY_DATA:
            logger.warning(f"⚠️ DUMMY GET: Not fetching session {session_id} result from DB.")
            return None
            
        try:
            import asyncio
            db = asyncio.run(self.get_mongo_db())
            collection = db[config.RESULTS_COLLECTION]
            result = asyncio.run(collection.find_one({"session_id": session_id}))
            
            if result:
                result['_id'] = str(result['_id'])
                return result
            return None
            
        except Exception as e:
            logger.error(f"❌ Sync session result error: {e}")
            raise Exception(f"Session result retrieval failed: {e}")
    
    async def close_connections(self):
        """Cleanup method for graceful shutdown"""
        if self._mongo_client:
            self._mongo_client.close()
        logger.info("🔌 Database connections closed")