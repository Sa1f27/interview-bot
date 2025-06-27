# weekend_mocktest/main.py - Pure RAG-Based System with Strict Error Handling
import logging
import os
import time
import uuid
import random
import re
import pymongo
import pyodbc
import markdown
import io
import gc
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from groq import Groq
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER

# RAG imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from urllib.parse import quote_plus

# ——— Configuration ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent

# ——— Database Configuration ———
DB_CONFIG = {
    "DRIVER": "ODBC Driver 17 for SQL Server",
    "SERVER": "192.168.48.200",
    "DATABASE": "SuperDB",
    "UID": "sa",
    "PWD": "Welcome@123",
}
CONNECTION_STRING = (
    f"DRIVER={{{DB_CONFIG['DRIVER']}}};"
    f"SERVER={DB_CONFIG['SERVER']};"
    f"DATABASE={DB_CONFIG['DATABASE']};"
    f"UID={DB_CONFIG['UID']};"
    f"PWD={DB_CONFIG['PWD']}"
)

# MongoDB credentials
MONGO_USER = "LanTech"
MONGO_PASS = "L@nc^ere@0012"
MONGO_HOST = "192.168.48.201:27017"
MONGO_DB_NAME = "Api-1"
MONGO_AUTH_SOURCE = "admin"

# ——— Pure RAG Database Manager (No Fallbacks) ———
class PureRAGDatabaseManager:
    def __init__(self, connection_string, db_name):
        logger.info("🚀 Initializing Pure RAG Database Manager")
        
        # MongoDB connection - fail fast if unavailable
        try:
            self.client = pymongo.MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=10,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000
            )
            
            # Test connection immediately
            self.client.server_info()
            
            self.db = self.client[db_name]
            self.transcripts_collection = self.db["original-1"]
            self.test_results_collection = self.db["mock_test_results"]
            
            # Create indexes for performance
            try:
                self.test_results_collection.create_index("test_id")
                self.test_results_collection.create_index("timestamp")
                logger.info("✅ Database indexes created")
            except Exception as idx_error:
                logger.warning(f"⚠️ Index creation failed: {idx_error}")
            
            logger.info("✅ MongoDB connection established")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise Exception(f"MongoDB connection failure: {e}")
        
        # RAG initialization - mandatory for operation
        self.vector_store = None
        self.embeddings = None
        self.rag_enabled = False
        
        try:
            self._init_rag_strict()
            if not self.rag_enabled:
                raise Exception("RAG system failed to initialize")
            logger.info("✅ Pure RAG system initialized successfully")
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            raise Exception(f"RAG system initialization failure: {e}")
    
    def _init_rag_strict(self):
        """Strict RAG initialization - fail if unable to build proper system"""
        try:
            vector_path = BASE_DIR / "vector_store"
            cache_file = vector_path / "cache.txt"
            
            # Load from cache if recent and valid
            if (cache_file.exists() and 
                (time.time() - cache_file.stat().st_mtime) < 43200):
                
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'batch_size': 16}
                    )
                    
                    self.vector_store = FAISS.load_local(
                        str(vector_path), 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    # Validate vector store has content
                    if self.vector_store.index.ntotal == 0:
                        raise Exception("Vector store is empty")
                    
                    self.rag_enabled = True
                    logger.info(f"✅ RAG loaded from cache with {self.vector_store.index.ntotal} vectors")
                    return
                    
                except Exception as cache_error:
                    logger.warning(f"⚠️ Cache load failed: {cache_error}, rebuilding...")
            
            # Build fresh RAG system
            self._build_rag_strict()
            
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            self.rag_enabled = False
            raise Exception(f"RAG system cannot be initialized: {e}")
    
    def _build_rag_strict(self):
        """Build RAG system with strict validation - fail if insufficient data"""
        try:
            logger.info("🔨 Building RAG system from scratch")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'batch_size': 32, 'normalize_embeddings': True}
            )
            
            # Get summaries with strict validation
            cursor = self.transcripts_collection.find(
                {"summary": {"$exists": True, "$ne": ""}},
                {"summary": 1, "timestamp": 1, "date": 1, "session_id": 1}
            ).limit(100)
            
            summaries = list(cursor)
            
            if not summaries:
                raise Exception("No summaries found in database for RAG construction")
            
            if len(summaries) < 5:
                raise Exception(f"Insufficient summaries for quality RAG: found {len(summaries)}, need at least 5")
            
            # Create documents with quality validation
            documents = []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=384,
                chunk_overlap=32,
                separators=["\n\n", "\n", ". "]
            )
            
            total_content_length = 0
            
            for doc in summaries:
                summary_text = doc.get("summary", "").strip()
                
                if len(summary_text) < 100:  # Skip very short summaries
                    continue
                
                chunks = text_splitter.split_text(summary_text)
                
                for chunk in chunks[:5]:  # Limit chunks per document
                    if len(chunk.strip()) > 50:  # Quality threshold
                        documents.append(Document(
                            page_content=chunk.strip(),
                            metadata={
                                "source_id": str(doc["_id"]),
                                "date": str(doc.get("date", "unknown")),
                                "session_id": str(doc.get("session_id", "unknown"))
                            }
                        ))
                        total_content_length += len(chunk)
            
            if not documents:
                raise Exception("No valid documents created from summaries")
            
            if len(documents) < 10:
                raise Exception(f"Insufficient quality chunks: created {len(documents)}, need at least 10")
            
            if total_content_length < 5000:
                raise Exception(f"Insufficient content for quality RAG: {total_content_length} chars, need at least 5000")
            
            # Build vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Validate vector store
            if self.vector_store.index.ntotal == 0:
                raise Exception("Vector store creation failed - no vectors generated")
            
            # Save to cache
            vector_path = BASE_DIR / "vector_store"
            vector_path.mkdir(exist_ok=True)
            self.vector_store.save_local(str(vector_path))
            
            with open(vector_path / "cache.txt", "w") as f:
                f.write(f"Quality build: {time.time()}, vectors: {self.vector_store.index.ntotal}")
            
            self.rag_enabled = True
            logger.info(f"✅ RAG built successfully: {len(documents)} chunks, {self.vector_store.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"❌ RAG build failed: {e}")
            self.rag_enabled = False
            raise Exception(f"RAG construction failed: {e}")
    
    def load_transcript_strict(self, query_context: str = "programming development concepts algorithms"):
        """Pure RAG-based transcript loading - no fallbacks"""
        if not self.rag_enabled or not self.vector_store:
            raise Exception("RAG system not available - cannot generate authentic content")
        
        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(query_context, k=5)
            
            if not docs:
                raise Exception("No relevant content found in RAG vector store")
            
            # Combine and validate context
            context_parts = []
            for doc in docs:
                content = doc.page_content.strip()
                if len(content) > 30:  # Quality threshold
                    context_parts.append(content)
            
            if not context_parts:
                raise Exception("No quality content retrieved from RAG")
            
            combined_context = "\n\n".join(context_parts)
            
            if len(combined_context) < 200:
                raise Exception(f"Insufficient context retrieved: {len(combined_context)} chars, need at least 200")
            
            logger.info(f"✅ RAG context retrieved: {len(combined_context)} chars from {len(context_parts)} chunks")
            return combined_context
            
        except Exception as e:
            logger.error(f"❌ RAG retrieval failed: {e}")
            raise Exception(f"RAG content retrieval failed: {e}")
    
    def get_rag_stats(self):
        """Get comprehensive RAG statistics"""
        if self.rag_enabled and self.vector_store:
            try:
                total_vectors = self.vector_store.index.ntotal
                return {
                    "status": "active",
                    "total_vectors": total_vectors,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "quality_threshold": "strict",
                    "min_content_length": 200,
                    "vector_store_path": str(BASE_DIR / "vector_store")
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}
        else:
            return {"status": "not_initialized"}
    
    def validate_system_health(self):
        """Comprehensive system health validation"""
        health_status = {
            "mongodb": False,
            "rag_system": False,
            "vector_store": False,
            "content_quality": False,
            "overall": False
        }
        
        try:
            # Test MongoDB
            self.client.server_info()
            health_status["mongodb"] = True
            
            # Test RAG system
            if self.rag_enabled and self.vector_store:
                health_status["rag_system"] = True
                
                # Test vector store
                if self.vector_store.index.ntotal > 0:
                    health_status["vector_store"] = True
                    
                    # Test content quality
                    test_docs = self.vector_store.similarity_search("test", k=1)
                    if test_docs and len(test_docs[0].page_content) > 50:
                        health_status["content_quality"] = True
            
            health_status["overall"] = all([
                health_status["mongodb"],
                health_status["rag_system"], 
                health_status["vector_store"],
                health_status["content_quality"]
            ])
            
        except Exception as e:
            logger.error(f"Health validation failed: {e}")
        
        return health_status
    
    def close(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()

# Initialize database manager with strict validation
try:
    db_manager = PureRAGDatabaseManager(
        f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASS)}@{MONGO_HOST}/{MONGO_DB_NAME}?authSource={MONGO_AUTH_SOURCE}",
        MONGO_DB_NAME
    )
    logger.info("✅ Pure RAG Database Manager initialized successfully")
except Exception as e:
    logger.error(f"❌ Database Manager initialization failed: {e}")
    db_manager = None

# Initialize Groq client with strict validation
try:
    groq_client = Groq(timeout=30)
    
    # Test Groq connection
    test_completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "test"}],
        max_completion_tokens=10
    )
    
    if not test_completion.choices:
        raise Exception("Groq test call failed")
    
    logger.info("✅ Groq client initialized and tested successfully")
except Exception as e:
    logger.error(f"❌ Groq client initialization failed: {e}")
    groq_client = None

# In-memory storage
TESTS = {}
ANSWERS = {}

# ——— Strict SQL Server Functions ———
def fetch_student_info_strict():
    """Strict student info fetch - fail if unavailable"""
    try:
        conn = pyodbc.connect(CONNECTION_STRING, timeout=10)
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

# ——— Strict Question Generation ———
def generate_question_strict(user_type: str, question_type: str, context: str, difficulty: int = 1):
    """Strict question generation - no fallbacks"""
    if not groq_client:
        raise Exception("LLM service unavailable - cannot generate questions")
    
    if not context or len(context.strip()) < 100:
        raise Exception("Insufficient context for question generation")
    
    # Validate context quality
    context_preview = context[:1500]
    
    if user_type == "dev":
        prompt = f"""Generate a {question_type} programming question (difficulty {difficulty}/3) using this context:

CONTEXT: {context_preview}

Requirements:
- Create a practical coding challenge based on the context
- Include clear problem statement with specific requirements
- Specify expected solution approach and constraints
- Start with "## Question"
- Make it challenging but solvable
- Use real-world scenarios from the context"""
    else:
        prompt = f"""Generate a {question_type} multiple-choice question (difficulty {difficulty}/3) using this context:

CONTEXT: {context_preview}

Requirements:
- Test deep conceptual understanding from the context
- 4 options with exactly 1 correct answer
- Include nuanced distractors based on common misconceptions
- Format strictly as:
## Question
[Your question based on the context]

## Options
A) [Option A]
B) [Option B] 
C) [Option C]
D) [Option D]"""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=1000,
            top_p=0.9
        )
        
        if not completion.choices:
            raise Exception("LLM returned no response")
        
        response = completion.choices[0].message.content.strip()
        
        if not response or len(response) < 50:
            raise Exception("LLM returned insufficient content")
        
        if user_type == "non_dev":
            question, options = parse_mcq_strict(response)
            if not options or len(options) != 4:
                raise Exception("Failed to parse valid MCQ with 4 options")
            return question, options
        else:
            if "## Question" not in response:
                raise Exception("Invalid question format returned")
            return response, None
            
    except Exception as e:
        logger.error(f"❌ Question generation failed: {e}")
        raise Exception(f"Question generation failed: {e}")

def parse_mcq_strict(response: str):
    """Strict MCQ parsing with validation"""
    try:
        if "## Options" not in response:
            raise Exception("No options section found in response")
        
        q_part, opt_part = response.split("## Options", 1)
        question = q_part.replace("## Question", "").strip()
        
        if not question or len(question) < 20:
            raise Exception("Question too short or invalid")
        
        lines = [line.strip() for line in opt_part.split('\n') if line.strip()]
        options = []
        
        for line in lines:
            if re.match(r'^[A-D]\)', line):
                option_text = line[3:].strip()
                if len(option_text) > 5:  # Quality threshold
                    options.append(option_text)
        
        if len(options) != 4:
            raise Exception(f"Invalid number of options: found {len(options)}, need exactly 4")
        
        return question, options
        
    except Exception as e:
        raise Exception(f"MCQ parsing failed: {e}")

# ——— Strict Batch Evaluation ———
def batch_evaluate_test_strict(test_id: str, test_data: dict, answers_data: list) -> dict:
    """Strict batch evaluation - no fallbacks"""
    logger.info(f"🧠 Starting strict evaluation: {test_id}")
    
    if not groq_client:
        raise Exception("LLM service unavailable for evaluation")
    
    if not answers_data:
        raise Exception("No answers provided for evaluation")
    
    try:
        # Prepare Q&A pairs with validation
        qa_pairs = []
        for i, answer in enumerate(answers_data, 1):
            if not answer.get('question') or not answer.get('answer'):
                raise Exception(f"Invalid answer data for question {i}")
            
            question_preview = answer['question'][:200] + "..." if len(answer['question']) > 200 else answer['question']
            answer_preview = answer['answer'][:150] + "..." if len(answer['answer']) > 150 else answer['answer']
            
            qa_pairs.append(f"Q{i}: {question_preview}\nAnswer: {answer_preview}")
        
        # Comprehensive evaluation prompt
        batch_prompt = f"""Evaluate this {test_data['user_type']} test comprehensively and provide detailed analysis:

{chr(10).join(qa_pairs)}

Provide evaluation in this EXACT format:
SCORES: [comma-separated 1s and 0s for each question]
FEEDBACK: [detailed feedback for each question separated by |]
OVERALL_SCORE: [score out of 10]
PERFORMANCE_LEVEL: [Excellent/Good/Average/Needs Improvement]
STRENGTHS: [specific strengths observed]
IMPROVEMENTS: [specific areas needing improvement]
RECOMMENDATIONS: [actionable recommendations for learning]

Be thorough, specific, and constructive in your evaluation."""

        logger.info(f"📤 Sending evaluation request: {test_id}")
        
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": batch_prompt}],
            temperature=0.3,
            max_completion_tokens=1500,
            top_p=0.9
        )
        
        if not completion.choices:
            raise Exception("LLM evaluation returned no response")
        
        result = completion.choices[0].message.content.strip()
        
        if not result or len(result) < 100:
            raise Exception("LLM returned insufficient evaluation content")
        
        logger.info(f"📥 Evaluation response received: {test_id}")
        
        return parse_evaluation_result_strict(result, answers_data)
        
    except Exception as e:
        logger.error(f"❌ Strict evaluation failed: {e}")
        raise Exception(f"Evaluation failed: {e}")

def parse_evaluation_result_strict(result: str, answers_data: list) -> dict:
    """Simplified evaluation parsing - store detailed report as-is"""
    logger.info("🔍 Parsing evaluation result")
    
    try:
        parsed = {}
        lines = [line.strip() for line in result.split('\n') if line.strip()]
        
        # Extract only scores and feedbacks for conversation pairs
        scores_found = False
        feedback_found = False
        
        for line in lines:
            if line.startswith('SCORES:'):
                scores_str = line.replace('SCORES:', '').strip()
                scores = [int(x.strip()) for x in scores_str.split(',') if x.strip().isdigit()]
                
                if len(scores) != len(answers_data):
                    raise Exception(f"Score count mismatch: got {len(scores)}, expected {len(answers_data)}")
                
                parsed['scores'] = scores
                parsed['total_correct'] = sum(scores)
                scores_found = True
                
            elif line.startswith('FEEDBACK:'):
                feedback_str = line.replace('FEEDBACK:', '').strip()
                feedbacks = [f.strip() for f in feedback_str.split('|')]
                
                if len(feedbacks) != len(answers_data):
                    raise Exception(f"Feedback count mismatch: got {len(feedbacks)}, expected {len(answers_data)}")
                
                parsed['feedbacks'] = feedbacks
                feedback_found = True
        
        # Validate required components
        if not scores_found:
            raise Exception("No scores found in evaluation")
        
        if not feedback_found:
            raise Exception("No feedback found in evaluation")
        
        # Update answers with evaluation for conversation pairs
        for i, answer in enumerate(answers_data):
            answer['correct'] = bool(parsed['scores'][i])
            answer['feedback'] = parsed['feedbacks'][i]
        
        # Store the complete LLM response as evaluation report
        parsed['evaluation_report'] = result
        
        logger.info(f"✅ Evaluation parsed: {parsed['total_correct']}/{len(answers_data)}")
        return parsed
        
    except Exception as e:
        logger.error(f"❌ Evaluation parsing failed: {e}")
        raise Exception(f"Evaluation parsing failed: {e}")

def save_test_results_strict(test_id: str, test_data: dict, evaluation_result: dict) -> bool:
    """Simplified save operation with clean document structure"""
    logger.info(f"💾 Saving test results: {test_id}")
    
    if not db_manager or not db_manager.client:
        raise Exception("Database not available for saving results")
    
    try:
        # Get authentic student info
        student_id, first_name, last_name, session_id = fetch_student_info_strict()
        name = f"{first_name} {last_name}"
        
        # Create conversation pairs from answers
        conversation_pairs = []
        for i, answer_data in enumerate(ANSWERS.get(test_id, []), 1):
            conversation_pairs.append({
                "question_number": i,
                "question": answer_data.get("question", ""),
                "answer": answer_data.get("answer", ""),
                "correct": answer_data.get("correct", False),
                "feedback": answer_data.get("feedback", "")
            })
        
        # Calculate score percentage
        score_percentage = round((evaluation_result["total_correct"] / test_data["total_questions"]) * 100, 1)
        
        # Clean document structure
        document = {
            "test_id": test_id,
            "timestamp": time.time(),
            "student_id": student_id,
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
        result = db_manager.test_results_collection.insert_one(document)
        
        if not result.inserted_id:
            raise Exception("Database save operation failed")
        
        logger.info(f"✅ Test saved successfully: {test_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Save failed: {e}")
        raise Exception(f"Results save failed: {e}")

# ——— Memory Management ———
def cleanup_memory():
    """Periodic memory cleanup"""
    try:
        current_time = time.time()
        expired_tests = [
            test_id for test_id, test_data in TESTS.items() 
            if current_time - test_data.get("created_at", 0) > 1800
        ]
        
        for test_id in expired_tests:
            TESTS.pop(test_id, None)
            ANSWERS.pop(test_id, None)
        
        gc.collect()
        
        if expired_tests:
            logger.info(f"Memory cleanup: removed {len(expired_tests)} expired tests")
        
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")

def periodic_cleanup():
    """Background cleanup thread"""
    while True:
        time.sleep(1800)
        cleanup_memory()

# Start cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

# ——— Pydantic Models ———
class StartTestRequest(BaseModel):
    user_type: str

class SubmitAnswerRequest(BaseModel):
    test_id: str
    question_number: int
    answer: str

class TestResponse(BaseModel):
    test_id: str
    user_type: str
    question_number: int
    total_questions: int
    question_html: str
    options: Optional[List[str]] = None
    time_limit: int

class NextQuestionResponse(BaseModel):
    question_number: int
    total_questions: int
    question_html: str
    options: Optional[List[str]] = None
    time_limit: int

class SubmitAnswerResponse(BaseModel):
    test_completed: bool
    next_question: Optional[NextQuestionResponse] = None
    score: Optional[int] = None
    total_questions: Optional[int] = None
    analytics: Optional[str] = None

# ——— Remove redundant system validation models and endpoints ———

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: float
    active_tests: int

# ——— FastAPI Application ———
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simplified application lifespan management"""
    logger.info("🚀 Mock Test API starting...")
    
    # Basic validation only
    if not db_manager:
        logger.error("❌ Database manager not initialized")
        raise Exception("Database manager initialization failed")
    
    if not groq_client:
        logger.error("❌ Groq client not initialized")
        raise Exception("Groq client initialization failed")
    
    logger.info("✅ Core systems ready")
    
    yield
    
    logger.info("👋 Shutting down...")
    if db_manager:
        db_manager.close()

app = FastAPI(
    title="Clean Mock Test API",
    description="Streamlined RAG-based mock testing system",
    version="4.0.0-clean",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# ——— Remove system validation endpoint completely ———

@app.get("/")
async def home():
    """Home endpoint with basic status"""
    html_path = frontend_dir / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    
    return {
        "service": "Clean Mock Test API",
        "version": "1.2.0",
        "system_ready": bool(db_manager and groq_client),
        "timestamp": time.time()
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Simple health check"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database manager not available")
        
        return HealthResponse(
            status="ok",
            message="System operational",
            timestamp=time.time(),
            active_tests=len(TESTS)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/api/test/start", response_model=TestResponse)
def start_test(request: StartTestRequest):
    """Start test with strict validation"""
    logger.info(f"🚀 Starting test: {request.user_type}")
    
    # Validate request
    if request.user_type not in ["dev", "non_dev"]:
        raise HTTPException(status_code=400, detail="Invalid user type - must be 'dev' or 'non_dev'")
    
    # Validate system readiness
    if not db_manager or not groq_client:
        raise HTTPException(status_code=503, detail="System not ready - missing core components")
    
    if not db_manager.rag_enabled:
        raise HTTPException(status_code=503, detail="RAG system not available - cannot generate authentic content")
    
    try:
        # Generate unique test ID
        test_id = str(uuid.uuid4())
        total_questions = 10
        
        # Load authentic context from RAG
        transcript_content = db_manager.load_transcript_strict(
            f"{request.user_type} programming development concepts algorithms"
        )
        
        question_types = ["practical", "conceptual", "analytical"]
        
        # Initialize test data
        TESTS[test_id] = {
            "user_type": request.user_type, 
            "score": 0, 
            "question_count": 1, 
            "total_questions": total_questions,
            "question_types": question_types,
            "created_at": time.time(),
            "transcript_context": transcript_content
        }
        ANSWERS[test_id] = []

        # Generate first question using RAG context
        question, options = generate_question_strict(
            request.user_type, 
            question_types[0], 
            transcript_content,
            difficulty=1
        )
        
        # Store question and answer structure
        ANSWERS[test_id].append({
            "question": question, 
            "answer": "", 
            "options": options or []
        })

        logger.info(f"✅ Test started successfully: {test_id}")
        
        return TestResponse(
            test_id=test_id,
            user_type=request.user_type,
            question_number=1,
            total_questions=total_questions,
            question_html=markdown.markdown(question),
            options=options,
            time_limit=300 if request.user_type == "dev" else 120
        )
        
    except Exception as e:
        logger.error(f"❌ Test start failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Test initialization failed",
                "component": "RAG/LLM/Database",
                "message": str(e),
                "action": "Verify system health and retry"
            }
        )

@app.post("/api/test/submit", response_model=SubmitAnswerResponse)
def submit_answer(request: SubmitAnswerRequest):
    """Submit answer with strict validation and no fallbacks"""
    logger.info(f"📝 Submit request: {request.test_id}, Q{request.question_number}")
    
    # Validate test exists
    test = TESTS.get(request.test_id)
    if not test:
        logger.error(f"❌ Test not found: {request.test_id}")
        raise HTTPException(status_code=404, detail="Test not found or expired")
    
    # Validate question number
    if request.question_number != test["question_count"]:
        logger.error(f"❌ Invalid question number: {request.question_number} != {test['question_count']}")
        raise HTTPException(status_code=400, detail="Invalid question number")
    
    # Validate answer
    if not request.answer or not request.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty")

    try:
        # Process answer
        ans_data = ANSWERS[request.test_id][-1]
        full_answer = request.answer.strip()
        
        # Convert option index for MCQ
        if test["user_type"] == "non_dev" and request.answer.isdigit():
            try:
                option_index = int(request.answer)
                options = ans_data["options"]
                if 0 <= option_index < len(options):
                    full_answer = options[option_index]
                    logger.info(f"✅ MCQ answer converted: {option_index} -> {full_answer}")
                else:
                    raise HTTPException(status_code=400, detail="Invalid option index")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid option format")

        # Store answer
        ans_data["answer"] = full_answer
        logger.info(f"✅ Answer stored: {request.test_id}, Q{request.question_number}")

        # Check if test completed
        if test["question_count"] >= test["total_questions"]:
            logger.info(f"🏁 Test completed - starting evaluation: {request.test_id}")
            
            # Strict batch evaluation
            evaluation_result = batch_evaluate_test_strict(
                request.test_id, 
                test, 
                ANSWERS[request.test_id]
            )
            logger.info(f"✅ Evaluation completed: {request.test_id}")
            
            # Save to database with strict validation
            save_success = save_test_results_strict(
                test_id=request.test_id,
                test_data=test,
                evaluation_result=evaluation_result
            )
            
            if not save_success:
                logger.warning(f"⚠️ Save may have failed: {request.test_id}")
            
            # Clean up memory
            TESTS.pop(request.test_id, None)
            ANSWERS.pop(request.test_id, None)
            
            logger.info(f"🎉 Test completed successfully: {request.test_id}")
            
            return SubmitAnswerResponse(
                test_completed=True,
                score=evaluation_result["total_correct"],
                total_questions=test["total_questions"],
                analytics=evaluation_result["evaluation_report"]
            )

        # Generate next question
        logger.info(f"➡️ Generating next question: {request.test_id}")
        
        test["question_count"] += 1
        next_q_num = test["question_count"]
        
        next_type_index = (test["question_count"] - 1) % len(test["question_types"])
        next_type = test["question_types"][next_type_index]
        
        next_question, next_options = generate_question_strict(
            test["user_type"], 
            next_type, 
            test["transcript_context"],
            difficulty=next_q_num
        )
        
        ANSWERS[request.test_id].append({
            "question": next_question, 
            "answer": "", 
            "options": next_options or []
        })
        
        logger.info(f"✅ Next question generated: {request.test_id}, Q{next_q_num}")

        return SubmitAnswerResponse(
            test_completed=False,
            next_question=NextQuestionResponse(
                question_number=next_q_num,
                total_questions=test["total_questions"],
                question_html=markdown.markdown(next_question),
                options=next_options,
                time_limit=300 if test["user_type"] == "dev" else 120
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Submit answer failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Answer processing failed",
                "component": "LLM/RAG",
                "message": str(e),
                "test_id": request.test_id,
                "action": "Retry submission or check system health"
            }
        )

@app.get("/api/test/results/{test_id}")
async def get_test_results(test_id: str):
    """Get test results with strict validation"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        doc = db_manager.test_results_collection.find_one({"test_id": test_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Test results not found")
        
        return {
            "test_id": test_id,
            "score": doc.get("score", 0),
            "total_questions": doc.get("total_questions", 0),
            "score_percentage": doc.get("score_percentage", 0),
            "analytics": doc.get("evaluation_report", "Report not available"),
            "timestamp": doc.get("timestamp", 0),
            "pdf_available": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch results: {str(e)}")

@app.get("/api/test/pdf/{test_id}")
async def download_pdf(test_id: str):
    """Generate comprehensive PDF report"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        doc = db_manager.test_results_collection.find_one({"test_id": test_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Test not found")

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=LETTER)
        
        # Enhanced PDF generation
        p.setFont("Helvetica-Bold", 18)
        p.drawString(50, 750, f"Mock Test Results")
        
        p.setFont("Helvetica", 12)
        y = 700
        p.drawString(50, y, f"Test ID: {test_id}")
        p.drawString(50, y-20, f"Student: {doc.get('name', 'N/A')}")
        p.drawString(50, y-40, f"Type: {doc.get('user_type', 'N/A').title()}")
        p.drawString(50, y-60, f"Score: {doc.get('score', 0)}/{doc.get('total_questions', 0)} ({doc.get('score_percentage', 0)}%)")
        
        # Add timestamp
        import datetime
        timestamp = doc.get('timestamp', time.time())
        date_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        p.drawString(50, y-80, f"Completed: {date_str}")
        
        p.save()
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=test_results_{test_id}.pdf"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

# ——— Remove redundant debug and validation endpoints ———

# Keep only essential endpoints - remove system/validate, rag/refresh, and complex debug

@app.get("/api/debug/status")
async def debug_status():
    """Essential debug information only"""
    try:
        return {
            "api_status": "running",
            "timestamp": time.time(),
            "active_tests": len(TESTS),
            "groq_available": groq_client is not None,
            "mongodb_available": db_manager.client is not None if db_manager else False,
            "version": "4.0.0-clean"
        }
    except Exception as e:
        logger.error(f"Debug status failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug status failed: {str(e)}")

@app.get("/api/tests")
async def get_all_tests():
    """Get all tests with proper error handling"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        results = list(db_manager.test_results_collection.find(
            {},
            {"_id": 0, "test_id": 1, "name": 1, "score": 1, "total_questions": 1, 
             "score_percentage": 1, "timestamp": 1, "user_type": 1}
        ).sort("timestamp", -1).limit(50))
        
        return {
            "count": len(results),
            "results": results,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching tests: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch tests: {str(e)}")

@app.delete("/api/cleanup")
async def cleanup_resources():
    """Clean up resources with validation"""
    try:
        current_time = time.time()
        expired_tests = []
        
        for test_id, test_data in list(TESTS.items()):
            if current_time - test_data.get("created_at", 0) > 3600:
                expired_tests.append(test_id)
                TESTS.pop(test_id, None)
                ANSWERS.pop(test_id, None)
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"🧹 Cleanup completed: {len(expired_tests)} tests removed")
        
        return {
            "message": "Cleanup completed successfully",
            "tests_cleaned": len(expired_tests),
            "active_tests": len(TESTS),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint with system validation"""
    try:
        system_ready = bool(db_manager and groq_client and db_manager.rag_enabled)
        
        return {
            "message": "Pure RAG Mock Test API is operational",
            "system_ready": system_ready,
            "timestamp": time.time(),
            "version": "4.0.0-clean"
        }
    except Exception as e:
        logger.error(f"Test endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"System test failed: {str(e)}")

# ——— Startup Message ———
if db_manager and groq_client:
    logger.info("🚀 Clean Mock Test API loaded successfully")
    logger.info(f"✅ Features: RAG-based content, Clean data structure")
    logger.info(f"⚡ Status: Core systems operational")
else:
    logger.error("❌ System initialization incomplete - some components failed")
    

@app.get("/api/students")
async def get_unique_students():
    try:
        pipeline = [
            {"$group": {"_id": "$Student_ID", "name": {"$first": "$name"}}},
            {"$project": {"_id": 0, "Student_ID": "$_id", "name": 1}}
        ]
        students = list(db_manager.test_results_collection.aggregate(pipeline))
        return JSONResponse(content={"count": len(students), "students": students})
    except Exception as e:
        logger.error(f"Error fetching student list: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch student list")


@app.get("/api/students/{student_id}/tests")
async def get_tests_for_student(student_id: str):
    try:
        results = list(db_manager.test_results_collection.find(
            {"Student_ID": int(student_id)},
            {"_id": 0, "qa_details": 0, "question_types": 0}
        ))
        if not results:
            raise HTTPException(status_code=404, detail="No tests found for this student")
        return JSONResponse(content={"count": len(results), "tests": results})
    except Exception as e:
        logger.error(f"Error fetching tests for student ID {student_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch tests for student: {str(e)}")