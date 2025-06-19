# weekend_mocktest/main.py - Fixed for sub-app mounting
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import random
import re
import pymongo
from groq import Groq
import os
import time
import logging
import pyodbc
from contextlib import asynccontextmanager
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
import markdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# Get the correct base directory
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================
# Pydantic Models (Same as before)
# ========================
class StartTestRequest(BaseModel):
    user_type: str  # "dev" or "non_dev"

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
    pdf_available: Optional[bool] = None

class TestResultsResponse(BaseModel):
    test_id: str
    score: int
    total_questions: int
    analytics: str
    pdf_available: bool

class TestRecord(BaseModel):
    test_id: str
    Student_ID: str
    name: str
    session_id: str
    user_type: str
    final_score: int
    total_questions: int
    score_percentage: float
    test_completed: bool

class AllTestsResponse(BaseModel):
    count: int
    results: List[TestRecord]

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: float
    active_tests: int

# ========================
# Database Configuration (Same as before)
# ========================
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

def get_db_connection():
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        return conn
    except pyodbc.Error as e:
        logger.error(f"Database connection error: {e}")
        return None

def fetch_random_student_info():
    """Fetch a random ID, name from tbl_Student and session_id from session table from SQL Server"""
    try:
        conn = get_db_connection()
        if not conn:
            return None, None, None, None
        
        cursor = conn.cursor()
        cursor.execute("SELECT ID, First_Name, Last_Name FROM tbl_Student WHERE ID IS NOT NULL AND First_Name IS NOT NULL AND Last_Name IS NOT NULL")
        student_records = cursor.fetchall()
        
        if not student_records:
            logger.warning("No valid student data found in the database")
            return None, None, None, None

        cursor.execute("SELECT DISTINCT Session_ID FROM tbl_Session WHERE Session_ID IS NOT NULL")
        session_rows = cursor.fetchall()
        session_ids = [row[0] for row in session_rows]

        cursor.close()
        conn.close()

        selected_student = random.choice(student_records)
        student_id = selected_student[0]
        first_name = selected_student[1]
        last_name = selected_student[2]
        
        return (
            student_id,
            first_name,
            last_name,
            random.choice(session_ids) if session_ids else None
        )
    except Exception as e:
        logger.error(f"Error fetching student info: {e}")
        return None, None, None, None

# ========================
# Database Manager (Same as before)
# ========================
class DatabaseManager:
    def __init__(self, connection_string, db_name):
        try:
            self.client = pymongo.MongoClient(connection_string)
            self.db = self.client[db_name]
            self.transcripts_collection = self.db["drive"]
            self.test_results_collection = self.db["mock_test_results"]
            logger.info("MongoDB connection established")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            self.client = None
    
    def load_transcript(self):
        try:
            if not self.client:
                return "MongoDB connection not available"
            doc = self.transcripts_collection.find_one({}, sort=[("_id", -1)], projection={"_id": 0, "summary": 1})
            return doc["summary"] if doc and "summary" in doc else "No transcript summary found."
        except Exception as e:
            logger.error(f"Error loading transcript: {e}")
            return f"Error loading transcript: {str(e)}"
    
    def save_test_results(self, test_id: str, test_data: dict, answers_data: list) -> bool:
        try:
            if not self.client:
                logger.error("MongoDB connection not available for saving")
                return False
                
            total_questions = len(answers_data)
            correct_answers = sum(1 for answer in answers_data if answer["correct"])
            score_percentage = (correct_answers / total_questions * 100) if total_questions > 0 else 0
            
            student_id, first_name, last_name, session_id = fetch_random_student_info()
            name = f"{first_name} {last_name}" if first_name and last_name else "Unknown Student"
            if not student_id:
                student_id = "Unknown"
            
            qa_details = []
            for i, answer in enumerate(answers_data):
                qa_details.append({
                    "question_number": i + 1,
                    "question": answer["question"],
                    "user_answer": answer["answer"],
                    "correct": answer["correct"],
                    "feedback": answer["feedback"],
                    "options": answer.get("options", [])
                })
            
            document = {
                "test_id": test_id,
                "timestamp": time.time(),
                "Student_ID": student_id,
                "name": name,
                "session_id": session_id,
                "user_type": test_data["user_type"],
                "final_score": correct_answers,
                "total_questions": total_questions,
                "score_percentage": round(score_percentage, 2),
                "question_types": test_data["question_types"],
                "qa_details": qa_details,
                "test_completed": True
            }
            
            result = self.test_results_collection.insert_one(document)
            logger.info(f"Test results saved: {test_id}, score: {correct_answers}, doc_id: {result.inserted_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving test results for {test_id}: {e}")
            return False
    
    def close(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()

# Initialize database manager
db_manager = DatabaseManager(
    "mongodb://sa:L%40nc%5Eere%400012@192.168.48.200:27017/?authSource=admin", 
    "test"
)

# Initialize Groq client
try:
    groq_client = Groq()
    logger.info("Groq client initialized")
except Exception as e:
    logger.error(f"Groq client initialization failed: {e}")
    groq_client = None

# In-memory storage for active tests
TESTS = {}
ANSWERS = {}

# ========================
# Business Logic Functions (Same as before)
# ========================
def generate_question(user_type, question_type, transcript, difficulty, prev_answer):
    """Generate a question using Groq AI"""
    if not groq_client:
        return "## Question\nGroq client not available. Please check configuration.", None
    
    prompt = (
        "You are a weekly mock test generator creating a single question based on the provided transcript. "
        "Act like an examiner. Generate only the question content. Use markdown: start with '## Question'. "
        "Ensure the response is concise, 70% transcript-based, 30% critical thinking.\n"
        f"Transcript Summary:\n{transcript}"
    )
    
    if user_type == "dev":
        if question_type == "code_writing":
            prompt += "\nGenerate a code writing task with clear input/output requirements."
        elif question_type == "bug_fixing":
            prompt += "\nGenerate buggy code that needs to be fixed."
        else:
            prompt += "\nGenerate a development scenario with expected output."
        options = None
    else:
        prompt += (
            "\nGenerate a multiple-choice question. Format your response EXACTLY as follows:\n"
            "## Question\n[Your question here]\n\n"
            "## Options\nA) [Option A]\nB) [Option B]\nC) [Option C]\nD) [Option D]"
        )

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )
        response = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Question generation error: {e}")
        return "## Question\nError generating question. Please try again.", None

    question = response.strip()
    options = None
    
    if user_type == "non_dev":
        try:
            # Parse options from the response
            if "## Options" in question:
                q_part, opt_part = question.split("## Options", 1)
            elif "Options:" in question:
                q_part, opt_part = question.split("Options:", 1)
            else:
                # Fallback parsing
                lines = question.split('\n')
                q_lines = []
                opt_lines = []
                in_options = False
                
                for line in lines:
                    if re.match(r'^[A-D]\)', line.strip()) or re.match(r'^-\s*[A-D]\)', line.strip()):
                        in_options = True
                    
                    if in_options:
                        opt_lines.append(line)
                    else:
                        q_lines.append(line)
                
                q_part = '\n'.join(q_lines).strip()
                opt_part = '\n'.join(opt_lines).strip()
            
            # Extract options using regex
            option_patterns = [
                r'[A-D]\)\s*(.*?)(?=\n[A-D]\)|$)',
                r'-\s*[A-D]\)\s*(.*?)(?=\n-\s*[A-D]\)|$)',
                r'[A-D]\.\s*(.*?)(?=\n[A-D]\.|$)',
            ]
            
            options = []
            for pattern in option_patterns:
                matches = re.findall(pattern, opt_part, re.DOTALL | re.MULTILINE)
                if matches and len(matches) >= 4:
                    options = [match.strip().replace('\n', ' ') for match in matches[:4]]
                    break
            
            if len(options) != 4:
                options = [
                    "Option A - Default option",
                    "Option B - Default option", 
                    "Option C - Default option",
                    "Option D - Default option"
                ]
            
            question = q_part.strip()
            
        except Exception as e:
            logger.error(f"Option parsing error: {e}")
            options = [
                "Option A - Parsing error",
                "Option B - Parsing error",
                "Option C - Parsing error", 
                "Option D - Parsing error"
            ]
    
    return question.strip(), options

def evaluate_answer(question, answer):
    """Evaluate an answer using Groq AI"""
    if not groq_client:
        return False, "Evaluation service not available"
    
    eval_prompt = f"""
    Evaluate this answer to the question:
    
    Question: {question}
    Answer: {answer}
    
    Respond with either 'Correct', 'Incorrect', or 'Partial' followed by a brief explanation.
    Keep the feedback under 50 words.
    """
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": eval_prompt}],
            temperature=0.3,
            max_completion_tokens=100,
            top_p=1,
            stream=False
        )
        result = completion.choices[0].message.content
        correct = "Correct" in result
        return correct, result
    except Exception as e:
        logger.error(f"Answer evaluation error: {e}")
        return False, f"Evaluation failed: {str(e)}"

def generate_analytics(test_id):
    """Generate analytics from test data"""
    if test_id not in ANSWERS:
        return "No test data found"
    
    answers = ANSWERS[test_id]
    total = len(answers)
    correct = sum(1 for a in answers if a["correct"])
    
    breakdown = []
    for i, a in enumerate(answers):
        status = "✅" if a["correct"] else "❌"
        breakdown.append(f"Q{i+1}: {status}\nFeedback: {a['feedback']}")
    
    return f"**Score: {correct}/{total}**\n\n" + "\n\n".join(breakdown)

def generate_analytics_from_db(doc):
    """Generate analytics from database document"""
    qa_details = doc.get("qa_details", [])
    total = len(qa_details)
    correct = sum(1 for qa in qa_details if qa.get("correct", False))
    
    breakdown = []
    for i, qa in enumerate(qa_details):
        status = "✅" if qa.get("correct", False) else "❌"
        q_num = qa.get("question_number", i+1)
        feedback = qa.get("feedback", "No feedback")
        breakdown.append(f"Q{q_num}: {status}\nFeedback: {feedback}")
    
    return f"**Score: {correct}/{total}**\n\n" + "\n\n".join(breakdown)

# ========================
# FastAPI Application Setup
# ========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Weekend Mock Test API starting up...")
    yield
    logger.info("Weekend Mock Test API shutting down...")
    if db_manager:
        db_manager.close()

# Create FastAPI app for sub-mounting
app = FastAPI(
    title="Weekend Mock Test API",
    description="Sub-application for AI-powered mock testing system",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Mount Static Files for HTML Testing
# ========================
# Mount static files (this allows the HTML file to be served)
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    logger.info(f"Mounted static files from {frontend_dir}")

# ========================
# Routes for Sub-App Structure
# ========================

# Root route of the sub-app (will be available at /weekend_mocktest/)
@app.get("/")
async def sub_app_home():
    """Sub-app home - serve HTML if available, otherwise JSON"""
    html_path = os.path.join(BASE_DIR, "frontend", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {
            "service": "Weekend Mock Test API",
            "version": "2.0.0",
            "status": "running",
            "description": "Sub-application for AI-powered mock testing",
            "endpoints": {
                "health": "/api/health",
                "start_test": "POST /api/test/start",
                "submit_answer": "POST /api/test/submit", 
                "get_results": "GET /api/test/results/{test_id}",
                "download_pdf": "GET /api/test/pdf/{test_id}",
                "all_tests": "GET /api/tests",
                "specific_test": "GET /api/tests/{test_id}"
            },
            "html_interface": "/static/index.html" if os.path.exists(html_path) else "Not available",
            "timestamp": time.time()
        }

# ========================
# API Endpoints (Fixed paths for sub-app)
# ========================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        message="Weekend Mock Test API is running",
        timestamp=time.time(),
        active_tests=len(TESTS)
    )

@app.post("/api/test/start", response_model=TestResponse)
async def start_test(request: StartTestRequest):
    """Start a new test"""
    if request.user_type not in ["dev", "non_dev"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid user type. Must be 'dev' or 'non_dev'"
        )

    test_id = str(uuid.uuid4())
    transcript_content = db_manager.load_transcript()
    
    question_types = (
        ["code_writing"] * 4 + ["bug_fixing"] * 3 + ["scenario"] * 3
        if request.user_type == "dev"
        else ["mcq"] * 5 + ["scenario_mcq"] * 5
    )
    random.shuffle(question_types)
    
    TESTS[test_id] = {
        "user_type": request.user_type, 
        "score": 0, 
        "question_count": 1, 
        "question_types": question_types,
        "created_at": time.time()
    }
    ANSWERS[test_id] = []

    q_type = question_types[0]
    question, options = generate_question(
        request.user_type, 
        q_type, 
        transcript_content, 
        difficulty=1, 
        prev_answer=None
    )
    
    ANSWERS[test_id].append({
        "question": question, 
        "answer": "", 
        "correct": False, 
        "options": options or [], 
        "feedback": ""
    })

    logger.info(f"Test started: {test_id}, user_type: {request.user_type}")
    
    return TestResponse(
        test_id=test_id,
        user_type=request.user_type,
        question_number=1,
        total_questions=2,
        question_html=markdown.markdown(question),
        options=options,
        time_limit=300 if request.user_type == "dev" else 120
    )

@app.post("/api/test/submit", response_model=SubmitAnswerResponse)
async def submit_answer(request: SubmitAnswerRequest):
    """Submit an answer"""
    test = TESTS.get(request.test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    
    user_type = test["user_type"]
    count = test["question_count"]
    
    if request.question_number != count:
        raise HTTPException(status_code=400, detail="Invalid question number")

    ans_data = ANSWERS[request.test_id][-1]
    options = ans_data["options"]
    full_answer = request.answer
    
    # Convert option index to full answer for non_dev
    if user_type == "non_dev" and request.answer.isdigit():
        try:
            option_index = int(request.answer)
            if 0 <= option_index < len(options):
                full_answer = options[option_index]
        except (ValueError, IndexError):
            pass

    # Evaluate the answer
    correct, feedback = evaluate_answer(ans_data['question'], full_answer)
    
    ans_data.update({
        "answer": full_answer, 
        "correct": correct, 
        "feedback": feedback
    })
    
    if correct:
        test["score"] += 1

    logger.info(f"Answer submitted for {request.test_id}, correct: {correct}")

    # Check if test is completed
    if count >= 2:
        save_success = db_manager.save_test_results(
            test_id=request.test_id,
            test_data=test,
            answers_data=ANSWERS[request.test_id]
        )
        
        if not save_success:
            logger.warning(f"Failed to save test results for {request.test_id}")
        
        analytics = generate_analytics(request.test_id)
        
        logger.info(f"Test completed: {request.test_id}, score: {test['score']}")
        
        return SubmitAnswerResponse(
            test_completed=True,
            score=test["score"],
            total_questions=2,
            analytics=analytics,
            pdf_available=True
        )

    # Generate next question
    test["question_count"] += 1
    next_type_index = count % len(test["question_types"])
    next_type = test["question_types"][next_type_index]
    transcript = db_manager.load_transcript()
    
    next_question, next_options = generate_question(
        user_type, 
        next_type, 
        transcript, 
        difficulty=test["score"] // 2 + 1, 
        prev_answer=full_answer
    )
    
    ANSWERS[request.test_id].append({
        "question": next_question, 
        "answer": "", 
        "correct": False, 
        "options": next_options or [], 
        "feedback": ""
    })

    return SubmitAnswerResponse(
        test_completed=False,
        next_question=NextQuestionResponse(
            question_number=test["question_count"],
            total_questions=2,
            question_html=markdown.markdown(next_question),
            options=next_options,
            time_limit=300 if user_type == "dev" else 120
        )
    )

@app.get("/api/test/results/{test_id}", response_model=TestResultsResponse)
async def get_test_results(test_id: str):
    """Get test results"""
    try:
        doc = db_manager.test_results_collection.find_one({"test_id": test_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Test results not found")
        
        analytics = generate_analytics_from_db(doc)
        
        return TestResultsResponse(
            test_id=test_id,
            score=doc.get("final_score", 0),
            total_questions=doc.get("total_questions", 0),
            analytics=analytics,
            pdf_available=True
        )
    except Exception as e:
        logger.error(f"Error fetching results for {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch test results")

@app.get("/api/test/pdf/{test_id}")
async def download_results_pdf(test_id: str):
    """Download test results as PDF"""
    try:
        doc = db_manager.test_results_collection.find_one({"test_id": test_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Test ID not found")

        # Generate PDF
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=LETTER)
        width, height = LETTER
        margin = 50
        y = height - margin

        def write_line(label, value, indent=0):
            nonlocal y
            if y < margin + 50:
                p.showPage()
                y = height - margin
                p.setFont("Helvetica", 12)
            p.drawString(margin + indent, y, f"{label}: {value}")
            y -= 20

        # PDF Content
        p.setFont("Helvetica-Bold", 14)
        p.drawString(margin, y, f"Mock Test Results - {test_id}")
        y -= 30

        p.setFont("Helvetica", 12)
        write_line("Name", doc.get("name", "N/A"))
        write_line("Student ID", str(doc.get("Student_ID", "N/A")))
        write_line("Session ID", str(doc.get("session_id", "N/A")))
        write_line("User Type", doc.get("user_type", "N/A"))
        write_line("Score", f"{doc.get('final_score', 0)}/{doc.get('total_questions', 0)}")
        write_line("Percentage", f"{doc.get('score_percentage', 0)}%")
        
        try:
            ts = float(doc.get("timestamp", time.time()))
            timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        except:
            timestr = "N/A"
        write_line("Date", timestr)

        # Q&A Details
        y -= 20
        p.setFont("Helvetica-Bold", 12)
        p.drawString(margin, y, "Question Details:")
        y -= 20

        p.setFont("Helvetica", 10)
        for idx, entry in enumerate(doc.get("qa_details", []), start=1):
            if y < margin + 100:
                p.showPage()
                y = height - margin
                p.setFont("Helvetica", 10)
            
            write_line(f"Q{idx}", entry.get("question", "N/A")[:80] + "...")
            write_line("Answer", entry.get("user_answer", "N/A")[:80] + "...")
            write_line("Result", "✓ Correct" if entry.get("correct") else "✗ Incorrect")
            write_line("Feedback", entry.get("feedback", "N/A")[:60] + "...")
            y -= 10

        p.showPage()
        p.save()
        buffer.seek(0)

        filename = f"mock_test_results_{test_id}.pdf"
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"PDF generation error for {test_id}: {e}")
        raise HTTPException(status_code=500, detail="PDF generation failed")

@app.get("/api/tests", response_model=AllTestsResponse)
async def get_all_tests():
    """Get all tests"""
    try:
        results = list(db_manager.test_results_collection.find(
            {},
            {"_id": 0, "timestamp": 0, "question_types": 0, "qa_details": 0}
        ))
        
        test_records = []
        for result in results:
            test_records.append(TestRecord(
                test_id=result.get("test_id", ""),
                Student_ID=str(result.get("Student_ID", "")),
                name=result.get("name", ""),
                session_id=str(result.get("session_id", "")),
                user_type=result.get("user_type", ""),
                final_score=result.get("final_score", 0),
                total_questions=result.get("total_questions", 0),
                score_percentage=result.get("score_percentage", 0.0),
                test_completed=result.get("test_completed", False)
            ))
        
        return AllTestsResponse(count=len(test_records), results=test_records)
    except Exception as e:
        logger.error(f"Error fetching all tests: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tests")

@app.get("/api/tests/{test_id}", response_model=TestRecord)
async def get_test_by_id(test_id: str):
    """Get specific test by ID"""
    try:
        result = db_manager.test_results_collection.find_one(
            {"test_id": test_id},
            {"_id": 0, "timestamp": 0, "question_types": 0, "qa_details": 0}
        )
        if not result:
            raise HTTPException(status_code=404, detail="Test not found")
        
        return TestRecord(
            test_id=result.get("test_id", ""),
            Student_ID=str(result.get("Student_ID", "")),
            name=result.get("name", ""),
            session_id=str(result.get("session_id", "")),
            user_type=result.get("user_type", ""),
            final_score=result.get("final_score", 0),
            total_questions=result.get("total_questions", 0),
            score_percentage=result.get("score_percentage", 0.0),
            test_completed=result.get("test_completed", False)
        )
    except Exception as e:
        logger.error(f"Error fetching test {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch test")

@app.delete("/api/cleanup")
async def cleanup_resources():
    """Clean up expired tests and resources"""
    try:
        current_time = time.time()
        expired_tests = []
        
        # Clean up tests older than 1 hour
        for test_id, test_data in list(TESTS.items()):
            if current_time - test_data.get("created_at", 0) > 3600:  # 1 hour
                expired_tests.append(test_id)
                TESTS.pop(test_id, None)
                ANSWERS.pop(test_id, None)
        
        logger.info(f"Cleaned up {len(expired_tests)} expired tests")
        
        return {
            "message": "Cleanup completed",
            "tests_cleaned": len(expired_tests),
            "active_tests": len(TESTS)
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail="Cleanup failed")

# Debug endpoint for development
@app.get("/api/debug/status")
async def debug_status():
    """Debug endpoint to check API status"""
    return {
        "api_status": "running",
        "timestamp": time.time(),
        "active_tests": len(TESTS),
        "groq_available": groq_client is not None,
        "mongodb_available": db_manager.client is not None,
        "environment": "development",
        "base_dir": BASE_DIR,
        "frontend_available": os.path.exists(os.path.join(BASE_DIR, "frontend", "index.html"))
    }