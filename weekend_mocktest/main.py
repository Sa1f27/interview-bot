# REST API FastAPI Mock Test Application
from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import markdown
import uuid
import random
import re
import pymongo
from groq import Groq
import os
import textwrap 
import time
import logging
import pyodbc
from contextlib import asynccontextmanager
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
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

class TestResultsResponse(BaseModel):
    test_id: str
    score: int
    total_questions: int
    analytics: str
    pdf_available: bool

class QuestionData(BaseModel):
    question: str
    answer: str
    correct: bool
    options: List[str]
    feedback: str

# --- Setup ---
BASE_DIR = os.path.dirname(__file__)

# Initialize database manager
db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("Mock test application starting up...")
    yield
    # Shutdown
    if db_manager:
        db_manager.close()
    logger.info("Mock test application shut down")

app = FastAPI(
    title="Weekend Mock Test API",
    description="REST API for mock testing with AI-generated questions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "frontend"))

# Groq Client
client = Groq()

# Jinja2 A-D option filter
def to_letter(index: int) -> str:
    return chr(65 + index)
templates.env.filters['to_letter'] = to_letter

# SQL Server connection parameters
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

# Fetch a random student ID, name, and session ID from SQL Server
def fetch_random_student_info():
    """Fetch a random ID, name from tbl_Student and session_id from session table from SQL Server"""
    try:
        conn = get_db_connection()
        if not conn:
            return None, None, None, None
        
        cursor = conn.cursor()
        # Fetch all student records (ID, First_Name, Last_Name)
        cursor.execute("SELECT ID, First_Name, Last_Name FROM tbl_Student WHERE ID IS NOT NULL AND First_Name IS NOT NULL AND Last_Name IS NOT NULL")
        
        student_records = cursor.fetchall()
        
        if not student_records:
            logger.warning("No valid student data found in the database")
            return None, None, None, None

        # Fetch distinct Session_ID
        cursor.execute("SELECT DISTINCT Session_ID FROM tbl_Session WHERE Session_ID IS NOT NULL")
        session_rows = cursor.fetchall()
        session_ids = [row[0] for row in session_rows]

        cursor.close()
        conn.close()

        # Randomly select one student record
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

# MongoDB Manager Class
class DatabaseManager:
    """MongoDB database manager for mock test application"""
    def __init__(self, connection_string, db_name):
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[db_name]
        self.transcripts_collection = self.db["drive"]
        self.test_results_collection = self.db["mock_test_results"]
    
    def load_transcript(self):
        """Fetch the latest lecture summary from the database"""
        try:
            doc = self.transcripts_collection.find_one({}, sort=[("_id", -1)], projection={"_id": 0, "summary": 1})
            return doc["summary"] if doc and "summary" in doc else "Summary not found."
        except Exception as e:
            return f"Error loading summary: {str(e)}"
    
    def save_test_results(self, test_id: str, test_data: dict, answers_data: list) -> bool:
        """Save test results to the test_results collection"""
        try:
            # Calculate additional metrics
            total_questions = len(answers_data)
            correct_answers = sum(1 for answer in answers_data if answer["correct"])
            score_percentage = (correct_answers / total_questions * 100) if total_questions > 0 else 0
            
            # Fetch student ID from SQL Server
            student_id, first_name, last_name, session_id = fetch_random_student_info()
            if student_id is None:
                logger.warning("Could not fetch student ID from SQL Server")
                
            name = f"{first_name} {last_name}" if first_name and last_name else "Unknown Student"
            if not student_id:
                student_id = "Unknown Student ID"
            
            # Prepare question-answer details
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
            
            # Create the document to insert
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
            
            # Insert into the test_results collection
            result = self.test_results_collection.insert_one(document)
            logger.info(f"Test results saved successfully for test {test_id}, name: {name}, Student_ID: {student_id}, score: {correct_answers}, session_id: {session_id}, document ID: {result.inserted_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving test results for test {test_id}: {e}")
            return False
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'client'):
            self.client.close()

# Initialize database manager
db_manager = DatabaseManager(
    "mongodb://sa:L%40nc%5Eere%400012@192.168.48.200:27017/?authSource=admin", 
    "test"
)

# Helper function to load transcript using the database manager
def load_transcript():
    return db_manager.load_transcript()

# In-memory storage
TESTS = {}
ANSWERS = {}

# ============= REST API ENDPOINTS =============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Mock test API is running"}

@app.post("/api/test/start", response_model=TestResponse)
async def start_test_api(request: StartTestRequest):
    """Start a new test - REST API endpoint"""
    if request.user_type not in ["dev", "non_dev"]:
        raise HTTPException(status_code=400, detail="Invalid user type. Must be 'dev' or 'non_dev'")

    test_id = str(uuid.uuid4())
    transcript_content = load_transcript()
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
        "question_types": question_types
    }
    ANSWERS[test_id] = []

    q_type = question_types[0]
    question, options = generate_question(request.user_type, q_type, transcript_content, difficulty=1, prev_answer=None)
    ANSWERS[test_id].append({
        "question": question, 
        "answer": "", 
        "correct": False, 
        "options": options or [], 
        "feedback": ""
    })

    return TestResponse(
        test_id=test_id,
        user_type=request.user_type,
        question_number=1,
        total_questions=2,
        question_html=markdown.markdown(question),
        options=options,
        time_limit=300 if request.user_type == "dev" else 120
    )

@app.post("/api/test/submit", response_model=Dict[str, Any])
async def submit_answer_api(request: SubmitAnswerRequest):
    """Submit an answer - REST API endpoint"""
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
    
    if user_type == "non_dev" and request.answer.isdigit() and int(request.answer) < len(options):
        full_answer = options[int(request.answer)]

    # Evaluate the answer
    eval_prompt = f"Evaluate answer: {ans_data['question']}\nAnswer: {full_answer}\nReturn 'Correct', 'Incorrect', or 'Partial' with a concise comment."
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": eval_prompt}],
            temperature=1,
            max_completion_tokens=100,
            top_p=1,
            stream=False
        )
        result = completion.choices[0].message.content
        correct = "Correct" in result
    except Exception as e:
        result = f"Evaluation failed: {str(e)}"
        correct = False

    ans_data.update({"answer": full_answer, "correct": correct, "feedback": result})
    if correct:
        test["score"] += 1

    # Check if test is completed
    if count >= 2:
        # Save test results to MongoDB
        save_success = db_manager.save_test_results(
            test_id=request.test_id,
            test_data=test,
            answers_data=ANSWERS[request.test_id]
        )
        
        if not save_success:
            logger.warning(f"Failed to save test results for test {request.test_id}")
        
        analytics = generate_analytics(request.test_id)
        return {
            "test_completed": True,
            "score": test["score"],
            "total_questions": 2,
            "analytics": analytics,
            "pdf_available": True
        }

    # Generate next question
    test["question_count"] += 1
    next_type = test["question_types"][count % len(test["question_types"])]
    transcript = load_transcript()
    q, opts = generate_question(user_type, next_type, transcript, test["score"] // 2 + 1, full_answer)
    ANSWERS[request.test_id].append({
        "question": q, 
        "answer": "", 
        "correct": False, 
        "options": opts or [], 
        "feedback": ""
    })

    return {
        "test_completed": False,
        "next_question": {
            "question_number": test["question_count"],
            "total_questions": 2,
            "question_html": markdown.markdown(q),
            "options": opts,
            "time_limit": 300 if user_type == "dev" else 120
        }
    }

@app.get("/api/test/results/{test_id}", response_model=TestResultsResponse)
async def get_test_results_api(test_id: str):
    """Get test results - REST API endpoint"""
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

@app.get("/api/test/pdf/{test_id}")
async def download_results_pdf_api(test_id: str):
    """Download test results as PDF - REST API endpoint"""
    doc = db_manager.test_results_collection.find_one({"test_id": test_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Test ID not found in database")

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

    p.setFont("Helvetica-Bold", 14)
    p.drawString(margin, y, f"Mock Test Results - Test ID: {test_id}")
    y -= 30

    p.setFont("Helvetica", 12)
    write_line("Name", doc.get("name", "N/A"))
    write_line("Student_ID", str(doc.get("Student_ID", "N/A")))
    write_line("Session_ID", str(doc.get("session_id", "N/A")))
    try:
        ts = float(doc.get("timestamp", time.time()))
        timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except:
        timestr = "N/A"
    write_line("Saved At", timestr)
    write_line("Score", f"{doc.get('final_score', 0)}/{doc.get('total_questions', 0)}")
    write_line("Score %", doc.get("score_percentage", "N/A"))
    write_line("User Type", doc.get("user_type", "N/A"))
    y -= 10

    p.setFont("Helvetica-Bold", 12)
    if y < margin + 30:
        p.showPage()
        y = height - margin
    p.drawString(margin, y, "Q&A Details:")
    y -= 20

    p.setFont("Helvetica", 11)
    for idx, entry in enumerate(doc.get("qa_details", []), start=1):
        if y < margin + 80:
            p.showPage()
            y = height - margin
            p.setFont("Helvetica", 11)
        write_line(f"{idx}. Question", entry.get("question", "N/A"))
        write_line("   User Answer", entry.get("user_answer", "N/A"), indent=10)
        write_line("   Correct", str(entry.get("correct", "N/A")), indent=10)
        write_line("   Feedback", entry.get("feedback", "N/A"), indent=10)
        options = entry.get("options", [])
        if options:
            write_line("   Options", ", ".join(options), indent=10)
        y -= 5

    p.showPage()
    p.save()
    buffer.seek(0)
    filename = f"mock_test_results_{test_id}.pdf"

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ============= FRONTEND ROUTES (Original HTML endpoints) =============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Frontend home page"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "page": "start"
    })

@app.post("/start-test", response_class=HTMLResponse)
async def start_test_frontend(request: Request, user_type: str = Form(...)):
    """Frontend start test endpoint"""
    # Use the API endpoint internally
    try:
        start_request = StartTestRequest(user_type=user_type)
        test_response = await start_test_api(start_request)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "page": "test",
            "test_id": test_response.test_id,
            "user_type": test_response.user_type,
            "question_html": test_response.question_html,
            "options": test_response.options,
            "question_number": test_response.question_number,
            "total_questions": test_response.total_questions,
            "time_limit": test_response.time_limit
        })
    except HTTPException as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "page": "error",
            "message": e.detail
        })

@app.post("/submit-answer", response_class=HTMLResponse)
async def submit_answer_frontend(request: Request, test_id: str = Form(...), answer: str = Form(default=""), question_number: int = Form(...)):
    """Frontend submit answer endpoint"""
    try:
        submit_request = SubmitAnswerRequest(
            test_id=test_id,
            question_number=question_number,
            answer=answer
        )
        result = await submit_answer_api(submit_request)
        
        if result["test_completed"]:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "page": "results",
                "score": result["score"],
                "analytics": result["analytics"],
                "test_id": test_id,
                "pdf_url": f"./download_results/{test_id}"
            })
        else:
            next_q = result["next_question"]
            test = TESTS.get(test_id)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "page": "test",
                "test_id": test_id,
                "user_type": test["user_type"],
                "question_html": next_q["question_html"],
                "options": next_q["options"],
                "question_number": next_q["question_number"],
                "total_questions": next_q["total_questions"],
                "time_limit": next_q["time_limit"]
            })
    except HTTPException as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "page": "error",
            "message": e.detail
        })

@app.get("/download_results/{test_id}")
async def download_results_frontend(test_id: str):
    """Frontend download results endpoint"""
    return await download_results_pdf_api(test_id)

# ============= HELPER FUNCTIONS =============

def generate_question(user_type, question_type, transcript, difficulty, prev_answer):
    """Generate a question using Groq AI"""
    prompt = (
        "You are a weekly mock test generator creating a single question based on the provided transcript. "
        "Act like an examiner. Generate only the question content. Use markdown: start with '## Question'. "
        "Ensure the response is concise, 70% transcript-based, 30% critical thinking.\n"
        f"Transcript Summary:\n{transcript}"
    )
    if user_type == "dev":
        if question_type == "code_writing":
            prompt += "\nGenerate a code writing task with I/O."
        elif question_type == "bug_fixing":
            prompt += "\nGenerate buggy code to fix."
        else:
            prompt += "\nGenerate a dev scenario with expected output."
        options = None
    else:
        prompt += (
            "\nGenerate a multiple-choice question. Format your response EXACTLY as follows:\n"
            "## Question\n[Your question here]\n\n"
            "## Options\nA) [Option A]\nB) [Option B]\nC) [Option C]\nD) [Option D]"
        )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )
        response = completion.choices[0].message.content
    except Exception:
        return "## Question\nError generating question.", None

    question = response.strip()
    options = None
    
    if user_type == "non_dev":
        try:
            # Split by "## Options" or "Options:" to separate question from options
            if "## Options" in question:
                q_part, opt_part = question.split("## Options", 1)
            elif "Options:" in question:
                q_part, opt_part = question.split("Options:", 1)
            else:
                # If no clear separation, try to find the options pattern
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
            
            # Extract options using regex patterns
            option_patterns = [
                r'[A-D]\)\s*(.*?)(?=\n[A-D]\)|$)',  # A) Option text
                r'-\s*[A-D]\)\s*(.*?)(?=\n-\s*[A-D]\)|$)',  # - A) Option text
                r'[A-D]\.\s*(.*?)(?=\n[A-D]\.|$)',  # A. Option text
            ]
            
            options = []
            for pattern in option_patterns:
                matches = re.findall(pattern, opt_part, re.DOTALL | re.MULTILINE)
                if matches and len(matches) >= 4:
                    options = [match.strip().replace('\n', ' ') for match in matches[:4]]
                    break
            
            # If we couldn't extract exactly 4 options, create default ones
            if len(options) != 4:
                options = [
                    "Option A - Please regenerate question",
                    "Option B - Please regenerate question", 
                    "Option C - Please regenerate question",
                    "Option D - Please regenerate question"
                ]
            
            question = q_part.strip()
            
        except Exception as e:
            # Fallback options
            options = [
                "Option A - Error parsing options",
                "Option B - Error parsing options",
                "Option C - Error parsing options", 
                "Option D - Error parsing options"
            ]
    
    return question.strip(), options

def generate_analytics(test_id):
    """Generate analytics from in-memory test data"""
    answers = ANSWERS[test_id]
    total = len(answers)
    correct = sum(1 for a in answers if a["correct"])
    breakdown = "\n\n".join(
        f"Q{i+1}: {'✅' if a['correct'] else '❌'}\nFeedback: {a['feedback']}"
        for i, a in enumerate(answers)
    )
    return f"**Score: {correct}/{total}**\n\n{breakdown}"

def generate_analytics_from_db(doc):
    """Generate analytics from database document"""
    qa_details = doc.get("qa_details", [])
    total = len(qa_details)
    correct = sum(1 for qa in qa_details if qa.get("correct", False))
    breakdown = "\n\n".join(
        f"Q{qa.get('question_number', i+1)}: {'✅' if qa.get('correct', False) else '❌'}\nFeedback: {qa.get('feedback', 'No feedback')}"
        for i, qa in enumerate(qa_details)
    )
    return f"**Score: {correct}/{total}**\n\n{breakdown}"


@app.get("/api/tests")
async def get_all_tests():
    try:
        results = list(db_manager.test_results_collection.find(
            {},
            {"_id": 0, "timestamp": 0, "question_types": 0, "qa_details": 0}
        ))
        return JSONResponse(content={"count": len(results), "results": results})
    except Exception as e:
        logger.error(f"Error fetching test results: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch test results")

@app.get("/api/tests/{test_id}")
async def get_test_by_id(test_id: str):
    try:
        result = db_manager.test_results_collection.find_one(
            {"test_id": test_id},
            {"_id": 0, "timestamp": 0, "question_types": 0, "qa_details": 0}
        )
        if not result:
            raise HTTPException(status_code=404, detail="Test result not found")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error fetching test ID {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch test result")

