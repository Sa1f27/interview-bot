# Simplified FastAPI Mock Test Application
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import markdown
from fpdf import FPDF
import uuid
import random
import re
import pymongo
from groq import Groq
import os
import time
import logging
import pyodbc
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Setup ---
BASE_DIR = os.path.dirname(__file__)

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 template directory
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

def fetch_random_student_id():
    try:
        conn = get_db_connection()
        if not conn:
            return None
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT Student_ID FROM tbl_Student_Info")
        rows = cursor.fetchall()
        student_ids = [row[0] for row in rows if row[0] is not None]
        cursor.close()
        conn.close()
        return random.choice(student_ids) if student_ids else None
    except Exception as e:
        logger.error(f"Error fetching student ID: {e}")
        return None
    
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
    
    def save_test_results(self, session_id: str, session_data: dict, answers_data: list) -> bool:
        """Save test results to the test_results collection"""
        try:
            # Calculate additional metrics
            total_questions = len(answers_data)
            correct_answers = sum(1 for answer in answers_data if answer["correct"])
            score_percentage = (correct_answers / total_questions * 100) if total_questions > 0 else 0
            
            # Fetch student ID from SQL Server
            student_id = fetch_random_student_id()
            if student_id is None:
                logger.warning("Could not fetch student ID from SQL Server")
                
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
                "session_id": session_id,
                "timestamp": time.time(),
                "student_id": student_id,
                "user_type": session_data["user_type"],
                "final_score": correct_answers,
                "total_questions": total_questions,
                "score_percentage": round(score_percentage, 2),
                "question_types": session_data["question_types"],
                "qa_details": qa_details,
                "test_completed": True
            }
            
            # Insert into the test_results collection
            result = self.test_results_collection.insert_one(document)
            logger.info(f"Test results saved successfully for session {session_id}, document ID: {result.inserted_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving test results for session {session_id}: {e}")
            return False
    
    def close(self):
        """Close the database connection"""
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
SESSIONS = {}
ANSWERS = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "page": "start"
    })

@app.post("/start-test", response_class=HTMLResponse)
async def start_test(request: Request, user_type: str = Form(...)):
    if user_type not in ["dev", "non_dev"]:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "page": "error",
            "message": "Invalid user type"
        })

    session_id = str(uuid.uuid4())
    transcript_content = load_transcript()
    question_types = (
        ["code_writing"] * 4 + ["bug_fixing"] * 3 + ["scenario"] * 3
        if user_type == "dev"
        else ["mcq"] * 5 + ["scenario_mcq"] * 5
    )
    random.shuffle(question_types)
    SESSIONS[session_id] = {
        "user_type": user_type, 
        "score": 0, 
        "question_count": 1, 
        "question_types": question_types
    }
    ANSWERS[session_id] = []

    q_type = question_types[0]
    question, options = generate_question(user_type, q_type, transcript_content, difficulty=1, prev_answer=None)
    ANSWERS[session_id].append({
        "question": question, 
        "answer": "", 
        "correct": False, 
        "options": options or [], 
        "feedback": ""
    })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "test",
        "session_id": session_id,
        "user_type": user_type,
        "question_html": markdown.markdown(question),
        "options": options,
        "question_number": 1,
        "total_questions": 10,
        "time_limit": 300 if user_type == "dev" else 120
    })

def generate_question(user_type, question_type, transcript, difficulty, prev_answer):
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

@app.post("/submit-answer", response_class=HTMLResponse)
async def submit_answer(request: Request, session_id: str = Form(...), answer: str = Form(default=""), question_number: int = Form(...)):
    session = SESSIONS.get(session_id)
    if not session:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "page": "error",
            "message": "Session not found"
        })
    
    user_type = session["user_type"]
    score = session["score"]
    count = session["question_count"]
    if question_number != count:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "page": "error",
            "message": "Invalid question number"
        })

    ans_data = ANSWERS[session_id][-1]
    options = ans_data["options"]
    full_answer = answer
    if user_type == "non_dev" and answer.isdigit() and int(answer) < len(options):
        full_answer = options[int(answer)]

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
        session["score"] += 1

    # Test completed
    if count >= 10:
        # Save test results to MongoDB
        save_success = db_manager.save_test_results(
            session_id=session_id,
            session_data=session,
            answers_data=ANSWERS[session_id]
        )
        
        if not save_success:
            logger.warning(f"Failed to save test results for session {session_id}")
        
        analytics = generate_analytics(session_id)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "page": "results",
            "score": session["score"],
            "analytics": analytics,
            "session_id": session_id
        })

    # Next question
    session["question_count"] += 1
    next_type = session["question_types"][count % len(session["question_types"])]
    transcript = load_transcript()
    q, opts = generate_question(user_type, next_type, transcript, session["score"] // 2 + 1, full_answer)
    ANSWERS[session_id].append({
        "question": q, 
        "answer": "", 
        "correct": False, 
        "options": opts or [], 
        "feedback": ""
    })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "test",
        "session_id": session_id,
        "user_type": user_type,
        "question_html": markdown.markdown(q),
        "options": opts,
        "question_number": session["question_count"],
        "total_questions": 10,
        "time_limit": 300 if user_type == "dev" else 120
    })

def generate_analytics(session_id):
    answers = ANSWERS[session_id]
    total = len(answers)
    correct = sum(1 for a in answers if a["correct"])
    breakdown = "\n\n".join(
        f"Q{i+1}: {'✅' if a['correct'] else '❌'}\nFeedback: {a['feedback']}"
        for i, a in enumerate(answers)
    )
    return f"**Score: {correct}/{total}**\n\n{breakdown}"

@app.get("/export-results", response_class=FileResponse)
async def export_results(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Mock Test Results", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Final Score: {session['score']}/10", ln=True)
    pdf.ln(10)
    
    # Add detailed analytics
    analytics_text = generate_analytics(session_id).replace("**", "").replace("✅", "Correct").replace("❌", "Incorrect")
    pdf.multi_cell(0, 10, txt=analytics_text)
    
    pdf_path = f"test_results_{session_id}.pdf"
    pdf.output(pdf_path)
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path)

# Handle application shutdown
@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources on application shutdown"""
    db_manager.close()