import uuid
import sqlite3
import os
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from groq import Groq
import markdown
from fpdf import FPDF
import random
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Register custom Jinja2 filter
def to_letter(index: int) -> str:
    """Convert index (0-3) to option letter (A-D)."""
    return chr(65 + index)  # 65 is ASCII for 'A'

templates.env.filters['to_letter'] = to_letter

client = Groq()

# Hardcoded transcript file path (replace with your actual path)
TRANSCRIPT_FILE_PATH = os.path.join("uploads", "transcript-test.txt")
# SQLite database setup
DB_FILE = "mock_test.db"
if not os.path.exists(DB_FILE):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE sessions (id TEXT PRIMARY KEY, user_type TEXT, question_count INTEGER, score INTEGER, question_types TEXT)''')
    c.execute('''CREATE TABLE answers (session_id TEXT, question_number INTEGER, question_text TEXT, answer TEXT, is_correct BOOLEAN, options TEXT, feedback TEXT)''')
    conn.commit()
    conn.close()

# Cached transcript
transcript_cache = None

def load_transcript():
    global transcript_cache
    if transcript_cache is None:
        try:
            with open(TRANSCRIPT_FILE_PATH, "r") as file:
                full_transcript = file.read()
            summary_prompt = (
                f"Summarize the following transcript into 100-200 words, capturing key points about programming and project management: {full_transcript}"
            )
            try:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": summary_prompt}],
                    temperature=0.5,
                    max_completion_tokens=300,
                    top_p=1,
                    stream=False
                )
                transcript_cache = completion.choices[0].message.content
            except Exception as e:
                transcript_cache = "Summary failed. Sample content about programming and project management."
        except FileNotFoundError:
            transcript_cache = "Sample transcript content about programming and project management."
    return transcript_cache

# Helper functions
def generate_id():
    return str(uuid.uuid4())

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def get_session(session_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session = c.fetchone()
    conn.close()
    return session

def update_session(session_id, question_count, score):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE sessions SET question_count = ?, score = ? WHERE id = ?", (question_count, score, session_id))
    conn.commit()
    conn.close()

def save_answer(session_id, question_number, question_text, answer, is_correct, options="", feedback=""):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO answers (session_id, question_number, question_text, answer, is_correct, options, feedback) VALUES (?, ?, ?, ?, ?, ?, ?)", 
              (session_id, question_number, question_text, answer, is_correct, options, feedback))
    conn.commit()
    conn.close()

def get_question(session_id, question_number):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT question_text FROM answers WHERE session_id = ? AND question_number = ?", (session_id, question_number))
    result = c.fetchone()
    conn.close()
    return result["question_text"] if result else None

# Home page to select user type
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("start_test.html", {"request": request})

# Endpoint to start a new test session
@app.post("/start-test", response_class=HTMLResponse)
async def start_test(request: Request, user_type: str = Form(...)):
    if user_type not in ["dev", "non_dev"]:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Invalid user type"})
    
    session_id = generate_id()
    transcript_content = load_transcript()
    
    # Insert new session into database
    conn = get_db_connection()
    c = conn.cursor()
    question_types = (
        ["code_writing"] * 4 + ["bug_fixing"] * 3 + ["scenario"] * 3
        if user_type == "dev"
        else ["mcq"] * 5 + ["scenario_mcq"] * 5
    )
    random.shuffle(question_types)
    question_types_str = ",".join(question_types)
    c.execute("INSERT INTO sessions (id, user_type, question_count, score, question_types) VALUES (?, ?, ?, ?, ?)", 
              (session_id, user_type, 1, 0, question_types_str))
    conn.commit()
    conn.close()
    
    # Generate first question
    question, options, hint = generate_question(user_type, question_types[0], transcript_content, difficulty=1, prev_answer=None)
    
 
    # Save question
    options_str = "|".join(options) if options else ""
    save_answer(session_id, 1, question, "", False, options_str)
    
    return templates.TemplateResponse(
        "test.html",
        {
            "request": request, "session_id": session_id, "user_type": user_type,
            "question_html": markdown.markdown(question), "options": options, "hint": hint,
            "question_number": 1, "total_questions": 10, "time_limit": 300 if user_type == "dev" else 120
        }
    )

# Function to generate questions based on type and difficulty
def generate_question(user_type, question_type, transcript, difficulty, prev_answer):
    prompt = (
        "You are a weekly mock test generator creating a single question based on the provided transcript. act like an examiner. "
        "Generate only the question content as specified, with no text before or after. "
        "Use markdown: start with '## Question' followed by the question text. "
        "Ensure the response is concise, clear, and 70% transcript-based, 30% critical thinking. "
        "For answers like 'I don’t know', 'Sorry', or 'I’m not sure', acknowledge and proceed. "
        "End with 'Hint:' followed by a subtle hint on a new line."
    )
    
    if user_type == "dev":
        if question_type == "code_writing":
            prompt += (
                f"Generate a code writing exercise for a developer at difficulty level {difficulty}. "
                "Provide a clear task under '## Question' requiring a function or script. "
                "Include input/output expectations in a ``` code block. "
                "Format code with ``` ... ``` for proper syntax highlighting. "
                "No options or multiple-choice elements."
            )
        elif question_type == "bug_fixing":
            prompt += (
                f"Generate a bug fixing exercise for a developer at difficulty level {difficulty}. "
                "Provide dummy code with 1-3 errors in a ``` code block under '## Question'. "
                "Include a task description explaining the intended functionality. "
                "Format code with ``` ... ``` for proper syntax highlighting. "
                "No options or multiple-choice elements."
            )
        else:  # scenario
            prompt += (
                f"Generate a real-world problem-solving scenario for a developer at difficulty level {difficulty}. "
                "Provide a scenario description under '## Question' requiring a code solution. "
                "Include input/output expectations in a ``` code block. "
                "Format code with ``` ... ``` for proper syntax highlighting. "
                "No options or multiple-choice elements."
            )
        options = None
    else:
        if question_type == "mcq":
            prompt += (
                f"Generate a standard multiple-choice question for a non-developer at difficulty level {difficulty}. "
                "Provide the question text under '## Question'. "
                "Include exactly four options under 'Options:' as a markdown list: "
                "- A) [option text]\n- B) [option text]\n- C) [option text]\n- D) [option text]. "
                "Ensure options are concise, distinct, and vertically aligned."
            )
        else:  # scenario_mcq
            prompt += (
                f"Generate a scenario-based multiple-choice question for a non-developer at difficulty level {difficulty}. "
                "Provide a brief scenario and question text under '## Question'. "
                "Include exactly four options under 'Options:' as a markdown list: "
                "- A) [option text]\n- B) [option text]\n- C) [option text]\n- D) [option text]. "
                "Ensure options are concise, distinct, and vertically aligned."
            )
        options = None
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )
        response = completion.choices[0].message.content
    except Exception as e:
        response = "## Question\nError generating question."
        return response, None, "No hint available"
    
    # Parse response
    question, hint = response.split("Hint:", 1) if "Hint:" in response else (response, "No hint provided")
    options = None
    if user_type == "non_dev":
        try:
            question_part, options_part = question.split("Options:", 1)
            options = re.findall(r'-\s*[A-D]\)\s*(.*?)(?=(?:-\s*[A-D]\)|$))', options_part, re.DOTALL)
            options = [opt.strip() for opt in options if opt.strip()]
            if len(options) != 4:
                raise ValueError("Expected 4 options")
            question = question_part.strip()
        except:
            options = ["Option A", "Option B", "Option C", "Option D"]
    return question, options, hint.strip()

# Endpoint to handle answer submission
@app.post("/submit-answer", response_class=HTMLResponse)
async def submit_answer(request: Request, session_id: str = Form(...), answer: str = Form(default=""), question_number: int = Form(...)):
    session = get_session(session_id)
    if not session:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Session not found"})
    
    user_type = session["user_type"]
    question_count = session["question_count"]
    score = session["score"]
    
    # Validate question number
    if question_number != question_count:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Invalid question number"})
    
    # Fetch question text
    question_text = get_question(session_id, question_number)
    if not question_text:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Question not found"})
    
    # Initialize options
    options = None
    full_answer = answer
    
    # For non-dev, map answer letter/index to full option text
    if user_type == "non_dev" and answer:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT options FROM answers WHERE session_id = ? AND question_number = ?", (session_id, question_number))
        result = c.fetchone()
        conn.close()
        if result and result["options"]:
            options = result["options"].split("|")
            try:
                # Answer is an index (0-3) from radio buttons
                index = int(answer)
                if 0 <= index < len(options):
                    full_answer = options[index]
            except ValueError:
                # Fallback: match answer to option text
                for opt in options:
                    if answer.upper() in opt:
                        full_answer = opt
                        break
    
    # Evaluate answer
    eval_prompt = (
        f"Evaluate the following answer for correctness: Question: {question_text}\nAnswer: {full_answer}\n"
        f"Return 'Correct', 'Incorrect', or 'Partial' followed by a brief feedback comment."
    )
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": eval_prompt}],
            temperature=1,
            max_completion_tokens=100,
            top_p=1,
            stream=False
        )
        eval_response = completion.choices[0].message.content
        is_correct = "Correct" in eval_response
        feedback = eval_response
    except Exception as e:
        is_correct = False
        feedback = f"Error evaluating answer: {str(e)}"
    
    # Update answer with feedback
    options_str = "|".join(options) if options else ""
    save_answer(session_id, question_number, question_text, full_answer, is_correct, options_str, feedback)
    
    if is_correct:
        score += 1
    
    # Update session
    update_session(session_id, question_count + 1, score)
    
    # Check for completion
    TOTAL_QUESTIONS = 10
    if question_count >= TOTAL_QUESTIONS:
        analytics = generate_analytics(session_id)
        badges = generate_badges(score)
        return templates.TemplateResponse(
            "conclusion.html",
            {"request": request, "score": score, "analytics": analytics, "badges": badges, "session_id": session_id}
        )
    
    # Generate next question
    difficulty = 1 + (score // 2)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT question_types FROM sessions WHERE id = ?", (session_id,))
    result = c.fetchone()
    conn.close()
    if result and result["question_types"]:
        question_types = result["question_types"].split(",")
    else:
        question_types = ["code_writing"] * 4 + ["bug_fixing"] * 3 + ["scenario"] * 3 if user_type == "dev" else ["mcq"] * 5 + ["scenario_mcq"] * 5
        random.shuffle(question_types)
    question_type = question_types[question_count % len(question_types)]
    question, options, hint = generate_question(user_type, question_type, load_transcript(), difficulty, full_answer)
    
    # Save next question and options
    options_str = "|".join(options) if options else ""
    save_answer(session_id, question_count + 1, question, "", False, options_str)
    
    return templates.TemplateResponse(
        "test.html",
        {
            "request": request, "session_id": session_id, "user_type": user_type,
            "question_html": markdown.markdown(question), 
            "options": options, "hint": hint,
            "question_number": question_count + 1, "total_questions": TOTAL_QUESTIONS,
            "time_limit": 300 if user_type == "dev" else 120
        }
    )

# Endpoint for live hints
@app.get("/hint/{session_id}/{question_number}")
async def get_hint(session_id: str, question_number: int):
    question_text = get_question(session_id, question_number)
    if not question_text:
        return "Question not found"
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": f"Provide a subtle hint for this question: {question_text}"}],
            temperature=1,
            max_completion_tokens=100,
            top_p=1,
            stream=False
        )
        return completion.choices[0].message.content
    except:
        return "Error fetching hint"

# Updated analytics to include feedback
def generate_analytics(session_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT question_text, answer, is_correct, feedback FROM answers WHERE session_id = ?", (session_id,))
    answers = c.fetchall()
    conn.close()
    
    total = len(answers)
    correct = sum(1 for row in answers if row["is_correct"])
    by_type = {"code_writing": 0, "bug_fixing": 0, "scenario": 0, "mcq": 0, "scenario_mcq": 0}
    analytics = f"Total Correct: {correct}/{total}\n\n"
    for i, row in enumerate(answers, 1):
        question = row["question_text"].lower()
        analytics += f"Question {i}: {'Correct' if row['is_correct'] else 'Incorrect'}\nFeedback: {row['feedback']}\n\n"
        for q_type in by_type:
            if q_type in question:
                by_type[q_type] += 1 if row["is_correct"] else 0
    analytics += f"Performance by Type: {by_type}"
    return analytics

# Updated badges
def generate_badges(score):
    badges = []
    if score >= 5:
        badges.append("5 Correct Answers")
    if score >= 8:
        badges.append("High Scorer")
    return badges

# Endpoint to export results as PDF
@app.get("/export-results", response_class=FileResponse)
async def export_results(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    score = session["score"]
    analytics = generate_analytics(session_id)
    badges = generate_badges(score)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Test calls Results", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Score: {score}/10", ln=True)
    pdf.multi_cell(0, 10, txt=f"Analytics:\n{analytics}")
    pdf.multi_cell(0, 10, txt=f"Badges: {', '.join(badges) if badges else 'None'}")
    
    pdf_file = f"results_{session_id}.pdf"
    pdf.output(pdf_file)
    return FileResponse(pdf_file, media_type="application/pdf", filename=pdf_file)