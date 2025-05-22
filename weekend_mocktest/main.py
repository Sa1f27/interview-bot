# Final FastAPI app with MongoDB summary loading and no SQLite
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import markdown
from fpdf import FPDF
import uuid
import random
import re
import pymongo
from groq import Groq
import os

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
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Optional: Mount static files (uncomment if needed)
# app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Groq Client
client = Groq()


# Jinja2 A-D option filter
def to_letter(index: int) -> str:
    return chr(65 + index)
templates.env.filters['to_letter'] = to_letter

# MongoDB Summary Fetch
mongo_client = pymongo.MongoClient("mongodb://sa:L%40nc%5Eere%400012@192.168.48.200:27017/?authSource=admin")
db = mongo_client["test"]
transcripts_collection = db["drive"]

def load_transcript():
    try:
        doc = transcripts_collection.find_one({}, sort=[("_id", -1)], projection={"_id": 0, "summary": 1})
        return doc["summary"] if doc and "summary" in doc else "Summary not found."
    except Exception as e:
        return f"Error loading summary: {str(e)}"

# In-memory storage
SESSIONS = {}
ANSWERS = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("start_test.html", {"request": request})

@app.post("/start-test", response_class=HTMLResponse)
async def start_test(request: Request, user_type: str = Form(...)):
    if user_type not in ["dev", "non_dev"]:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Invalid user type"})

    session_id = str(uuid.uuid4())
    transcript_content = load_transcript()
    question_types = (
        ["code_writing"] * 4 + ["bug_fixing"] * 3 + ["scenario"] * 3
        if user_type == "dev"
        else ["mcq"] * 5 + ["scenario_mcq"] * 5
    )
    random.shuffle(question_types)
    SESSIONS[session_id] = {"user_type": user_type, "score": 0, "question_count": 1, "question_types": question_types}
    ANSWERS[session_id] = []

    q_type = question_types[0]
    question, options = generate_question(user_type, q_type, transcript_content, difficulty=1, prev_answer=None)
    ANSWERS[session_id].append({"question": question, "answer": "", "correct": False, "options": options or [], "feedback": ""})

    return templates.TemplateResponse(
        "test.html",
        {
            "request": request, "session_id": session_id, "user_type": user_type,
            "question_html": markdown.markdown(question), "options": options,
            "question_number": 1, "total_questions": 10, "time_limit": 300 if user_type == "dev" else 120
        }
    )

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
        prompt += "\nGenerate a multiple-choice question with 4 markdown list options (A-D)."    

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
    except Exception:
        return "## Question\nError generating question.", None

    question = response.strip()
    options = None
    if user_type == "non_dev":
        try:
            q_part, opt_part = question.split("Options:", 1)
            options = re.findall(r'-\s*[A-D]\)\s*(.*?)(?=(?:-\s*[A-D]\)|$))', opt_part, re.DOTALL)
            options = [opt.strip() for opt in options if opt.strip()]
            if len(options) != 4:
                raise ValueError()
            question = q_part.strip()
        except:
            options = ["Option A", "Option B", "Option C", "Option D"]
    return question.strip(), options

@app.post("/submit-answer", response_class=HTMLResponse)
async def submit_answer(request: Request, session_id: str = Form(...), answer: str = Form(default=""), question_number: int = Form(...)):
    session = SESSIONS.get(session_id)
    if not session:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Session not found"})
    
    user_type = session["user_type"]
    score = session["score"]
    count = session["question_count"]
    if question_number != count:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Invalid question number"})

    ans_data = ANSWERS[session_id][-1]
    options = ans_data["options"]
    full_answer = answer
    if user_type == "non_dev" and answer.isdigit() and int(answer) < len(options):
        full_answer = options[int(answer)]

    eval_prompt = f"Evaluate answer: {ans_data['question']}\nAnswer: {full_answer}\nReturn 'Correct', 'Incorrect', or 'Partial' with a comment."
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

    if count >= 10:
        return templates.TemplateResponse(
            "conclusion.html",
            {"request": request, "score": session["score"], "analytics": generate_analytics(session_id),
             "badges": generate_badges(session["score"]), "session_id": session_id}
        )

    session["question_count"] += 1
    next_type = session["question_types"][count % len(session["question_types"])]
    transcript = load_transcript()
    q, opts = generate_question(user_type, next_type, transcript, session["score"] // 2 + 1, full_answer)
    ANSWERS[session_id].append({"question": q, "answer": "", "correct": False, "options": opts or [], "feedback": ""})

    return templates.TemplateResponse(
        "test.html",
        {
            "request": request, "session_id": session_id, "user_type": user_type,
            "question_html": markdown.markdown(q), "options": opts,
            "question_number": session["question_count"], "total_questions": 10,
            "time_limit": 300 if user_type == "dev" else 120
        }
    )

def generate_analytics(session_id):
    answers = ANSWERS[session_id]
    total = len(answers)
    correct = sum(1 for a in answers if a["correct"])
    breakdown = "\n\n".join(
        f"Q{i+1}: {'✅' if a['correct'] else '❌'}\nFeedback: {a['feedback']}"
        for i, a in enumerate(answers)
    )
    return f"Correct: {correct}/{total}\n\n{breakdown}"

def generate_badges(score):
    return [b for b, s in [("5 Correct Answers", 5), ("High Scorer", 8)] if score >= s]

@app.get("/export-results", response_class=FileResponse)
async def export_results(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Test Results", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Score: {session['score']}/10", ln=True)
    pdf.multi_cell(0, 10, txt=generate_analytics(session_id))
    badges = generate_badges(session['score'])
    pdf.multi_cell(0, 10, txt=f"Badges: {', '.join(badges) if badges else 'None'}")
    pdf_path = f"results_{session_id}.pdf"
    pdf.output(pdf_path)
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7004, reload=True)
