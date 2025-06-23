from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import asyncio
import uuid
import logging
import random
import textwrap 
import subprocess
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import pyodbc
import random
import edge_tts
import pymongo
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from contextlib import asynccontextmanager
import io
from reportlab.pdfgen import canvas
from urllib.parse import quote_plus
from reportlab.lib.pagesizes import LETTER
from fastapi.responses import StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
INACTIVITY_TIMEOUT = 300
TTS_SPEED = 1.0

# ========================
# Models and schemas
# ========================

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

def fetch_random_student_info():
    """Fetch a random ID, name from tbl_Student and session_id from session table from SQL Server"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        # Fetch all student records (ID, First_Name, Last_Name)
        cursor.execute("SELECT ID, First_Name, Last_Name FROM tbl_Student WHERE ID IS NOT NULL AND First_Name IS NOT NULL AND Last_Name IS NOT NULL")
        
        student_records = cursor.fetchall()
        
        if not student_records:
            logger.warning("No valid student data found in the database")
            return None

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
        return None

class Session:
    """Session data model for a test session"""
    def __init__(self, summary: str, voice: str):
        self.summary = summary
        self.voice = voice        
        self.conversation_log = []
        self.last_activity = time.time()
        self.question_index = 0
        self.current_concept = None

class ConversationEntry:
    """Single Q&A entry in the conversation log"""
    def __init__(self, question: str, answer: str = None, concept: str = None):
        self.question = question
        self.answer = answer
        self.concept = concept
        self.timestamp = time.time()

class RecordRequest(BaseModel):
    """Request model for recording and responding endpoint"""
    test_id: str

class TestResponse(BaseModel):
    """Response model for test creation"""
    test_id: str
    question: str
    audio_path: str

class ConversationResponse(BaseModel):
    """Response model for conversation updates"""
    ended: bool
    response: str
    audio_path: str

class SummaryResponse(BaseModel):
    """Response model for test summary"""
    summary: str
    analytics: Dict[str, Any]
    pdf_url: Optional[str] 

# ========================
# Application setup
# ========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("Daily standup system starting up...")
    yield
    # Shutdown
    db_manager.close()
    AudioManager.clean_audio_folder()
    logger.info("Daily standup system shut down")

app = FastAPI(title="Voice-Based Testing System", lifespan=lifespan)

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# Mount static directories
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="static")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# ========================
# Database setup
# ========================

class DatabaseManager:
    """MongoDB database manager"""
    def __init__(self, connection_string, db_name):
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[db_name]
        self.transcripts = self.db["original-1"]
        self.conversations = self.db["daily_standup_results-1"]
    
    def get_latest_summary(self) -> str:
        """Fetch the latest lecture summary - simple and safe"""
        try:
            # Simple query: get any document with summary, sorted by timestamp desc
            doc = self.transcripts.find_one(
                {"summary": {"$exists": True, "$ne": None, "$ne": ""}},
                sort=[("timestamp", -1)]
            )
            
            if doc and doc.get("summary"):
                return doc["summary"]
            
            # If timestamp sorting fails, try without sorting
            doc = self.transcripts.find_one({"summary": {"$exists": True, "$ne": None, "$ne": ""}})
            
            if doc and doc.get("summary"):
                return doc["summary"]
                
            raise ValueError("No summary found in the database")
            
        except Exception as e:
            logger.error(f"Error fetching summary: {e}")
            raise ValueError("No summary found in the database")
    
    def save_test_data(self, test_id: str, conversation_log: List[ConversationEntry], evaluation: str) -> bool:
        """Save test data to the conversation collection"""
        try:
            # Fetch random student ID from SQL Server
            student_id, first_name, last_name, session_id = fetch_random_student_info()
            name = first_name + " " + last_name if first_name and last_name else "Unknown Student"
            
            # Extract score from evaluation text
            import re
            score_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', evaluation)
            extracted_score = float(score_match.group(1)) if score_match else None
            
            # Convert conversation log to the required format
            conversation_data = []
            for entry in conversation_log:
                conversation_data.append({
                    "question": entry.question,
                    "answer": entry.answer,
                    "concept": entry.concept,
                    "timestamp": entry.timestamp
                })
            
            # Create the document to insert
            document = {
                "test_id": test_id,
                "Student_ID": student_id,
                "name": name,
                "session_id": session_id,
                "timestamp": time.time(),
                "conversation_log": conversation_data,
                "evaluation": evaluation,
                "score": extracted_score 
            }
            
            # Insert into the conversation collection
            result = self.conversations.insert_one(document)
            logger.info(f"Test data saved successfully for test {test_id}, name: {name}, Student_ID: {student_id}, score: {extracted_score}, session_id: {session_id}, document ID: {result.inserted_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving test data for test {test_id}: {e}")
            return False
    
    def close(self):
        """Close the database connection"""
        self.client.close()

# MongoDB credentials
MONGO_USER = "LanTech"
MONGO_PASS = "L@nc^ere@0012"
MONGO_HOST = "192.168.48.201:27017"
MONGO_DB_NAME = "Api-1"
MONGO_AUTH_SOURCE = "admin"

db_manager = DatabaseManager(
    f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASS)}@{MONGO_HOST}/{MONGO_DB_NAME}?authSource={MONGO_AUTH_SOURCE}",
    MONGO_DB_NAME
)

# ========================
# Test management
# ========================

class TestManager:
    """Manages test sessions"""
    def __init__(self):
        self.tests: Dict[str, Session] = {}
    
    def create_test(self, summary: str, voice: str) -> str:
        """Create a new test session"""
        test_id = str(uuid.uuid4())
        self.tests[test_id] = Session(summary, voice)
        return test_id
    
    def get_truncated_conversation_history(self, test_id: str, window_size: int = 5) -> str:
        """Return a string of the last `window_size` Q&A pairs formatted"""
        test = self.validate_test(test_id)
        last_entries = test.conversation_log[-window_size:]
        history = []
        for entry in last_entries:
            q_line = f"Q: {entry.question}"
            a_line = f"A: {entry.answer}" if entry.answer else "A: "
            history.append(f"{q_line}\n{a_line}")
        return "\n\n".join(history)
    
    def get_test(self, test_id: str) -> Optional[Session]:
        """Get a test by ID"""
        return self.tests.get(test_id)
    
    def validate_test(self, test_id: str) -> Session:
        """Validate test ID and update last activity"""
        test = self.get_test(test_id)
        if not test:
            raise HTTPException(status_code=400, detail="Test not found")
        
        if time.time() > test.last_activity + INACTIVITY_TIMEOUT:
            raise HTTPException(status_code=400, detail="Test timed out")
        
        test.last_activity = time.time()
        return test
    
    def add_question(self, test_id: str, question: str, concept: str = None):
        """Add a question to the test conversation log"""
        test = self.validate_test(test_id)
        test.conversation_log.append(ConversationEntry(question=question, concept=concept))
        test.current_concept = concept
        test.question_index += 1
    
    def add_answer(self, test_id: str, answer: str):
        """Add an answer to the last question in the conversation log"""
        test = self.validate_test(test_id)
        if test.conversation_log:
            test.conversation_log[-1].answer = answer
    
# Initialize test manager
test_manager = TestManager()

# ========================
# LLM and prompt setup
# ========================

class LLMManager:
    """Manages LLM interactions for question generation and evaluation"""
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.8)
        self.parser = StrOutputParser()
        
        # Define prompts
        self.question_prompt = PromptTemplate.from_template("""
        You are conducting a comprehensive voice-based test designed to last for at least 15 minutes.
        Your goal is to assess the student's deep understanding of the topic.

        Start by greeting the user and using the provided Lecture Summary to form questions. As the test progresses,
        you should broaden the scope and ask follow-up questions and related questions based on your general knowledge
        of the subject and. This will help evaluate their understanding beyond the summary.

        Lecture Summary:
        {summary}

        Conversation so far:
        {history}

        Instructions:
        1. Extract a key concept from the lecture that hasn't been covered yet.
        2. Formulate a clear, concise question about that concept.
        3. If most concepts from the summary are covered, ask a more general or advanced question on the topic.

        First identify the concept, then ask one question about it.
        Format your response as:
        CONCEPT: [the concept]
        QUESTION: [your question]
        """)
        
        self.followup_prompt = PromptTemplate.from_template("""
        You are an engaging and friendly interviewer. Your goal is to have a natural, flowing conversation with a student to understand their knowledge, not just to conduct a rigid test. Make the student feel comfortable. Use conversational language, show empathy, and react naturally to their answers.

        **Your Persona:**
        - **Curious and Encouraging:** Act genuinely interested in what they have to say. Use phrases like "That's interesting, can you tell me more about...", "Oh, that's a great point.", or "I see, so how does that connect to...".
        - **Conversational, not Robotic:** Avoid stiff, formal language. Your questions should feel like a natural part of a conversation.
        - **Empathetic:** If the user is struggling, be encouraging. You might say, "No worries, that's a tricky concept. Let's try looking at it from another angle." or "That's a common point of confusion."
        - **Avoids Repetition:** Do not ask a question that is identical to one you've already asked in the 'Recent Conversation' history. You can re-approach a concept, but you must phrase the question differently.

        **Context of the Conversation:**
        - **Topic:** Based on the Lecture Summary.
        - **Lecture Summary:** {summary}
        - **Current Concept:** {concept}
        - **Recent Conversation:**
        {history}
        - **Your Last Question:** {previous_question}
        - **Their Response:** {user_response}
        - **Progress:** This is question {current_question_number} out of {total_questions}.

        **Your Task:**
        Based on their response, decide how to continue the conversation.

        1.  **If they seem to understand:**
            - Acknowledge their answer positively ("Great explanation!", "That makes sense.").
            - Smoothly transition to a new, related concept from the summary or your own knowledge.
            - Ask your next question conversationally.

        2.  **If they seem to be struggling or are incorrect:**
            - Be gentle and supportive. Don't say "You're wrong."
            - Try to rephrase the question with simple english words or ask a simpler one about the **same concept** to help them. For example: "Okay, let's break that down a bit. What's the first step in that process?"

        3.  **If their response is off-topic:**
            - Gently steer them back. For example: "That's an interesting thought. Let's bring it back to [the topic] for a moment."

        **Output Format (Strictly follow this):**
        UNDERSTANDING: [YES or NO]
        CONCEPT: [The concept for your next question. Can be the same or new.]
        QUESTION: [Your natural, conversational follow-up question. Do NOT include any preamble like "My question is...". Just the question itself.]

        """)
        
        self.evaluation_prompt = PromptTemplate.from_template("""
        You are evaluating a student's performance in a comprehensive voice-based test on the topic below.
        The test assessed understanding of both the provided summary and broader related knowledge.

        Lecture Summary:
        {summary}

        Full Q&A log:
        {conversation}

        Generate a concise, strict evaluation. Your response must include the following sections:
        1. Key Strengths: (2-3 bullet points)
        2. Areas for Improvement: (1-2 bullet points)
        3. Concept Coverage: (Brief summary of how well they covered concepts from the summary and beyond)
        4. Recommendation: (One specific recommendation for further study)
        5. Final Score: A numeric score on a separate line, in the format 'Final Score: X/10'.

        Be very strict with the score. If the user gave multiple irrelevant or off-topic answers, give a score of 0/10.
        Keep the total evaluation under 200 words.
        """)
    
    def _parse_llm_response(self, response: str, keys: List[str]) -> Dict[str, str]:
        """Parse structured responses from the LLM"""
        result = {}
        lines = response.strip().split('\n')
        for line in lines:
            for key in keys:
                prefix = f"{key}:"
                if line.startswith(prefix):
                    result[key.lower()] = line[len(prefix):].strip()
                    break
        return result
    
    async def generate_first_question(self, summary: str) -> Dict[str, str]:
        """Generate the first question for a test session"""
        chain = self.question_prompt | self.llm | self.parser
        response = await chain.ainvoke({"summary": summary, "history": ""})
        return self._parse_llm_response(response, ["CONCEPT", "QUESTION"])
    
    async def generate_followup(self, 
                              summary: str, 
                              history: str, 
                              previous_question: str, 
                              user_response: str,
                              concept: str,
                              current_question_number: int, total_questions: int) -> Dict[str, str]:
        """Generate a follow-up question based on the user's response"""
        chain = self.followup_prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "summary": summary,
            "history": history,
            "previous_question": previous_question,
            "user_response": user_response,
            "concept": concept,
            "current_question_number": current_question_number,
            "total_questions": total_questions
        })
        return self._parse_llm_response(
            response, 
            ["UNDERSTANDING", "CONCEPT", "QUESTION"]
        )
    
    async def generate_evaluation(self, summary: str, conversation: str) -> str:
        """Generate an evaluation of the test session"""
        chain = self.evaluation_prompt | self.llm | self.parser
        return await chain.ainvoke({
            "summary": summary,
            "conversation": conversation
        })

# Initialize LLM manager
llm_manager = LLMManager()

# ========================
# Audio utilities (Simplified)
# ========================

class AudioManager:
    """Manages audio transcription and text-to-speech (no longer handles recording)"""
    
    @staticmethod
    def clean_audio_folder():
        """Clean up old audio files"""
        for filename in os.listdir(AUDIO_DIR):
            if filename.endswith(".mp3") and os.path.getmtime(os.path.join(AUDIO_DIR, filename)) < time.time() - 3600:
                try:
                    os.remove(os.path.join(AUDIO_DIR, filename))
                except Exception as e:
                    logger.warning(f"Failed to delete {filename}: {e}")
    
    @staticmethod
    async def text_to_speech(text: str, voice: str, speed: float = TTS_SPEED) -> Optional[str]:
        """Convert text to speech using Edge TTS"""
        timestamp = int(time.time() * 1000)
        raw_path = os.path.join(AUDIO_DIR, f"ai_raw_{timestamp}.mp3")
        final_path = os.path.join(AUDIO_DIR, f"ai_{timestamp}.mp3")
        
        try:
            AudioManager.clean_audio_folder()
            await edge_tts.Communicate(text, voice).save(raw_path)
            
            # Speed up the audio
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-filter:a", f"atempo={speed}", "-vn", final_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            os.remove(raw_path)
            if os.path.exists(final_path):
                return f"./audio/{os.path.basename(final_path)}"
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
        
        return None
    
    @staticmethod
    def transcribe(filepath: str) -> str:
        """Transcribe audio using Groq's Whisper API"""
        logger.info(f"Transcribing audio: {filepath}")
        try:
            # Using ChatGroq for transcription (using Whisper)
            from groq import Groq
            groq_client = Groq()
            
            with open(filepath, "rb") as file:
                result = groq_client.audio.transcriptions.create(
                    file=(filepath, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json"
                )
            return result.text.strip()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise

# ========================
# Helper functions
# ========================

def get_random_voice() -> str:
    """Get a random TTS voice"""
    voices = ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural"]
    return random.choice(voices)

# ========================
# API Endpoints
# ========================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main application page"""
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

@app.get("/start_test", response_model=TestResponse)
async def start_test():
    """Start a new test session"""
    try:
        # Get latest lecture summary
        summary = db_manager.get_latest_summary()
        
        # Create a new test
        voice = get_random_voice()
        test_id = test_manager.create_test(summary, voice)
        
        # Generate first question
        question_data = await llm_manager.generate_first_question(summary)
        question = question_data.get("question", "What can you tell me about this topic?")
        concept = question_data.get("concept", "general understanding")
        
        # Add question to test
        test_manager.add_question(test_id, question, concept)
        
        # Generate audio for the question
        audio_path = await AudioManager.text_to_speech(question, voice)
        
        return {
            "test_id": test_id, 
            "question": question, 
            "audio_path": audio_path,
        }
    
    except Exception as e:
        logger.error(f"Error starting test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse

@app.post("/record_and_respond", response_model=ConversationResponse)
async def record_and_respond(
    audio: UploadFile = File(...),
    test_id: str = Form(...)
):
    total_questions = 5 # Configurable total questions
    """Process user's uploaded audio response and provide the next question"""
    try:
        test = test_manager.validate_test(test_id)

        # Save uploaded audio to file system
        audio_filename = os.path.join(TEMP_DIR, f"user_audio_{int(time.time())}.webm")
        with open(audio_filename, "wb") as f:
            f.write(await audio.read())

        # Transcribe the uploaded audio
        user_response = AudioManager.transcribe(audio_filename)
        logger.info(f"Transcribed user response: {user_response}")

        # Clean up the temporary audio file
        try:
            os.remove(audio_filename)
        except Exception as e:
            logger.warning(f"Failed to clean up audio file: {e}")

        # Log the user's answer
        test_manager.add_answer(test_id, user_response)

        # Check if the test is now complete (i.e., user has answered the last question)
        if test.question_index >= total_questions:
            closing_message = "The test has ended. Thank you for your participation."
            audio_path = await AudioManager.text_to_speech(closing_message, test.voice)
            return {"ended": True, "response": closing_message, "audio_path": audio_path}

        # Generate follow-up question using LLM
        history = test_manager.get_truncated_conversation_history(test_id)
        last_question = test.conversation_log[-1].question
        last_concept = test.current_concept or test.conversation_log[-1].concept
        followup_data = await llm_manager.generate_followup(
            test.summary,
            history,
            last_question,
            user_response,
            last_concept,
            # Pass the number for the *next* question to be generated
            test.question_index + 1, 
            total_questions)

        # Extract next question and other info
        next_question = followup_data.get("question", "Can you elaborate more on that?")
        next_concept = followup_data.get("concept", last_concept)

        # Update test log and synthesize speech
        test_manager.add_question(test_id, next_question, next_concept)
        audio_path = await AudioManager.text_to_speech(next_question, test.voice)

        return {
            "ended": False,
            "response": next_question,
            "audio_path": audio_path,
        }

    except Exception as e:
        logger.error(f"Error processing response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary", response_model=SummaryResponse)
async def get_summary(test_id: str):
    """Get a summary evaluation of the test session"""
    try:
        test = test_manager.validate_test(test_id)
        history = test_manager.get_truncated_conversation_history(test_id)

        # Generate evaluation
        evaluation = await llm_manager.generate_evaluation(
            test.summary,
            history
        )
        
        # Save test data to MongoDB
        save_success = db_manager.save_test_data(
            test_id=test_id,
            conversation_log=test.conversation_log,
            evaluation=evaluation
        )
        
        if not save_success:
            logger.warning(f"Failed to save test data for test {test_id}")
        
        # Calculate analytics
        num_questions = len(test.conversation_log)
        answers = [entry.answer for entry in test.conversation_log if entry.answer]
        avg_length = sum(len(answer.split()) for answer in answers) / len(answers) if answers else 0
        
        # Calculate concept coverage
        unique_concepts = set(entry.concept for entry in test.conversation_log if entry.concept)
        concept_coverage = len(unique_concepts)
        
        return {
            "summary": evaluation,
            "analytics": {
                "num_questions": num_questions,
                "avg_response_length": round(avg_length, 1),
                "concept_coverage": concept_coverage
            },
            "pdf_url": f"./download_results/{test_id}"
        }

    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cleanup endpoint
@app.get("/cleanup")
async def cleanup_resources():
    """Clean up audio files and expired tests"""
    try:
        # Clean up audio files
        AudioManager.clean_audio_folder()
        
        # Clean up expired tests
        current_time = time.time()
        expired_tests = [
            tid for tid, test in test_manager.tests.items()
            if current_time > test.last_activity + INACTIVITY_TIMEOUT * 2
        ]
        
        for tid in expired_tests:
            test_manager.tests.pop(tid, None)
        
        return {"message": f"Cleaned up {len(expired_tests)} expired tests and old audio files"}
    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
@app.get("/api/download_results/{test_id}")
async def download_results_pdf(test_id: str):
    """Fetch the saved test document from MongoDB and stream it as a PDF."""
    try:
        # Query MongoDB
        doc = db_manager.conversations.find_one({"test_id": test_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Test ID not found")

        # Create PDF in memory
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=LETTER)
        width, height = LETTER
        margin = 50
        y = height - margin

        def write_line(canvas_obj, current_y, label: str, value: str, indent: int = 0):
            if current_y < margin + 50:
                canvas_obj.showPage()
                canvas_obj.setFont("Helvetica", 12)
                return height - margin
            canvas_obj.drawString(margin + indent, current_y, f"{label}: {value}")
            return current_y - 20

        # Header
        p.setFont("Helvetica-Bold", 14)
        p.drawString(margin, y, f"Daily Standup Results – Test ID: {test_id}")
        y -= 30

        # Write document fields
        p.setFont("Helvetica", 12)
        
        name_val = doc.get("name", "N/A")
        y = write_line(p, y, "Name", str(name_val))

        student_val = doc.get("Student_ID", "N/A")
        y = write_line(p, y, "Student_ID", str(student_val))

        session_val = doc.get("session_id", "N/A")
        y = write_line(p, y, "Session_ID", str(session_val))

        try:
            ts = float(doc.get("timestamp", time.time()))
            timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        except:
            timestr = "N/A"
        y = write_line(p, y, "Saved At", timestr)

        score_val = doc.get("score", "N/A")
        y = write_line(p, y, "Score", str(score_val))

        eval_val = doc.get("evaluation", "N/A")
        wrapped_eval = textwrap.wrap(str(eval_val), 80)
        p.setFont("Helvetica-Bold", 12)
        if y < margin + 50:
            p.showPage()
            y = height - margin
            p.setFont("Helvetica-Bold", 12)
        p.drawString(margin, y, "Evaluation:")
        y -= 20
        p.setFont("Helvetica", 12)
        for line in wrapped_eval:
            y = write_line(p, y, "", line)

        y -= 10

        # Conversation Log
        p.setFont("Helvetica-Bold", 12)
        if y < margin + 30:
            p.showPage()
            y = height - margin
            p.setFont("Helvetica-Bold", 12)
        p.drawString(margin, y, "Conversation Log:")
        y -= 20

        p.setFont("Helvetica", 11)
        for idx, entry in enumerate(doc.get("conversation_log", []), start=1):
            if y < margin + 80:
                p.showPage()
                p.setFont("Helvetica", 11)
                y = height - margin

            concept_val = entry.get("concept", "N/A")
            y = write_line(p, y, f"{idx}. Concept", concept_val)

            question_val = entry.get("question", "N/A")
            y = write_line(p, y, "    Question", question_val, indent=15)

            answer_val = entry.get("answer", "N/A")
            y = write_line(p, y, "    Answer", answer_val, indent=15)

            try:
                ets = float(entry.get("timestamp", time.time()))
                etimestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ets))
            except:
                etimestr = "N/A"
            y = write_line(p, y, "    Asked/Answered At", etimestr, indent=15)

            y -= 10

        p.showPage()
        p.save()
        buffer.seek(0)

        filename = f"standup_results_{test_id}.pdf"
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error while generating PDF for {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while generating PDF")

@app.get("/api/tests", response_class=JSONResponse)
async def get_all_tests():
    """Get all test documents from the MongoDB collection"""
    try:
        results = list(db_manager.conversations.find(
            {}, 
            {"_id": 0, "conversation_log": 0, "evaluation": 0, "timestamp": 0}
        ))
        return {"tests": results}
    except Exception as e:
        logger.error(f"Error fetching all tests: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tests")

@app.get("/api/tests/{test_id}", response_class=JSONResponse)
async def get_test_by_id(test_id: str):
    """Get a specific test document by test_id"""
    try:
        result = db_manager.conversations.find_one(
            {"test_id": test_id}, 
            {"_id": 0, "conversation_log": 0, "evaluation": 0, "timestamp": 0}
        )
        if not result:
            raise HTTPException(status_code=404, detail="Test not found")
        return result
    except Exception as e:
        logger.error(f"Error fetching test {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve test")
    
    
@app.get("/api/standup-students", response_class=JSONResponse)
async def get_unique_standup_students():
    """
    Return distinct Student_ID and name from daily standup results.
    """
    try:
        pipeline = [
            {"$group": {"_id": "$Student_ID", "name": {"$first": "$name"}}},
            {"$project": {"_id": 0, "Student_ID": "$_id", "name": 1}}
        ]
        students = list(db_manager.conversations.aggregate(pipeline))
        return {"count": len(students), "students": students}
    except Exception as e:
        logger.error(f"Error fetching standup students: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve students")

@app.get("/api/standup-students/{student_id}/tests", response_class=JSONResponse)
async def get_tests_for_standup_student(student_id: str):
    """
    Get all test documents for a specific student, excluding conversation and evaluation.
    """
    try:
        student_id_int = int(student_id)
        results = list(db_manager.conversations.find(
            {"Student_ID": student_id_int},
            {"_id": 0, "conversation_log": 0, "evaluation": 0, "timestamp": 0}
        ))
        if not results:
            raise HTTPException(status_code=404, detail="No tests found for this student")
        return {"count": len(results), "tests": results}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid student ID format")
    except Exception as e:
        logger.error(f"Error fetching tests for student ID {student_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tests for student")