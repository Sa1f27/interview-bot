from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
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
TOTAL_QUESTIONS = 20  # Total number of questions in a test
ESTIMATED_SECONDS_PER_QUESTION = 180  # 3 minutes, for UI timer estimation

# Environment configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")  # Configure via environment variable

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
        logger.info("Successfully connected to SQL Server.")
        return conn
    except pyodbc.Error as e:
        # Log the error and re-raise to ensure it's handled upstream
        logger.error(f"SQL Server database connection error: {e}", exc_info=True)
        raise

def fetch_random_student_info():
    """Fetch a random ID, name from tbl_Student and session_id from session table from SQL Server"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        logger.info("Attempting to fetch random student info from SQL Server.")
        
        cursor = conn.cursor()
        # Fetch all student records (ID, First_Name, Last_Name)
        cursor.execute("SELECT ID, First_Name, Last_Name FROM tbl_Student WHERE ID IS NOT NULL AND First_Name IS NOT NULL AND Last_Name IS NOT NULL")
        
        student_records = cursor.fetchall()
        
        if not student_records:
            logger.warning("No valid student data found in tbl_Student. Cannot assign student to test.")
            return None

        # Fetch distinct Session_ID
        cursor.execute("SELECT DISTINCT Session_ID FROM tbl_Session WHERE Session_ID IS NOT NULL")
        session_rows = cursor.fetchall()
        session_ids = [row[0] for row in session_rows]
        if not session_ids:
            logger.warning("No valid session IDs found in tbl_Session. Session ID will be None.")

        cursor.close()
        conn.close()

        logger.info(f"Found {len(student_records)} student records and {len(session_ids)} session IDs.")
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
        ) # Return None for session_id if no session_ids are found
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
    duration_sec: int

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
    logger.info("Daily standup API server starting up...")
    yield
    # Shutdown
    db_manager.close()
    AudioManager.clean_audio_folder()
    logger.info("Daily standup API server shut down")

app = FastAPI(
    title="Daily Standup Voice Testing API", 
    description="API for voice-based daily standup testing system",
    version="1.0.0",
    lifespan=lifespan
)

# Get base directory and setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# ========================
# CORS Configuration
# ========================
def setup_cors(app: FastAPI, frontend_origin: str):
    """Setup CORS middleware with configurable origins"""
    if frontend_origin == "*":
        # Development mode - allow all origins
        logger.warning("CORS configured for ALL origins - NOT RECOMMENDED FOR PRODUCTION")
        allowed_origins = ["*"]
    else:
        # Production mode - specific origins
        allowed_origins = [origin.strip() for origin in frontend_origin.split(",")]
        logger.info(f"CORS configured for origins: {allowed_origins}")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

setup_cors(app, FRONTEND_ORIGIN)

# ========================
# Static Audio File Serving
# ========================
# Mount audio directory to serve generated audio files over HTTP/HTTPS
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
logger.info(f"Audio files served from: /audio (directory: {AUDIO_DIR})")

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
        logger.info(f"Attempting to save test data for test_id: {test_id}")
        try:
            # Fetch random student ID from SQL Server
            student_info = fetch_random_student_info()
            if not student_info:
                logger.error("Failed to fetch student info from SQL Server")
                return False
                
            logger.info(f"Fetched student info: {student_info}")
            student_id, first_name, last_name, session_id = student_info
            name = f"{first_name} {last_name}" if first_name and last_name else "Unknown Student"
            
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
            logger.error(f"Error saving test data for test {test_id}: {e}", exc_info=True) # Log traceback
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
    
    def cleanup_expired_tests(self):
        """Remove expired test sessions"""
        current_time = time.time()
        expired_tests = [
            tid for tid, test in self.tests.items()
            if current_time > test.last_activity + INACTIVITY_TIMEOUT * 2
        ]
        
        for tid in expired_tests:
            self.tests.pop(tid, None)
        
        return len(expired_tests)

# Initialize test manager
test_manager = TestManager()

# ========================
# LLM and prompt setup
# ========================

# ========================
# LLM and prompt setup
# ========================

class LLMManager:
    """Manages LLM interactions for question generation and evaluation"""
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.8)
        self.parser = StrOutputParser()
        
        # Define prompts
        self.question_prompt = PromptTemplate.from_template("""
        You are a friendly and supportive interviewer conducting a voice-based daily standup. Your goal is to create a comfortable, welcoming environment.

        **Your Task:**
        Start with a warm, simple greeting that puts the person at ease. Use everyday language and be genuinely friendly.
        Examples: "Hello there! I hope you're having a good day." or "Hi! Thanks for joining me today."
        Do NOT ask direct questions yet - this is just to make them feel welcome and relaxed.

        **Important Guidelines:**
        - Use simple, clear English
        - Be warm and encouraging
        - Keep it conversational and natural
        - Make them feel comfortable and supported

        **Context (for your reference only):**
        Lecture Summary: {summary}
        Conversation so far: {history}

        **Output Format (Strictly follow this):**
        CONCEPT: greeting
        QUESTION: [Your warm, friendly greeting without direct questions]
        """)
        
        self.followup_prompt = PromptTemplate.from_template("""
        You are a kind and supportive interviewer conducting a voice-based daily standup. Your approach should be encouraging and understanding.

        **Context:**
        - **Lecture Summary:** {summary}
        - **Current Concept:** {concept}
        - **Recent Conversation History:**
        {history}
        - **Your Last Question:** {previous_question}
        - **Student's Response:** {user_response}
        - **Progress:** This is question {current_question_number} of {total_questions}.

        **Your Task:**

        **IF THE 'Current Concept' IS 'greeting':**
        The user has responded to your greeting. Now gently transition to the standup.
        1. Acknowledge their response warmly (e.g., "That's wonderful to hear!")
        2. Gently introduce the standup (e.g., "Let's start with something easy...")
        3. Ask your first question using simple, clear language based on the **Lecture Summary**
        4. Set UNDERSTANDING to YES

        **OTHERWISE (for all other concepts):**
        The user has answered a standup question. Be supportive and encouraging.
        1. **If they seem to understand:** Give brief, positive feedback ("Great job explaining that!") and ask about a related topic
        2. **If they seem to struggle or are uncertain:** Be very supportive:
           - Reassure them ("That's okay, no pressure at all!")
           - Offer help ("Would you like me to rephrase the question?")
           - Give them an option ("We can skip this if you'd prefer")
           - Simplify the question or ask something easier
        3. **If they seem unprepared or stressed:** Suggest they can reschedule if needed
        4. **If their response is unclear:** Gently guide them back with encouragement

        **Communication Style:**
        - Use simple, everyday English
        - Be patient and understanding
        - Give brief, constructive feedback
        - Keep questions conversational, not formal
        - Always be supportive and reassuring

        **Output Format (Strictly follow this):**
        UNDERSTANDING: [YES or NO]
        CONCEPT: [The concept for your next question]
        QUESTION: [Your supportive, conversational question in simple English]
        """)
        
        self.evaluation_prompt = PromptTemplate.from_template("""
        You are evaluating a student's performance in a supportive voice-based daily standup on the topic below.
        The standup assessed understanding in a friendly, encouraging environment.

        Lecture Summary:
        {summary}

        Full Q&A log:
        {conversation}

        Generate a kind but honest evaluation. Your response must include the following sections:
        1. Key Strengths: (2-3 positive points about their performance)
        2. Areas for Growth: (1-2 gentle suggestions for improvement)
        3. Concept Understanding: (Brief summary of how well they grasped the concepts)
        4. Encouragement: (One supportive recommendation for continued learning)
        5. Final Score: A fair score on a separate line, in the format 'Final Score: X/10'.

        Be encouraging but fair with the score. Consider their effort and engagement, not just technical accuracy.
        Keep the total evaluation under 200 words and maintain a supportive, constructive tone.
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
                              current_question_number: int, 
                              total_questions: int) -> Dict[str, str]:
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
# Audio utilities
# ========================

class AudioManager:
    """Manages audio transcription and text-to-speech"""
    
    @staticmethod
    def clean_audio_folder():
        """Clean up old audio files"""
        try:
            current_time = time.time()
            cleanup_count = 0
            
            for filename in os.listdir(AUDIO_DIR):
                if filename.endswith(".mp3"):
                    file_path = os.path.join(AUDIO_DIR, filename)
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    # Remove files older than 1 hour
                    if file_age > 3600:
                        try:
                            os.remove(file_path)
                            cleanup_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete {filename}: {e}")
            
            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} old audio files")
                
        except Exception as e:
            logger.error(f"Error during audio cleanup: {e}")
    
    @staticmethod
    async def text_to_speech(text: str, voice: str, speed: float = TTS_SPEED) -> Optional[str]:
        """Convert text to speech using Edge TTS"""
        timestamp = int(time.time() * 1000)
        raw_path = os.path.join(AUDIO_DIR, f"ai_raw_{timestamp}.mp3")
        final_path = os.path.join(AUDIO_DIR, f"ai_{timestamp}.mp3")
        
        try:
            # Clean old files before generating new ones
            AudioManager.clean_audio_folder()
            
            # Generate TTS audio
            await edge_tts.Communicate(text, voice).save(raw_path)

            # Apply speed adjustment with ffmpeg
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-filter:a", f"atempo={speed}", "-vn", final_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Clean up raw file
            if os.path.exists(raw_path):
                os.remove(raw_path)
            
            # Wait for final file to appear and return relative path
            for _ in range(10):
                if os.path.exists(final_path):
                    return f"/audio/{os.path.basename(final_path)}"
                await asyncio.sleep(0.1)

            logger.error(f"TTS final audio file missing after wait: {final_path}")
            return None

        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            # Clean up any partial files
            for path in [raw_path, final_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
        
        return None
    
    @staticmethod
    def transcribe(filepath: str) -> str:
        """Transcribe audio using Groq's Whisper API"""
        logger.info(f"Transcribing audio: {filepath}")
        try:
            from groq import Groq
            groq_client = Groq()
            
            with open(filepath, "rb") as file:
                result = groq_client.audio.transcriptions.create(
                    file=(filepath, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json"
                )
            
            transcribed_text = result.text.strip()
            logger.info(f"Transcription successful: {transcribed_text[:100]}...")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")

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

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Daily Standup Voice Testing API",
        "timestamp": time.time()
    }

@app.get("/api/info", tags=["info"])
async def api_info():
    """API information endpoint"""
    return {
        "name": "Daily Standup Voice Testing API",
        "version": "1.0.0",
        "description": "Voice-based daily standup testing system",
        "endpoints": {
            "start_test": "POST /start_test - Start a new test session",
            "record_and_respond": "POST /record_and_respond - Process audio response",
            "summary": "GET /summary?test_id={id} - Get test evaluation",
            "download_results": "GET /api/download_results/{test_id} - Download PDF results",
            "tests": "GET /api/tests - Get all test results",
            "cleanup": "GET /cleanup - Clean up resources"
        },
        "audio_serving": "/audio/{filename} - Serve generated audio files"
    }

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
        
        estimated_duration = TOTAL_QUESTIONS * ESTIMATED_SECONDS_PER_QUESTION
        return {
            "test_id": test_id, 
            "question": question, 
            "audio_path": audio_path or "",
            "duration_sec": estimated_duration
        }
    
    except Exception as e:
        logger.error(f"Error starting test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start test: {str(e)}")

from fastapi import UploadFile, File, Form

@app.post("/record_and_respond", response_model=ConversationResponse)
async def record_and_respond(
    audio: UploadFile = File(...),
    test_id: str = Form(...)
):
    """Process user's uploaded audio response and provide the next question"""
    try:
        test = test_manager.validate_test(test_id)

        # Validate uploaded file
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file format")

        # Save uploaded audio to file system
        audio_filename = os.path.join(TEMP_DIR, f"user_audio_{int(time.time())}_{test_id}.webm")
        try:
            content = await audio.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty audio file received")
                
            with open(audio_filename, "wb") as f:
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save audio file: {str(e)}")

        # Transcribe the uploaded audio
        try:
            user_response = AudioManager.transcribe(audio_filename)
            if not user_response.strip():
                raise HTTPException(status_code=400, detail="No speech detected in audio")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")
        finally:
            # Clean up the temporary audio file
            try:
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
            except Exception as e:
                logger.warning(f"Failed to clean up audio file {audio_filename}: {e}")

        logger.info(f"Test {test_id}: Transcribed user response: {user_response}")

        # Log the user's answer
        test_manager.add_answer(test_id, user_response)

        # Check if the test is now complete
        if test.question_index >= TOTAL_QUESTIONS:
            logger.info(f"Test {test_id} completed. Question index ({test.question_index}) reached TOTAL_QUESTIONS ({TOTAL_QUESTIONS}).")
            closing_message = "The test has ended. Thank you for your participation."
            audio_path = await AudioManager.text_to_speech(closing_message, test.voice)
            return {
                "ended": True, 
                "response": closing_message, 
                "audio_path": audio_path or ""
            }

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
            test.question_index + 1,  # Next question number
            TOTAL_QUESTIONS
        )

        # Extract next question and concept
        next_question = followup_data.get("question", "Can you elaborate more on that?")
        next_concept = followup_data.get("concept", last_concept)

        # Update test log and synthesize speech
        test_manager.add_question(test_id, next_question, next_concept)
        audio_path = await AudioManager.text_to_speech(next_question, test.voice)

        return {
            "ended": False,
            "response": next_question,
            "audio_path": audio_path or "",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing response for test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/summary", response_model=SummaryResponse)
async def get_summary(test_id: str):
    """Get a summary evaluation of the test session"""
    logger.info(f"Summary endpoint called for test_id: {test_id}")
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
            logger.error(f"Failed to save test data for test {test_id} after evaluation.")
        
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
            "pdf_url": f"/api/download_results/{test_id}"
        }

    except HTTPException:
        raise
    except Exception as e: # Log traceback for general exceptions
        logger.error(f"Error generating summary for test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@app.get("/cleanup")
async def cleanup_resources():
    """Clean up audio files and expired tests"""
    try:
        # Clean up audio files
        AudioManager.clean_audio_folder()
        
        # Clean up expired tests
        expired_count = test_manager.cleanup_expired_tests()
        
        return {
            "message": f"Cleaned up {expired_count} expired tests and old audio files",
            "expired_tests": expired_count,
            "timestamp": time.time()
        }
    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
 
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
        return {
            "tests": results,
            "count": len(results),
            "timestamp": time.time()
        }
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching test {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve test")
    
@app.get("/api/standup-students", response_class=JSONResponse)
async def get_unique_standup_students():
    """Return distinct Student_ID and name from daily standup results."""
    try:
        pipeline = [
            {"$group": {"_id": "$Student_ID", "name": {"$first": "$name"}}},
            {"$project": {"_id": 0, "Student_ID": "$_id", "name": 1}},
            {"$sort": {"Student_ID": 1}}
        ]
        students = list(db_manager.conversations.aggregate(pipeline))
        return {
            "count": len(students), 
            "students": students,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error fetching standup students: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve students")

@app.get("/api/standup-students/{student_id}/tests", response_class=JSONResponse)
async def get_tests_for_standup_student(student_id: str):
    """Get all test documents for a specific student, excluding conversation and evaluation."""
    try:
        student_id_int = int(student_id)
        results = list(db_manager.conversations.find(
            {"Student_ID": student_id_int},
            {"_id": 0, "conversation_log": 0, "evaluation": 0, "timestamp": 0}
        ).sort("timestamp", -1))  # Sort by most recent first
        
        if not results:
            raise HTTPException(status_code=404, detail="No tests found for this student")
        
        return {
            "count": len(results), 
            "tests": results,
            "student_id": student_id_int,
            "timestamp": time.time()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid student ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching tests for student ID {student_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tests for student")

# ========================
# Additional API endpoints for monitoring
# ========================

@app.get("/api/stats", response_class=JSONResponse)
async def get_system_stats():
    """Get system statistics"""
    try:
        # Count active tests
        active_tests = len(test_manager.tests)
        
        # Count audio files
        audio_files = len([f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')])
        
        # Get database stats
        total_tests = db_manager.conversations.count_documents({})
        
        return {
            "active_test_sessions": active_tests,
            "audio_files_on_disk": audio_files,
            "total_completed_tests": total_tests,
            "audio_directory": AUDIO_DIR,
            "temp_directory": TEMP_DIR,
            "cors_origins": FRONTEND_ORIGIN,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error fetching system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")

@app.delete("/api/tests/{test_id}")
async def delete_test_result(test_id: str):
    """Delete a specific test result from the database"""
    try:
        result = db_manager.conversations.delete_one({"test_id": test_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Test not found")
        
        logger.info(f"Deleted test result: {test_id}")
        return {
            "message": f"Test {test_id} deleted successfully",
            "deleted_count": result.deleted_count,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting test {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete test")

# ========================
# Error handlers
# ========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unhandled errors"""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )
try:
    app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
    logger.info(f"✅ Audio files served from: /audio (directory: {AUDIO_DIR})")
except Exception as e:
    logger.error(f"❌ Failed to mount audio directory: {e}")


from fastapi.responses import HTMLResponse
import os

@app.get("/test", response_class=HTMLResponse)
async def serve_test_page():
    """Serve the conversation test HTML page"""
    html_file = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_file):
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Update the API URL in the HTML to be relative
        html_content = html_content.replace(
            "const API_BASE_URL = 'https://192.168.48.11:8060/daily_standup';",
            "const API_BASE_URL = window.location.origin + '/daily_standup';"
        )
        
        return HTMLResponse(content=html_content)
    else:
        return HTMLResponse(content="<h1>Test page not found</h1><p>Please place index.html in the daily_standup directory</p>")


@app.get("/")
async def daily_standup_root():
    """Root endpoint for daily standup sub-app"""
    return {
        "service": "Daily Standup Voice Testing API",
        "version": "1.0.0",
        "status": "running",
        "base_url": "/daily_standup",
        "test_page": "/daily_standup/test",
        "endpoints": {
            "health": "/daily_standup/health",
            "test_frontend": "/daily_standup/test",
            "start_test": "/daily_standup/start_test", 
            "record_and_respond": "/daily_standup/record_and_respond",
            "summary": "/daily_standup/summary",
            "download_pdf": "/daily_standup/api/download_results/{test_id}",
            "all_tests": "/daily_standup/api/tests",
            "students": "/daily_standup/api/standup-students",
            "cleanup": "/daily_standup/cleanup",
            "audio_files": "/daily_standup/audio/{filename}"
        },
        "instructions": {
            "test_conversation": "Visit /daily_standup/test to test the voice conversation",
            "api_docs": "Visit /daily_standup/docs for interactive API documentation"
        },
        "timestamp": time.time()
    }
    
@app.get("/debug/test-mongo") # https://192.168.48.31:8060/daily_standup/debug/test-mongo
async def test_mongodb_connection():
    """Debug endpoint to test MongoDB connection and data"""
    try:
        # Test MongoDB connection
        count = db_manager.conversations.count_documents({})
        
        # Get recent test
        recent_test = db_manager.conversations.find_one(
            {}, 
            sort=[("timestamp", -1)]
        )
        
        return {
            "mongodb_connected": True,
            "total_tests_saved": count,
            "latest_test": {
                "test_id": recent_test.get("test_id") if recent_test else None,
                "timestamp": recent_test.get("timestamp") if recent_test else None,
                "name": recent_test.get("name") if recent_test else None
            } if recent_test else None
        }
    except Exception as e:
        return {
            "mongodb_connected": False,
            "error": str(e)
        }
        
        
# ========================
# Development utilities
# ========================

if __name__ == "__main__":
    import uvicorn
    import socket
    
    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    local_ip = get_local_ip()
    port = 8061  # Different port to avoid conflicts with main launcher
    
    print(f"🚀 Starting Daily Standup API Server")
    print(f"📡 Server: https://{local_ip}:{port}")
    print(f"📋 API Docs: https://{local_ip}:{port}/docs")
    print(f"🔊 Audio Files: https://{local_ip}:{port}/audio/")
    print(f"🌐 CORS Origins: {FRONTEND_ORIGIN}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
        ssl_certfile="certs/cert.pem",
        ssl_keyfile="certs/key.pem",
    )