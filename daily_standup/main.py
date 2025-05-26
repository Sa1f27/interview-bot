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
import subprocess
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import edge_tts
import pymongo
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
BLOCK_SIZE = 4096
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 2.0
MAX_RECORDING_DURATION = 10.0
TEST_DURATION_SEC = 120
INACTIVITY_TIMEOUT = 120
AUDIO_DIR = "audio"
TTS_SPEED = 1.2

# ========================
# Models and schemas
# ========================

class Session:
    """Session data model for a test session"""
    def __init__(self, summary: str, voice: str):
        self.summary = summary
        self.voice = voice
        self.conversation_log = []
        self.deadline = time.time() + TEST_DURATION_SEC
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
    session_id: str

class SessionResponse(BaseModel):
    """Response model for session creation"""
    session_id: str
    question: str
    audio_path: str

class ConversationResponse(BaseModel):
    """Response model for conversation updates"""
    ended: bool
    response: str
    audio_path: str
    feedback: Optional[str] = None

class SummaryResponse(BaseModel):
    """Response model for test summary"""
    summary: str
    analytics: Dict[str, Any]

# ========================
# Application setup
# ========================

# Initialize FastAPI app
app = FastAPI(title="Voice-Based Testing System")

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="static")
app.mount("/audio", StaticFiles(directory=os.path.join(BASE_DIR, "audio")), name="audio")

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# ========================
# Database setup
# ========================

class DatabaseManager:
    """MongoDB database manager"""
    def __init__(self, connection_string, db_name):
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[db_name]
        self.transcripts = self.db["drive"]
    
    def get_latest_summary(self) -> str:
        """Fetch the latest lecture summary from the database"""
        doc = self.transcripts.find_one({}, {"_id": 0, "summary": 1}, sort=[("timestamp", -1)])
        if not doc or "summary" not in doc:
            raise ValueError("No summary found in the database")
        return doc["summary"]
    
    def close(self):
        """Close the database connection"""
        self.client.close()

# Initialize database
db_manager = DatabaseManager(
    "mongodb://sa:L%40nc%5Eere%400012@192.168.48.200:27017/?authSource=admin", 
    "test"
)

# ========================
# Session management
# ========================

class SessionManager:
    """Manages test sessions"""
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
    
    def create_session(self, summary: str, voice: str) -> str:
        """Create a new test session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(summary, voice)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID"""
        return self.sessions.get(session_id)
    
    def validate_session(self, session_id: str) -> Session:
        """Validate session ID and update last activity"""
        session = self.get_session(session_id)
        if not session:
            raise HTTPException(status_code=400, detail="Session not found")
        
        if time.time() > session.last_activity + INACTIVITY_TIMEOUT:
            raise HTTPException(status_code=400, detail="Session timed out")
        
        session.last_activity = time.time()
        return session
    
    def add_question(self, session_id: str, question: str, concept: str = None):
        """Add a question to the session conversation log"""
        session = self.validate_session(session_id)
        session.conversation_log.append(ConversationEntry(question=question, concept=concept))
        session.current_concept = concept
        session.question_index += 1
    
    def add_answer(self, session_id: str, answer: str):
        """Add an answer to the last question in the conversation log"""
        session = self.validate_session(session_id)
        if session.conversation_log:
            session.conversation_log[-1].answer = answer
    
    def get_conversation_history(self, session_id: str) -> str:
        """Get formatted conversation history for a session"""
        session = self.validate_session(session_id)
        history = []
        for entry in session.conversation_log:
            q_line = f"Q: {entry.question}"
            a_line = f"A: {entry.answer}" if entry.answer else "A: "
            history.append(f"{q_line}\n{a_line}")
        return "\n\n".join(history)
    
    def is_test_ended(self, session_id: str) -> bool:
        """Check if the test has ended"""
        session = self.validate_session(session_id)
        return time.time() > session.deadline

# Initialize session manager
session_manager = SessionManager()

# ========================
# LLM and prompt setup
# ========================

class LLMManager:
    """Manages LLM interactions for question generation and evaluation"""
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
        self.parser = StrOutputParser()
        
        # Define prompts
        self.question_prompt = PromptTemplate.from_template("""
        You are conducting a voice-based test. Analyze the lecture summary below to ask the next 
        most important question to test the student's understanding of a key concept.

        Lecture Summary:
        {summary}

        Conversation so far:
        {history}

        Extract one key concept from the lecture that hasn't been covered yet.
        Then formulate a clear, concise question about that concept.

        First identify the concept, then ask one question about it.
        Format your response as:
        CONCEPT: [the concept]
        QUESTION: [your question]
        """)
        
        self.followup_prompt = PromptTemplate.from_template("""
        You are conducting a voice-based test.

        Lecture Summary:
        {summary}

        Current concept being tested: {concept}

        Conversation so far:
        {history}

        Previous question: {previous_question}
        User's response: {user_response}

        Evaluate whether the user's response demonstrates understanding of the concept.
        
        If their response shows understanding:
            - Extract a new key concept from the lecture that hasn't been covered
            - Create a new question about that concept
        
        If their response does NOT show understanding:
            - Ask a simpler follow-up question about the SAME concept

        Format your response as:
        UNDERSTANDING: [YES or NO]
        CONCEPT: [same concept or new concept]
        QUESTION: [your follow-up question]
        FEEDBACK: [brief constructive feedback - 1-2 sentences max]
        """)
        
        self.evaluation_prompt = PromptTemplate.from_template("""
        You are evaluating a student's voice-based test on the topic below:

        Lecture Summary:
        {summary}

        Full Q&A log:
        {conversation}

        Generate a concise evaluation with:
        1. Key strengths (2-3 points)
        2. Areas for improvement (1-2 points)
        3. How well they covered the core concepts
        4. One specific recommendation for further study

        Be constructive and specific. Do NOT give numeric scores.
        Keep your evaluation under 200 words.
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
                              concept: str) -> Dict[str, str]:
        """Generate a follow-up question based on the user's response"""
        chain = self.followup_prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "summary": summary,
            "history": history,
            "previous_question": previous_question,
            "user_response": user_response,
            "concept": concept
        })
        return self._parse_llm_response(
            response, 
            ["UNDERSTANDING", "CONCEPT", "QUESTION", "FEEDBACK"]
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
    """Manages audio recording, transcription and text-to-speech"""
    
    @staticmethod
    def record_audio() -> Optional[str]:
        """Record audio from the microphone and save to a file"""
        logger.info("Recording audio...")
        chunks, silence_start, recording = [], None, True
        start_time = time.time()

        def callback(indata, frames, time_info, status):
            nonlocal silence_start, recording
            if status:
                logger.error(f"Audio recording error: {status}")
                recording = False
                return
            
            rms = np.sqrt(np.mean(indata**2))
            chunks.append(indata.copy())
            
            # Check for silence
            if rms < SILENCE_THRESHOLD:
                silence_start = silence_start or time.time()
                if (time.time() - silence_start) > SILENCE_DURATION:
                    recording = False
            else:
                silence_start = None
            
            # Check for maximum duration
            if (time.time() - start_time) > MAX_RECORDING_DURATION:
                recording = False

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, 
                channels=1, 
                blocksize=BLOCK_SIZE, 
                callback=callback
            ):
                while recording:
                    sd.sleep(100)
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None

        # Process recorded audio
        if not chunks:
            return None
            
        audio = np.concatenate(chunks)
        if len(audio) / SAMPLE_RATE < 0.5:  # Too short
            return None

        filepath = os.path.join(AUDIO_DIR, f"temp_in_{int(time.time())}.wav")
        wavfile.write(filepath, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        return filepath
    
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
                return f"/{AUDIO_DIR}/{os.path.basename(final_path)}"
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
    voices = ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural", "en-IN-AditiNeural"]
    return random.choice(voices)

# ========================
# API Endpoints
# ========================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main application page"""
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

@app.get("/start_test", response_model=SessionResponse)
async def start_test():
    """Start a new test session"""
    try:
        # Get latest lecture summary
        summary = db_manager.get_latest_summary()
        
        # Create a new session
        voice = get_random_voice()
        session_id = session_manager.create_session(summary, voice)
        
        # Generate first question
        question_data = await llm_manager.generate_first_question(summary)
        question = question_data.get("question", "What can you tell me about this topic?")
        concept = question_data.get("concept", "general understanding")
        
        # Add question to session
        session_manager.add_question(session_id, question, concept)
        
        # Generate audio for the question
        audio_path = await AudioManager.text_to_speech(question, voice)
        
        return {"session_id": session_id, "question": question, "audio_path": audio_path}
    
    except Exception as e:
        logger.error(f"Error starting test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import UploadFile, File, Form

@app.post("/record_and_respond", response_model=ConversationResponse)
async def record_and_respond(
    session_id: str = Form(...),
    audio: UploadFile = File(...)
):
    """Process uploaded audio and provide the next question"""
    try:
        session = session_manager.validate_session(session_id)

        # Check if test has ended
        if session_manager.is_test_ended(session_id):
            closing_message = "The test has ended. Thank you for your participation."
            audio_path = await AudioManager.text_to_speech(closing_message, session.voice)
            return {
                "ended": True,
                "response": closing_message,
                "audio_path": audio_path
            }

        # Save uploaded audio file
        audio_filename = f"user_{int(time.time())}.webm"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        with open(audio_path, "wb") as out_file:
            out_file.write(await audio.read())

        # Transcribe user response
        user_response = AudioManager.transcribe(audio_path)

        # Log the answer to the last question
        last_question = session.conversation_log[-1].question
        last_concept = session.conversation_log[-1].concept
        session_manager.add_answer(session_id, user_response)

        # Generate follow-up
        history = session_manager.get_conversation_history(session_id)
        followup_data = await llm_manager.generate_followup(
            session.summary,
            history,
            last_question,
            user_response,
            last_concept
        )

        next_question = followup_data.get("question", "Can you elaborate more on that?")
        next_concept = followup_data.get("concept", last_concept)
        feedback = followup_data.get("feedback", "")

        session_manager.add_question(session_id, next_question, next_concept)

        # Generate audio
        next_audio_path = await AudioManager.text_to_speech(next_question, session.voice)

        return {
            "ended": False,
            "response": next_question,
            "audio_path": next_audio_path,
            "feedback": feedback
        }

    except Exception as e:
        logger.error(f"Error processing response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary", response_model=SummaryResponse)
async def get_summary(session_id: str):
    """Get a summary evaluation of the test session"""
    try:
        session = session_manager.validate_session(session_id)
        history = session_manager.get_conversation_history(session_id)
        
        # Generate evaluation
        evaluation = await llm_manager.generate_evaluation(
            session.summary,
            history
        )
        
        # Calculate analytics
        num_questions = len(session.conversation_log)
        answers = [entry.answer for entry in session.conversation_log if entry.answer]
        avg_length = sum(len(answer.split()) for answer in answers) / len(answers) if answers else 0
        
        # Calculate concept coverage
        unique_concepts = set(entry.concept for entry in session.conversation_log if entry.concept)
        concept_coverage = len(unique_concepts)
        
        return {
            "summary": evaluation,
            "analytics": {
                "num_questions": num_questions,
                "avg_response_length": round(avg_length, 1),
                "concept_coverage": concept_coverage
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cleanup endpoint
@app.get("/cleanup")
async def cleanup_resources():
    """Clean up audio files and expired sessions"""
    try:
        # Clean up audio files
        AudioManager.clean_audio_folder()
        
        # Clean up expired sessions
        current_time = time.time()
        expired_sessions = [
            sid for sid, session in session_manager.sessions.items()
            if current_time > session.last_activity + INACTIVITY_TIMEOUT * 2
        ]
        
        for sid in expired_sessions:
            session_manager.sessions.pop(sid, None)
        
        return {"message": f"Cleaned up {len(expired_sessions)} expired sessions and old audio files"}
    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Handle application shutdown
@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources on application shutdown"""
    db_manager.close()
    AudioManager.clean_audio_folder()

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7008, reload=True)