from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import asyncio
import uuid
import logging
import random
import edge_tts
import subprocess
from typing import Dict, List, Optional
from pydantic import BaseModel
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
INACTIVITY_TIMEOUT = 300  # 5 minutes
TTS_SPEED = 1.2

# Environment configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

# =======================
# Models and schemas
# =======================

class Session:
    """Session data model"""
    def __init__(self, summary: str, voice: str):
        self.summary = summary
        self.voice = voice
        self.conversation_log: List[Dict[str, str]] = []
        self.last_activity = time.time()
        self.question_index = 0
        self.current_concept = "General"
        self.greeting_step = 0

class TestManager:
    """Manages test sessions"""
    def __init__(self):
        self.tests: Dict[str, Session] = {}

    def create_test(self, summary: str, voice: str) -> str:
        """Create a new test session"""
        test_id = str(uuid.uuid4())
        self.tests[test_id] = Session(summary, voice)
        logger.info(f"Created test {test_id}")
        return test_id

    def get_test(self, test_id: str) -> Optional[Session]:
        """Get a test by ID"""
        return self.tests.get(test_id)

    def validate_test(self, test_id: str) -> Session:
        """Validate test ID and update last activity"""
        test = self.get_test(test_id)
        if not test:
            raise ValueError("Test not found")
        
        if time.time() > test.last_activity + INACTIVITY_TIMEOUT:
            self.tests.pop(test_id, None)
            raise ValueError("Test timed out")
        
        test.last_activity = time.time()
        return test

    def add_entry(self, test_id: str, role: str, content: str):
        """Add an entry to the conversation log"""
        test = self.validate_test(test_id)
        test.conversation_log.append({"role": role, "content": content})
        if role == "assistant":
            test.question_index += 1

# Initialize test manager
test_manager = TestManager()

# =======================
# AI and prompt setup
# =======================

class LLMManager:
    """Manages LLM interactions (simulated)"""
    def __init__(self):
        self.greetings = [
            "Hi there! I'm your AI interviewer. Are you ready to begin?",
            "Hello! I'll be conducting your interview today. Shall we start?",
            "Welcome! I'm ready to start the interview when you are."
        ]
        self.questions = [
            "Tell me about a recent project you're proud of.",
            "What's a major challenge you faced recently and how did you handle it?",
            "Describe your experience with Python and web development.",
            "How do you approach learning a new technology?",
            "What are your career goals for the next five years?",
            "This is the end of the interview. Thank you for your time."
        ]

    async def generate_initial_greeting(self) -> str:
        """Generate the initial greeting"""
        return random.choice(self.greetings)

    async def generate_next_question(self, test_id: str) -> str:
        """Generate the next question"""
        test = test_manager.validate_test(test_id)
        if test.question_index < len(self.questions):
            return self.questions[test.question_index]
        return "Thank you for your time. The interview is now complete."

    async def generate_evaluation(self, test_id: str) -> str:
        """Generate an evaluation of the test session"""
        test = test_manager.validate_test(test_id)
        # In a real scenario, you'd use an LLM to evaluate the conversation.
        # Here, we'll just provide a simple summary.
        num_questions = len([entry for entry in test.conversation_log if entry["role"] == "assistant"])
        num_answers = len([entry for entry in test.conversation_log if entry["role"] == "user"])
        return f"The interview consisted of {num_questions} questions and {num_answers} answers. The candidate was responsive and engaged."

# Initialize LLM manager
llm_manager = LLMManager()

# =======================
# Audio utilities
# =======================

class AudioManager:
    """Manages audio transcription and text-to-speech"""
    
    @staticmethod
    async def text_to_speech(text: str, voice: str, speed: float = TTS_SPEED) -> Optional[str]:
        """Convert text to speech using Edge TTS"""
        timestamp = int(time.time() * 1000)
        # Ensure audio directory exists
        audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
        os.makedirs(audio_dir, exist_ok=True)

        raw_path = os.path.join(audio_dir, f"ai_raw_{timestamp}.mp3")
        final_path = os.path.join(audio_dir, f"ai_{timestamp}.mp3")
        
        try:
            await edge_tts.Communicate(text, voice).save(raw_path)

            # Apply speed adjustment with ffmpeg
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-filter:a", f"atempo={speed}", "-vn", final_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            if os.path.exists(raw_path):
                os.remove(raw_path)
            
            return f"/audio/{os.path.basename(final_path)}"

        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            return None

    @staticmethod
    def get_random_voice() -> str:
        """Get a random TTS voice"""
        voices = ["en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"]
        return random.choice(voices)

# =======================
# Application setup
# =======================

app = FastAPI(
    title="AI Interview API",
    description="API for AI-driven interviews",
    version="2.0.0",
)

# Get base directory and setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static File Serving
# Mount the 'audio' directory to serve generated audio files.
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
# Mount the 'static' directory to serve JS, CSS, etc.
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=FileResponse)
async def read_index():
    """Serves the main index.html file."""
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

@app.post("/interview/{test_id}")
async def interview_step(test_id: str, request: Request):
    """Handles a single step of the interview."""
    try:
        test = test_manager.validate_test(test_id)
        
        # The client sends raw audio blob.
        audio_data = await request.body()

        # Here you would typically save the audio and transcribe it.
        # For this refactor, we'll simulate transcription.
        user_response = "This is a simulated user response."
        logger.info(f"Test {test_id}: Simulated user response: {user_response}")
        test_manager.add_entry(test_id, "user", user_response)

        # Generate the next question
        next_question = await llm_manager.generate_next_question(test_id)
        test_manager.add_entry(test_id, "assistant", next_question)
        
        # Generate audio for the next question
        audio_path = await AudioManager.text_to_speech(next_question, test.voice)
        
        # Send the question and audio path back to the client
        response = {
            "text": next_question,
            "audio_path": audio_path,
            "ended": "interview is now complete" in next_question.lower()
        }
        
        return response

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Interview step error for test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start_test")
async def start_test():
    """Starts a new interview session."""
    try:
        # The summary is now a placeholder
        summary = "This is a placeholder summary for the interview."
        voice = AudioManager.get_random_voice()
        test_id = test_manager.create_test(summary, voice)
        
        initial_question = await llm_manager.generate_initial_greeting()
        test_manager.add_entry(test_id, "assistant", initial_question)
        
        audio_path = await AudioManager.text_to_speech(initial_question, voice)
        
        return {
            "test_id": test_id,
            "question": initial_question,
            "audio_path": audio_path
        }
    except Exception as e:
        logger.error(f"Error starting test: {e}")
        raise HTTPException(status_code=500, detail="Failed to start test")

@app.get("/summary/{test_id}")
async def get_summary(test_id: str):
    """Get a summary of the interview."""
    try:
        test = test_manager.validate_test(test_id)
        evaluation = await llm_manager.generate_evaluation(test_id)
        return {
            "summary": evaluation,
            "conversation_log": test.conversation_log
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating summary for test {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary")


if __name__ == "__main__":
    import uvicorn
    
    print("Starting AI Interview API Server")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
