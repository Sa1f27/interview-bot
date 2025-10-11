import time
import uuid
import logging
import random
import edge_tts
import os
import subprocess
from typing import Dict, List, Optional
from groq import AsyncGroq

logger = logging.getLogger(__name__)

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY environment variable not set. LLM calls will fail.")


INACTIVITY_TIMEOUT = 300  # 5 minutes
TTS_SPEED = 1.2

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
        self.client = AsyncGroq(api_key=GROQ_API_KEY)

    async def generate_initial_greeting(self) -> str:
        return random.choice(self.greetings)

    async def _generate_response(self, system_prompt: str, test: Session) -> str:
        """
        This is a placeholder for a real LLM call.
        It simulates generating a response based on a system prompt and conversation history.
        """
        if not self.client.api_key:
            logger.error("Groq API key is not configured.")
            return "Error: AI model is not configured. Please set the GROQ_API_KEY."

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(test.conversation_log)

        try:
            logger.info(f"Calling Groq LLM for test...")
            response = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=100, # Reduced max_tokens for more concise questions
                temperature=0.7,
            )
            content = response.choices[0].message.content.strip()
            
            # Add a concluding phrase if the LLM indicates the end.
            if test.question_index >= 6:
                 return content + " The interview is now complete."

            return content
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return "I'm having trouble formulating my next question. Let's try again. Can you elaborate on your last point?"

    async def generate_next_question(self, test: Session) -> str:
        """
        Generates the next interview question based on the applicant's data and conversation history.
        """
        system_prompt = (
            "You are a strict and highly-technical interrogator. Your goal is to rigorously verify the candidate's skills and experience. "
            "Do not use pleasantries. Be direct, concise, and demanding. "
            "Ask one, and only one, probing follow-up question based on their last response and their profile. "
            "Challenge their answers. Ask for specific examples, metrics, and proof. "
            "If their answer is vague, demand specifics. If they claim success, question how it was measured. "
            "After about 6 questions, you must conclude the interview. "
            f"Here is the candidate's profile summary:\n---\n{test.summary}\n---"
        )
        return await self._generate_response(system_prompt, test)

    async def generate_evaluation(self, test: Session) -> str:
        """Generates a final evaluation of the interview."""
        system_prompt = (
            "You are a senior hiring manager reviewing a technical interrogation. "
            "You will be given the candidate's profile and the full interview transcript. "
            "Your task is to provide a brutally honest, concise evaluation. "
            "Focus on: 1. Technical depth shown. 2. Alignment between their profile and their answers. 3. Red flags or inconsistencies. 4. A final hire/no-hire recommendation. "
            "Use Markdown for formatting."
            f"Here is the candidate's profile summary:\n---\n{test.summary}\n---"
        )
        return await self._generate_response(system_prompt, test)

# =======================
# Audio utilities
# =======================

class AudioManager:
    """Manages audio transcription and text-to-speech"""
    
    @staticmethod
    async def text_to_speech(text: str, voice: str, speed: float = TTS_SPEED) -> Optional[str]:
        timestamp = int(time.time() * 1000)
        audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
        os.makedirs(audio_dir, exist_ok=True)

        raw_path = os.path.join(audio_dir, f"ai_raw_{timestamp}.mp3")
        final_path = os.path.join(audio_dir, f"ai_{timestamp}.mp3")
        
        try:
            await edge_tts.Communicate(text, voice).save(raw_path)
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
        voices = ["en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"]
        return random.choice(voices)

# =======================
# Singleton instances
# =======================

test_manager = TestManager()
llm_manager = LLMManager()
audio_manager = AudioManager()