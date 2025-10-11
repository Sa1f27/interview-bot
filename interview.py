import time
import uuid
import logging
import random
import edge_tts
import os
import subprocess
from typing import Dict, List, Optional
from groq import Groq
import tempfile

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

INACTIVITY_TIMEOUT = 600
TTS_SPEED = 1.15
MAX_QUESTIONS = 6
RESPONSE_TIMEOUT = 8  # seconds - move to next question if no valid response

class Session:
    def __init__(self, summary: str, voice: str, applicant_id: str):
        self.summary = summary
        self.voice = voice
        self.applicant_id = applicant_id  # Store applicant ID
        self.conversation_log: List[Dict[str, str]] = []
        self.last_activity = time.time()
        self.question_index = 0
        self.is_complete = False

class TestManager:
    def __init__(self):
        self.tests: Dict[str, Session] = {}

    def create_test(self, summary: str, voice: str, applicant_id: str) -> str:
        test_id = str(uuid.uuid4())
        self.tests[test_id] = Session(summary, voice, applicant_id)
        logger.info(f"Created test {test_id} for applicant {applicant_id}")
        return test_id

    def get_test(self, test_id: str) -> Optional[Session]:
        return self.tests.get(test_id)

    def validate_test(self, test_id: str) -> Session:
        test = self.get_test(test_id)
        if not test:
            raise ValueError("Test not found")
        if time.time() > test.last_activity + INACTIVITY_TIMEOUT:
            self.tests.pop(test_id, None)
            raise ValueError("Test timed out")
        test.last_activity = time.time()
        return test

    def add_entry(self, test_id: str, role: str, content: str):
        test = self.validate_test(test_id)
        test.conversation_log.append({"role": role, "content": content})
        if role == "assistant":
            test.question_index += 1
            
    def mark_complete(self, test_id: str):
        test = self.get_test(test_id)
        if test:
            test.is_complete = True

class LLMManager:
    def __init__(self):
        self.greetings = [
            "Hello! I've reviewed your profile. Let's begin with your background and experience.",
            "Good to meet you! I'm ready to assess your technical skills. Let's start.",
            "Welcome! I've studied your application. Let's dive into your expertise."
        ]

    def generate_initial_greeting(self, summary: str) -> str:
        try:
            messages = [
                {"role": "system", "content": "You are a technical interviewer. Give a brief greeting (1-2 sentences) and ask the first technical question based on the candidate's profile."},
                {"role": "user", "content": f"Candidate profile:\n{summary}\n\nGreet briefly and ask first question."}
            ]
            
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=0.7,
                max_completion_tokens=150,
                stream=False
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating greeting: {e}")
            return random.choice(self.greetings)

    def generate_next_question(self, test: Session) -> str:
        if test.question_index >= MAX_QUESTIONS:
            return "Thank you for your time and thoughtful responses. This concludes our interview today. We'll be in touch soon."
        
        try:
            system_prompt = (
                f"You are a technical interviewer. Ask ONE specific, probing question based on the candidate's last response. "
                f"Be direct and challenging. Keep it concise (1-2 sentences).\n\n"
                f"Candidate Profile:\n{test.summary}\n\n"
                f"Question {test.question_index + 1} of {MAX_QUESTIONS}. "
                f"After {MAX_QUESTIONS} questions, conclude the interview."
            )
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(test.conversation_log)
            
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=0.8,
                max_completion_tokens=150,
                stream=False
            )
            
            content = completion.choices[0].message.content.strip()
            
            if test.question_index >= MAX_QUESTIONS - 1:
                content += " This will be our final question."
            
            return content
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            return "Could you elaborate more on that?"

    def generate_evaluation(self, test: Session) -> str:
        try:
            system_prompt = (
                f"You are a hiring manager. Review this interview and provide evaluation:\n\n"
                f"## Technical Assessment\n- Knowledge depth\n- Example quality\n\n"
                f"## Profile Alignment\n- Consistency with stated experience\n\n"
                f"## Red Flags\n- Any concerns\n\n"
                f"## Recommendation\n- HIRE / NO HIRE / BORDERLINE with reasons\n\n"
                f"Candidate Profile:\n{test.summary}"
            )
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(test.conversation_log)
            
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=0.7,
                max_completion_tokens=800,
                stream=False
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating evaluation: {e}")
            return "Error generating evaluation."

class AudioManager:
    def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        if not audio_data or len(audio_data) < 1000:
            logger.warning("Audio data too small")
            return None
            
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            wav_path = temp_audio_path.replace(".webm", ".wav")
            
            try:
                subprocess.run([
                    "ffmpeg", "-i", temp_audio_path,
                    "-ar", "16000", "-ac", "1", "-y", wav_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except:
                wav_path = temp_audio_path
            
            with open(wav_path, "rb") as audio_file:
                logger.info("Transcribing with Whisper...")
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(wav_path), audio_file.read()),
                    model="whisper-large-v3-turbo",
                    temperature=0,
                    response_format="verbose_json"
                )
            
            try:
                os.remove(temp_audio_path)
                if wav_path != temp_audio_path:
                    os.remove(wav_path)
            except:
                pass
            
            transcript = transcription.text.strip()
            logger.info(f"Transcribed: {transcript}")
            return transcript if transcript else None
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    async def text_to_speech(self, text: str, voice: str) -> Optional[str]:
        if not text:
            return None
            
        timestamp = int(time.time() * 1000)
        audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
        os.makedirs(audio_dir, exist_ok=True)

        raw_path = os.path.join(audio_dir, f"ai_raw_{timestamp}.mp3")
        final_path = os.path.join(audio_dir, f"ai_{timestamp}.mp3")
        
        try:
            await edge_tts.Communicate(text, voice).save(raw_path)
            
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-filter:a", f"atempo={TTS_SPEED}",
                "-vn", final_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            if os.path.exists(raw_path):
                os.remove(raw_path)
            
            return f"/audio/{os.path.basename(final_path)}"
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    @staticmethod
    def get_random_voice() -> str:
        voices = ["en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"]
        return random.choice(voices)

test_manager = TestManager()
llm_manager = LLMManager()
audio_manager = AudioManager()