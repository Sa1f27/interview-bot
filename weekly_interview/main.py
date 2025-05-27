import os
import time
import uuid
import random
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import subprocess
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import edge_tts
import pymongo
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
# Audio recording parameters
SAMPLE_RATE = 16000
BLOCK_SIZE = 4096
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5
MAX_RECORDING_DURATION = 15.0
MIN_RECORDING_DURATION = 0.5

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# ========================
# Enums and Data Models
# ========================

class RoundType(Enum):
    TECHNICAL = "Technical"
    COMMUNICATION = "Communication" 
    HR = "HR"

class InterviewState(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ROUND_COMPLETE = "round_complete"
    INTERVIEW_COMPLETE = "interview_complete"

@dataclass
class ConversationEntry:
    user_input: str
    ai_response: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class InterviewSession:
    session_id: str
    current_round: RoundType
    state: InterviewState
    voice: str
    conversations: Dict[RoundType, List[ConversationEntry]] = field(default_factory=dict)
    round_start_time: float = field(default_factory=time.time)
    questions_asked: Dict[RoundType, int] = field(default_factory=lambda: {r: 0 for r in RoundType})
    last_activity: float = field(default_factory=time.time)

class RecordRequest(BaseModel):
    session_id: str

# ========================
# Database Manager
# ========================

class DatabaseManager:
    def __init__(self):
        self.client = pymongo.MongoClient(
            "mongodb://sa:L%40nc%5Eere%400012@192.168.48.200:27017/?authSource=admin"
        )
        self.db = self.client["test"]
        self.transcripts = self.db["drive"]
    
    def get_technical_content(self) -> str:
        """Get and summarize technical content from MongoDB"""
        try:
            docs = list(self.transcripts.find({}, {"_id": 0, "summary": 1}).limit(5))
            if not docs:
                return "No technical content available."
            
            summaries = [doc.get("summary", "") for doc in docs if doc.get("summary")]
            return "\n\n".join(summaries[:3])  # Limit to 3 summaries for efficiency
        except Exception as e:
            logger.error(f"Database error: {e}")
            return "Technical content unavailable."

# ========================
# LLM Manager
# ========================

class LLMManager:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, max_tokens=150)
        self.parser = StrOutputParser()
        
        # Optimized prompts for each round
        self.prompts = {
            RoundType.TECHNICAL: PromptTemplate.from_template("""
            You are conducting a technical interview. Based on this content:
            {technical_content}
            
            Conversation history:
            {history}
            
            Ask ONE practical, specific technical question. Questions asked so far: {question_count}
            
            Rules:
            - If this is question 1-2: Ask fundamental concepts
            - If this is question 3-4: Ask application-based questions  
            - If this is question 5+: Ask advanced problem-solving questions
            - If {question_count} >= 6: Say "Let's move to the communication round"
            
            Keep questions concise and focused.
            """),
            
            RoundType.COMMUNICATION: PromptTemplate.from_template("""
            You are testing communication skills. 
            
            Conversation history:
            {history}
            
            Questions asked: {question_count}
            
            Rules:
            - If this is question 1-2: Test clarity and articulation
            - If this is question 3-4: Test persuasion and explanation skills
            - If {question_count} >= 5: Say "Let's proceed to the HR round"
            
            Ask ONE question that tests verbal communication, confidence, or presentation skills.
            """),
            
            RoundType.HR: PromptTemplate.from_template("""
            You are an HR interviewer conducting behavioral assessment.
            
            Conversation history:
            {history}
            
            Questions asked: {question_count}
            
            Rules:
            - If this is question 1-2: Ask about past experiences and teamwork
            - If this is question 3-4: Ask situational/behavioral questions
            - If {question_count} >= 5: Say "Thank you for completing the interview"
            
            Ask ONE behavioral or situational question.
            """)
        }
        
        self.evaluation_prompt = PromptTemplate.from_template("""
        Evaluate this interview performance across all rounds:
        
        Technical Round:
        {technical_conversation}
        
        Communication Round:
        {communication_conversation}
        
        HR Round:
        {hr_conversation}
        
        Provide a concise evaluation with:
        1. Technical competency (2-3 sentences)
        2. Communication skills (2-3 sentences)  
        3. Cultural fit and soft skills (2-3 sentences)
        4. Overall recommendation (1-2 sentences)
        5. Proper scoring (1-10 scale)
        
        Keep the evaluation under 200 words and be constructive.
        """)
    
    async def generate_question(self, round_type: RoundType, history: str, 
                              question_count: int, technical_content: str = "") -> str:
        """Generate the next question for the current round"""
        try:
            prompt = self.prompts[round_type]
            chain = prompt | self.llm | self.parser
            
            response = await chain.ainvoke({
                "history": history,
                "question_count": question_count,
                "technical_content": technical_content
            })
            
            return response.strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._get_fallback_question(round_type, question_count)
    
    def _get_fallback_question(self, round_type: RoundType, question_count: int) -> str:
        """Fallback questions if LLM fails"""
        fallbacks = {
            RoundType.TECHNICAL: [
                "Can you explain your experience with databases?",
                "How do you approach debugging complex issues?",
                "Let's move to the communication round."
            ],
            RoundType.COMMUNICATION: [
                "How would you explain a technical concept to a non-technical person?",
                "Tell me about a time you had to present to a group.",
                "Let's proceed to the HR round."
            ],
            RoundType.HR: [
                "Tell me about a challenging situation you handled at work.",
                "How do you handle feedback and criticism?",
                "Thank you for completing the interview."
            ]
        }
        
        questions = fallbacks[round_type]
        return questions[min(question_count - 1, len(questions) - 1)]
    
    async def generate_evaluation(self, conversations: Dict[RoundType, List[ConversationEntry]]) -> str:
        """Generate final interview evaluation"""
        try:
            # Format conversations for each round
            formatted_convs = {}
            for round_type in RoundType:
                conv_list = conversations.get(round_type, [])
                formatted = "\n".join([
                    f"Q: {entry.ai_response}\nA: {entry.user_input}"
                    for entry in conv_list
                ]) if conv_list else "No conversation recorded."
                formatted_convs[f"{round_type.value.lower()}_conversation"] = formatted
            
            chain = self.evaluation_prompt | self.llm | self.parser
            return await chain.ainvoke(formatted_convs)
        except Exception as e:
            logger.error(f"Evaluation generation error: {e}")
            return "Interview evaluation could not be generated due to technical issues."

# ========================
# Audio Manager
# ========================

class AudioManager:
    def __init__(self):
        self.groq_client = Groq()
    
    @staticmethod
    def get_random_voice() -> str:
        """Get a random Indian English voice"""
        voices = ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural"]
        return random.choice(voices)
    
    @staticmethod
    def clean_old_audio_files():
        """Remove audio files older than 1 hour"""
        try:
            current_time = time.time()
            for filename in os.listdir(AUDIO_DIR):
                if filename.startswith(("ai_", "temp_")) and filename.endswith((".mp3", ".wav")):
                    filepath = os.path.join(AUDIO_DIR, filename)
                    if os.path.getmtime(filepath) < current_time - 3600:  # 1 hour
                        os.remove(filepath)
        except Exception as e:
            logger.warning(f"Audio cleanup error: {e}")
    
    async def text_to_speech(self, text: str, voice: str, speed: float = 1.3) -> Optional[str]:
        """Convert text to speech and return audio file path"""
        try:
            self.clean_old_audio_files()
            
            timestamp = int(time.time() * 1000)
            raw_path = os.path.join(AUDIO_DIR, f"ai_raw_{timestamp}.mp3")
            final_path = os.path.join(AUDIO_DIR, f"ai_{timestamp}.mp3")
            
            # Generate TTS
            await edge_tts.Communicate(text, voice).save(raw_path)
            
            # Speed adjustment
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-filter:a", f"atempo={speed}",
                "-vn", final_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            os.remove(raw_path)
            return f"./audio/{os.path.basename(final_path)}" if os.path.exists(final_path) else None
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def record_audio(self) -> Optional[str]:
        """Record audio from microphone"""
        logger.info("Starting audio recording...")
        
        chunks = []
        silence_start = None
        recording = True
        start_time = time.time()
        
        def callback(indata, frames, time_info, status):
            nonlocal silence_start, recording
            
            if status:
                logger.error(f"Audio callback error: {status}")
                recording = False
                return
            
            rms = np.sqrt(np.mean(indata**2))
            chunks.append(indata.copy())
            
            # Silence detection
            if rms < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    recording = False
            else:
                silence_start = None
            
            # Maximum duration check
            if time.time() - start_time > MAX_RECORDING_DURATION:
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
        
        if not chunks:
            return None
        
        # Process audio
        audio_data = np.concatenate(chunks)
        duration = len(audio_data) / SAMPLE_RATE
        
        if duration < MIN_RECORDING_DURATION:
            logger.info(f"Recording too short: {duration:.1f}s")
            return None
        
        # Save to file
        filepath = os.path.join(AUDIO_DIR, f"temp_input_{int(time.time())}.wav")
        wavfile.write(filepath, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
        
        logger.info(f"Recording saved: {duration:.1f}s")
        return filepath
    
    def transcribe_audio(self, filepath: str) -> Optional[str]:
        """Transcribe audio using Groq Whisper"""
        try:
            with open(filepath, "rb") as file:
                result = self.groq_client.audio.transcriptions.create(
                    file=(filepath, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json"
                )
            
            transcription = result.text.strip()
            logger.info(f"Transcribed: {transcription[:50]}...")
            return transcription if transcription else None
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

# ========================
# Session Manager
# ========================

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, InterviewSession] = {}
        self.db_manager = DatabaseManager()
        self.llm_manager = LLMManager()
        self.audio_manager = AudioManager()
    
    def create_session(self) -> str:
        """Create a new interview session"""
        session_id = str(uuid.uuid4())
        voice = AudioManager.get_random_voice()
        
        session = InterviewSession(
            session_id=session_id,
            current_round=RoundType.TECHNICAL,
            state=InterviewState.NOT_STARTED,
            voice=voice
        )
        
        # Initialize conversation logs for all rounds
        for round_type in RoundType:
            session.conversations[round_type] = []
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """Get session by ID with activity update"""
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = time.time()
        return session
    
    def should_end_round(self, session: InterviewSession) -> bool:
        """Determine if current round should end based on questions asked"""
        current_round = session.current_round
        questions_asked = session.questions_asked[current_round]
        
        # Dynamic round ending based on question count
        limits = {
            RoundType.TECHNICAL: 6,
            RoundType.COMMUNICATION: 5,
            RoundType.HR: 5
        }
        
        return questions_asked >= limits[current_round]
    
    def get_next_round(self, current_round: RoundType) -> Optional[RoundType]:
        """Get the next round in sequence"""
        sequence = [RoundType.TECHNICAL, RoundType.COMMUNICATION, RoundType.HR]
        try:
            current_idx = sequence.index(current_round)
            return sequence[current_idx + 1] if current_idx < len(sequence) - 1 else None
        except ValueError:
            return None
    
    def format_conversation_history(self, session: InterviewSession, round_type: RoundType) -> str:
        """Format conversation history for LLM context"""
        conversations = session.conversations.get(round_type, [])
        if not conversations:
            return "No previous conversation."
        
        # Only include last 3 exchanges to keep context manageable
        recent_conversations = conversations[-3:]
        
        formatted = []
        for entry in recent_conversations:
            formatted.append(f"AI: {entry.ai_response}")
            formatted.append(f"User: {entry.user_input}")
        
        return "\n".join(formatted)
    
    async def generate_first_question(self, session_id: str) -> Dict[str, Any]:
        """Generate the first question for a round"""
        session = self.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session.state = InterviewState.IN_PROGRESS
        session.round_start_time = time.time()
        
        # Get technical content for technical round
        technical_content = ""
        if session.current_round == RoundType.TECHNICAL:
            technical_content = self.db_manager.get_technical_content()
        
        # Generate question
        history = self.format_conversation_history(session, session.current_round)
        question_count = session.questions_asked[session.current_round] + 1
        
        question = await self.llm_manager.generate_question(
            session.current_round,
            history,
            question_count,
            technical_content
        )
        
        session.questions_asked[session.current_round] += 1
        
        # Generate audio
        audio_path = await self.audio_manager.text_to_speech(question, session.voice)
        
        return {
            "question": question,
            "audio_path": audio_path,
            "round": session.current_round.value
        }
    
    async def process_user_response(self, session_id: str) -> Dict[str, Any]:
        """Process user's audio response and generate AI reply"""
        session = self.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Record and transcribe audio
        audio_file = self.audio_manager.record_audio()
        if not audio_file:
            return {"error": "No audio recorded", "retry": True}
        
        user_text = self.audio_manager.transcribe_audio(audio_file)
        os.remove(audio_file)  # Cleanup
        
        if not user_text:
            return {"error": "Could not understand audio", "retry": True}
        
        logger.info(f"User response: {user_text[:100]}...")
        
        # Check if we should end the round or continue
        if self.should_end_round(session):
            return await self._handle_round_transition(session, user_text)
        
        # Generate next question
        technical_content = ""
        if session.current_round == RoundType.TECHNICAL:
            technical_content = self.db_manager.get_technical_content()
        
        history = self.format_conversation_history(session, session.current_round)
        question_count = session.questions_asked[session.current_round] + 1
        
        ai_response = await self.llm_manager.generate_question(
            session.current_round,
            history,
            question_count,
            technical_content
        )
        
        # Save conversation entry
        entry = ConversationEntry(user_input=user_text, ai_response=ai_response)
        session.conversations[session.current_round].append(entry)
        session.questions_asked[session.current_round] += 1
        
        # Check for round transition keywords in AI response
        transition_keywords = [
            "move to the communication round",
            "proceed to the HR round", 
            "completing the interview"
        ]
        
        is_transition = any(keyword in ai_response.lower() for keyword in transition_keywords)
        
        if is_transition:
            return await self._handle_round_transition(session, user_text, ai_response)
        
        # Generate audio for next question
        audio_path = await self.audio_manager.text_to_speech(ai_response, session.voice)
        
        return {
            "response": ai_response,
            "audio_path": audio_path,
            "continue": True,
            "round": session.current_round.value
        }
    
    async def _handle_round_transition(self, session: InterviewSession, 
                                     user_text: str, ai_response: str = None) -> Dict[str, Any]:
        """Handle transition between rounds or end of interview"""
        current_round = session.current_round
        
        # Save the user's response if we have an AI response
        if ai_response:
            entry = ConversationEntry(user_input=user_text, ai_response=ai_response)
            session.conversations[current_round].append(entry)
        
        # Check if there's a next round
        next_round = self.get_next_round(current_round)
        
        if next_round:
            # Transition to next round
            session.current_round = next_round
            session.state = InterviewState.ROUND_COMPLETE
            
            transition_message = f"Thank you for the {current_round.value.lower()} round. Let's move to the {next_round.value.lower()} round."
            audio_path = await self.audio_manager.text_to_speech(transition_message, session.voice)
            
            return {
                "response": transition_message,
                "audio_path": audio_path,
                "round_complete": True,
                "next_round": next_round.value,
                "current_round": current_round.value
            }
        else:
            # Interview complete
            session.state = InterviewState.INTERVIEW_COMPLETE
            
            completion_message = "Thank you for completing all interview rounds. Generating your evaluation report..."
            audio_path = await self.audio_manager.text_to_speech(completion_message, session.voice)
            
            return {
                "response": completion_message,
                "audio_path": audio_path,
                "interview_complete": True
            }
    
    async def generate_evaluation(self, session_id: str) -> Dict[str, Any]:
        """Generate final interview evaluation"""
        session = self.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        evaluation = await self.llm_manager.generate_evaluation(session.conversations)
        
        # Generate analytics
        total_questions = sum(session.questions_asked.values())
        total_responses = sum(len(convs) for convs in session.conversations.values())
        round_durations = {}
        
        # Calculate approximate round durations (simplified)
        for round_type in RoundType:
            question_count = session.questions_asked[round_type]
            round_durations[round_type.value] = f"{question_count} questions"
        
        return {
            "evaluation": evaluation,
            "analytics": {
                "total_questions": total_questions,
                "total_responses": total_responses,
                "round_breakdown": round_durations
            }
        }
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions (older than 2 hours)"""
        current_time = time.time()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session.last_activity > 7200  # 2 hours
        ]
        
        for sid in expired_sessions:
            self.sessions.pop(sid, None)
            logger.info(f"Cleaned up expired session: {sid}")

# ========================
# FastAPI Application
# ========================

app = FastAPI(title="AI Interview System")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="static")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
# Initialize managers
session_manager = SessionManager()

# ========================
# API Endpoints
# ========================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the interview interface"""
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

@app.get("/start_interview")
async def start_interview():
    """Start a new interview session"""
    try:
        session_id = session_manager.create_session()
        result = await session_manager.generate_first_question(session_id)
        result["session_id"] = session_id
        return result
    except Exception as e:
        logger.error(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record_and_respond")
async def record_and_respond(request: RecordRequest):
    """Record user response and provide AI reply"""
    try:
        return await session_manager.process_user_response(request.session_id)
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/start_next_round")
async def start_next_round(session_id: str):
    """Start the next interview round"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session.state = InterviewState.IN_PROGRESS
        return await session_manager.generate_first_question(session_id)
    except Exception as e:
        logger.error(f"Error starting next round: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate")
async def get_evaluation(session_id: str):
    """Get final interview evaluation"""
    try:
        return await session_manager.generate_evaluation(session_id)
    except Exception as e:
        logger.error(f"Error generating evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cleanup")
async def cleanup():
    """Clean up expired sessions and old files"""
    try:
        session_manager.cleanup_expired_sessions()
        AudioManager.clean_old_audio_files()
        return {"message": "Cleanup completed"}
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup/shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application"""
    logger.info("Interview system starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    session_manager.cleanup_expired_sessions()
    AudioManager.clean_old_audio_files()
    logger.info("Interview system shut down")
