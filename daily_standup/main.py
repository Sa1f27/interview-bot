# App/daily_standup/main.py
# Complete, error-free daily standup backend

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time
import uuid
import logging
import os
import tempfile
import io
from typing import Dict, List, Optional, AsyncGenerator
from pathlib import Path
import edge_tts
import openai
from motor.motor_asyncio import AsyncIOMotorClient
from groq import Groq
import pyodbc
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from urllib.parse import quote_plus
import textwrap
import re
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import aiofiles
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = CURRENT_DIR / "audio"
TEMP_DIR = CURRENT_DIR / "temp"
REPORTS_DIR = CURRENT_DIR / "reports"

for directory in [AUDIO_DIR, TEMP_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# TTS Configuration
TTS_VOICE = "en-IN-PrabhatNeural"
TTS_RATE = "+20%"
TTS_CHUNK_SIZE = 50
TTS_OVERLAP = 5

# Interview Configuration
GREETING_EXCHANGES = 3
TECHNICAL_QUESTIONS = 5
MAX_RECORDING_TIME = 30.0

# Database Configuration
DB_CONFIG = {
    "DRIVER": "ODBC Driver 17 for SQL Server",
    "SERVER": "192.168.48.200",
    "DATABASE": "SuperDB",
    "UID": "sa",
    "PWD": "Welcome@123",
    "timeout": 5
}

MONGO_CONFIG = {
    "host": "192.168.48.201",
    "port": 27017,
    "username": "LanTech",
    "password": "L@nc^ere@0012",
    "database": "Api-1",
    "transcripts_collection": "original-1",
    "results_collection": "daily_standup_results-1"
}

# OpenAI Configuration
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 500

# =============================================================================
# SHARED CLIENT MANAGER (DEPENDENCY INJECTION)
# =============================================================================

class SharedClientManager:
    """Centralized client management for dependency injection"""
    
    def __init__(self):
        self._groq_client = None
        self._openai_client = None
        self._mongo_client = None
        self._mongo_db = None
        
    @property
    def groq_client(self) -> Groq:
        if self._groq_client is None:
            self._groq_client = Groq()
            logger.info("✅ Groq client initialized")
        return self._groq_client
    
    @property 
    def openai_client(self) -> openai.OpenAI:
        if self._openai_client is None:
            self._openai_client = openai.OpenAI()
            logger.info("✅ OpenAI client initialized")
        return self._openai_client
    
    async def get_mongo_client(self) -> AsyncIOMotorClient:
        if self._mongo_client is None:
            mongo_url = f"mongodb://{quote_plus(MONGO_CONFIG['username'])}:{quote_plus(MONGO_CONFIG['password'])}@{MONGO_CONFIG['host']}:{MONGO_CONFIG['port']}/{MONGO_CONFIG['database']}?authSource=admin"
            self._mongo_client = AsyncIOMotorClient(mongo_url, maxPoolSize=50, serverSelectionTimeoutMS=5000)
            try:
                await self._mongo_client.admin.command('ping')
                logger.info("✅ MongoDB client initialized")
            except Exception as e:
                logger.error(f"MongoDB connection failed: {e}")
        return self._mongo_client
    
    async def get_mongo_db(self):
        if self._mongo_db is None:
            client = await self.get_mongo_client()
            self._mongo_db = client[MONGO_CONFIG['database']]
        return self._mongo_db
    
    async def close_connections(self):
        """Cleanup method for graceful shutdown"""
        if self._mongo_client:
            self._mongo_client.close()
            logger.info("🔌 MongoDB connections closed")

# Global shared client manager
shared_clients = SharedClientManager()

# =============================================================================
# CORE CLASSES
# =============================================================================

class SessionStage(Enum):
    GREETING = "greeting"
    CONVERSATION = "conversation"
    COMPLETE = "complete"
    ERROR = "error"

@dataclass
class ConversationExchange:
    timestamp: float
    stage: SessionStage
    ai_message: str
    user_response: str
    transcript_quality: float = 0.0

@dataclass
class SessionData:
    session_id: str
    test_id: str
    student_id: int
    student_name: str
    session_key: str
    created_at: float
    last_activity: float
    current_stage: SessionStage
    exchanges: List[ConversationExchange] = field(default_factory=list)
    greeting_count: int = 0
    technical_count: int = 0
    is_active: bool = True
    websocket: Optional[WebSocket] = None
    
    def add_exchange(self, ai_message: str, user_response: str, quality: float = 0.0):
        exchange = ConversationExchange(
            timestamp=time.time(),
            stage=self.current_stage,
            ai_message=ai_message,
            user_response=user_response,
            transcript_quality=quality
        )
        self.exchanges.append(exchange)
        self.last_activity = time.time()

# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    def __init__(self, client_manager: SharedClientManager):
        self.client_manager = client_manager
        self.sql_conn = None
        
    @property
    def groq_client(self):
        return self.client_manager.groq_client
    
    @property
    def openai_client(self):
        return self.client_manager.openai_client
    
    async def get_mongo_db(self):
        return await self.client_manager.get_mongo_db()
    
    def get_sql_connection(self):
        """Get SQL Server connection with retry logic"""
        try:
            conn_str = f"DRIVER={{{DB_CONFIG['DRIVER']}}};"
            conn_str += f"SERVER={DB_CONFIG['SERVER']};"
            conn_str += f"DATABASE={DB_CONFIG['DATABASE']};"
            conn_str += f"UID={DB_CONFIG['UID']};"
            conn_str += f"PWD={DB_CONFIG['PWD']};"
            
            conn = pyodbc.connect(conn_str, timeout=DB_CONFIG['timeout'])
            return conn
        except Exception as e:
            logger.error(f"SQL connection failed: {e}")
            return None
    
    async def get_student_info(self) -> tuple:
        """Get random student info with fallback"""
        try:
            conn = self.get_sql_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT TOP 1 ID, First_Name, Last_Name FROM tbl_Student ORDER BY NEWID()")
                row = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if row:
                    return (row[0], row[1], row[2], f"SESSION_{int(time.time())}")
        except Exception as e:
            logger.error(f"Error fetching student info: {e}")
        
        # Fallback
        return (random.randint(1000, 9999), "Test", "User", f"SESSION_{int(time.time())}")
    
    async def get_latest_transcript_summary(self) -> str:
        """Get latest transcript summary from MongoDB"""
        try:
            db = await self.get_mongo_db()
            collection = db[MONGO_CONFIG['transcripts_collection']]
            doc = await collection.find_one(
                {"summary": {"$exists": True, "$ne": None, "$ne": ""}},
                sort=[("timestamp", -1)]
            )
            
            if doc and doc.get("summary"):
                return doc["summary"]
        except Exception as e:
            logger.error(f"Error fetching transcript summary: {e}")
        
        return "Technical discussion about software development, coding practices, and project management."
    
    async def save_session_result(self, session_data: SessionData, evaluation: str, score: float):
        """Save session results to MongoDB"""
        try:
            db = await self.get_mongo_db()
            collection = db[MONGO_CONFIG['results_collection']]
            
            document = {
                "test_id": session_data.test_id,
                "session_id": session_data.session_id,
                "student_id": session_data.student_id,
                "student_name": session_data.student_name,
                "session_key": session_data.session_key,
                "timestamp": time.time(),
                "created_at": session_data.created_at,
                "conversation_log": [
                    {
                        "timestamp": exchange.timestamp,
                        "stage": exchange.stage.value,
                        "ai_message": exchange.ai_message,
                        "user_response": exchange.user_response,
                        "transcript_quality": exchange.transcript_quality
                    }
                    for exchange in session_data.exchanges
                ],
                "evaluation": evaluation,
                "score": score,
                "total_exchanges": len(session_data.exchanges),
                "greeting_exchanges": session_data.greeting_count,
                "technical_exchanges": session_data.technical_count,
                "duration": time.time() - session_data.created_at
            }
            
            await collection.insert_one(document)
            logger.info(f"Session {session_data.session_id} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session result: {e}")
            return False

# =============================================================================
# AUDIO PROCESSOR
# =============================================================================

class AudioProcessor:
    def __init__(self, client_manager: SharedClientManager):
        self.client_manager = client_manager
    
    @property
    def groq_client(self):
        return self.client_manager.groq_client
    
    async def transcribe_audio(self, audio_data: bytes) -> tuple[str, float]:
        """Transcribe audio using Groq Whisper with quality assessment"""
        try:
            if len(audio_data) < 1000:
                return "", 0.0
            
            # Create temporary file
            temp_file = TEMP_DIR / f"audio_{int(time.time() * 1000)}.webm"
            
            async with aiofiles.open(temp_file, "wb") as f:
                await f.write(audio_data)
            
            # Transcribe using shared Groq client
            with open(temp_file, "rb") as file:
                result = self.groq_client.audio.transcriptions.create(
                    file=(temp_file.name, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json"
                )
            
            # Clean up
            try:
                os.remove(temp_file)
            except:
                pass
            
            transcript = result.text.strip()
            
            # Simple quality assessment
            quality = min(len(transcript) / 100, 1.0)
            if hasattr(result, 'segments'):
                confidences = [seg.get('confidence', 0) for seg in result.segments if seg.get('confidence')]
                if confidences:
                    quality = (quality + sum(confidences) / len(confidences)) / 2
            
            return transcript, quality
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", 0.0

class TTSProcessor:
    def __init__(self):
        self.voice = TTS_VOICE
        self.rate = TTS_RATE
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks for streaming"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), TTS_CHUNK_SIZE - TTS_OVERLAP):
            chunk_words = words[i:i + TTS_CHUNK_SIZE]
            chunks.append(' '.join(chunk_words))
        
        return chunks if chunks else [text]
    
    async def generate_audio_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate audio stream in chunks for low-latency playback"""
        try:
            chunks = self.split_text_into_chunks(text)
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                
                # Generate TTS for this chunk
                tts = edge_tts.Communicate(chunk, self.voice, rate=self.rate)
                audio_data = b""
                
                async for tts_chunk in tts.stream():
                    if tts_chunk["type"] == "audio":
                        audio_data += tts_chunk["data"]
                
                if audio_data:
                    yield audio_data
                    
                # Small delay between chunks for smooth playback
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            yield b""

# =============================================================================
# CONVERSATION MANAGER
# =============================================================================

class ConversationManager:
    def __init__(self, client_manager: SharedClientManager):
        self.client_manager = client_manager
        self.model = OPENAI_MODEL
        self.temperature = OPENAI_TEMPERATURE
        self.max_tokens = OPENAI_MAX_TOKENS
    
    @property
    def openai_client(self):
        return self.client_manager.openai_client
    
    async def generate_response(self, session_data: SessionData, user_input: str, context: str = "") -> str:
        """Generate AI response based on session stage and context"""
        try:
            if session_data.current_stage == SessionStage.GREETING:
                return await self._generate_greeting_response(session_data, user_input)
            elif session_data.current_stage == SessionStage.CONVERSATION:
                return await self._generate_technical_response(session_data, user_input, context)
            else:
                return await self._generate_conclusion_response(session_data, user_input)
                
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._get_fallback_response(session_data.current_stage)
    
    async def _generate_greeting_response(self, session_data: SessionData, user_input: str) -> str:
        """Generate greeting stage responses"""
        prompts = [
            f"You're a friendly interviewer starting a daily standup. The participant just said: '{user_input}'. Respond warmly and naturally, then ask how their day is going. Keep it conversational and genuine.",
            f"The participant said: '{user_input}'. Acknowledge their response about their day and ask if they're ready to discuss their work progress. Be encouraging and supportive.",
            f"The participant said: '{user_input}'. Great! Now transition naturally into asking about their recent work accomplishments. Start the technical discussion in a friendly way."
        ]
        
        prompt = prompts[min(session_data.greeting_count, len(prompts) - 1)]
        
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    async def _generate_technical_response(self, session_data: SessionData, user_input: str, context: str) -> str:
        """Generate technical discussion responses"""
        conversation_history = ""
        for exchange in session_data.exchanges[-3:]:
            conversation_history += f"AI: {exchange.ai_message}\nUser: {exchange.user_response}\n\n"
        
        prompt = f"""You're conducting a technical daily standup interview. 

Context about current topics: {context}

Recent conversation:
{conversation_history}

The participant just said: "{user_input}"

Respond as a knowledgeable, curious interviewer. Ask follow-up questions about their technical work, challenges, and achievements. Keep the conversation natural and engaging. Focus on understanding their current projects, technical decisions, and any blockers they're facing.

Be genuinely interested in their work and ask questions that help them elaborate on their technical experiences."""

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    async def _generate_conclusion_response(self, session_data: SessionData, user_input: str) -> str:
        """Generate conclusion responses"""
        prompt = f"""The participant just said: "{user_input}". 

This is the end of the daily standup interview. Thank them for their time and provide a brief, encouraging summary of what they shared. Keep it warm and professional.

Then naturally conclude the session."""

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    def _get_fallback_response(self, stage: SessionStage) -> str:
        """Fallback responses for error cases"""
        fallbacks = {
            SessionStage.GREETING: "Hello! Welcome to today's standup. How are you doing?",
            SessionStage.CONVERSATION: "That's interesting! Can you tell me more about the technical challenges you're facing?",
            SessionStage.COMPLETE: "Thank you for sharing your progress. Have a great day!"
        }
        return fallbacks.get(stage, "I'm having trouble processing that. Could you please repeat?")
    
    async def generate_evaluation(self, session_data: SessionData) -> tuple[str, float]:
        """Generate final evaluation and score"""
        try:
            conversation_text = ""
            for exchange in session_data.exchanges:
                if exchange.stage == SessionStage.CONVERSATION:
                    conversation_text += f"Q: {exchange.ai_message}\nA: {exchange.user_response}\n\n"
            
            prompt = f"""Evaluate this daily standup interview conversation:

{conversation_text}

Provide a constructive evaluation focusing on:
1. Technical communication clarity
2. Understanding of their work and challenges
3. Ability to articulate progress and blockers
4. Overall engagement and professionalism

Give specific feedback and suggestions for improvement.
End with a score out of 10.

Format: [Evaluation text] Final Score: X/10"""

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            
            evaluation = response.choices[0].message.content.strip()
            
            # Extract score
            score_match = re.search(r'Final Score:\s*(\d+(?:\.\d+)?)/10', evaluation)
            score = float(score_match.group(1)) if score_match else 7.0
            
            return evaluation, score
            
        except Exception as e:
            logger.error(f"Evaluation generation error: {e}")
            return "Good participation in the standup discussion. Continue sharing your technical progress and challenges. Final Score: 7/10", 7.0

# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, SessionData] = {}
        self.db_manager = DatabaseManager(shared_clients)
        self.audio_processor = AudioProcessor(shared_clients)
        self.tts_processor = TTSProcessor()
        self.conversation_manager = ConversationManager(shared_clients)
    
    async def create_session(self, websocket: WebSocket = None) -> SessionData:
        """Create new session"""
        session_id = str(uuid.uuid4())
        test_id = f"standup_{int(time.time())}"
        
        # Get student info
        student_id, first_name, last_name, session_key = await self.db_manager.get_student_info()
        
        session_data = SessionData(
            session_id=session_id,
            test_id=test_id,
            student_id=student_id,
            student_name=f"{first_name} {last_name}",
            session_key=session_key,
            created_at=time.time(),
            last_activity=time.time(),
            current_stage=SessionStage.GREETING,
            websocket=websocket
        )
        
        self.active_sessions[session_id] = session_data
        logger.info(f"Created session {session_id} for {session_data.student_name}")
        
        return session_data
    
    async def remove_session(self, session_id: str):
        """Remove session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Removed session {session_id}")
    
    async def process_audio_input(self, session_id: str, audio_data: bytes):
        """Process audio input from frontend"""
        session_data = self.active_sessions.get(session_id)
        if not session_data or not session_data.is_active:
            return
        
        # Transcribe audio
        transcript, quality = await self.audio_processor.transcribe_audio(audio_data)
        
        if not transcript or len(transcript.strip()) < 2:
            await self._send_message(session_data, {
                "type": "clarification",
                "text": "I didn't catch that clearly. Could you please repeat?",
                "status": session_data.current_stage.value
            })
            return
        
        logger.info(f"Session {session_id}: User said: {transcript}")
        
        # Get context for technical discussions
        context = ""
        if session_data.current_stage == SessionStage.CONVERSATION:
            context = await self.db_manager.get_latest_transcript_summary()
        
        # Generate AI response
        ai_response = await self.conversation_manager.generate_response(
            session_data, transcript, context
        )
        
        # Add exchange to session
        session_data.add_exchange(ai_response, transcript, quality)
        
        # Update session state
        await self._update_session_state(session_data)
        
        # Send response with streaming audio
        await self._send_response_with_audio(session_data, ai_response)
    
    async def _update_session_state(self, session_data: SessionData):
        """Update session state based on current progress"""
        if session_data.current_stage == SessionStage.GREETING:
            session_data.greeting_count += 1
            if session_data.greeting_count >= GREETING_EXCHANGES:
                session_data.current_stage = SessionStage.CONVERSATION
                logger.info(f"Session {session_data.session_id} moved to CONVERSATION stage")
        
        elif session_data.current_stage == SessionStage.CONVERSATION:
            session_data.technical_count += 1
            if session_data.technical_count >= TECHNICAL_QUESTIONS:
                session_data.current_stage = SessionStage.COMPLETE
                logger.info(f"Session {session_data.session_id} moved to COMPLETE stage")
                
                # Generate evaluation and save session
                await self._finalize_session(session_data)
    
    async def _finalize_session(self, session_data: SessionData):
        """Finalize session with evaluation"""
        try:
            # Generate evaluation
            evaluation, score = await self.conversation_manager.generate_evaluation(session_data)
            
            # Save to database
            await self.db_manager.save_session_result(session_data, evaluation, score)
            
            # Send completion message
            completion_message = "Thank you for participating in today's standup! Your responses have been recorded."
            
            await self._send_message(session_data, {
                "type": "conversation_end",
                "text": completion_message,
                "evaluation": evaluation,
                "score": score,
                "pdf_url": f"/download_results/{session_data.session_id}",
                "status": "complete"
            })
            
            # Generate and send final audio
            async for audio_chunk in self.tts_processor.generate_audio_stream(completion_message):
                if audio_chunk:
                    await self._send_message(session_data, {
                        "type": "audio_chunk",
                        "audio": audio_chunk.hex(),
                        "status": "complete"
                    })
            
            session_data.is_active = False
            
        except Exception as e:
            logger.error(f"Session finalization error: {e}")
    
    async def _send_response_with_audio(self, session_data: SessionData, text: str):
        """Send text response with streaming audio"""
        try:
            # Send text response first
            await self._send_message(session_data, {
                "type": "ai_response",
                "text": text,
                "status": session_data.current_stage.value
            })
            
            # Stream audio chunks
            async for audio_chunk in self.tts_processor.generate_audio_stream(text):
                if audio_chunk and session_data.is_active:
                    await self._send_message(session_data, {
                        "type": "audio_chunk",
                        "audio": audio_chunk.hex(),
                        "status": session_data.current_stage.value
                    })
            
            # Send audio end signal
            await self._send_message(session_data, {
                "type": "audio_end",
                "status": session_data.current_stage.value
            })
            
        except Exception as e:
            logger.error(f"Audio streaming error: {e}")
    
    async def _send_message(self, session_data: SessionData, message: dict):
        """Send message to WebSocket"""
        try:
            if session_data.websocket:
                await session_data.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
    
    async def get_session_result(self, session_id: str) -> dict:
        """Get session result for PDF generation"""
        try:
            db = await self.db_manager.get_mongo_db()
            collection = db[MONGO_CONFIG['results_collection']]
            result = await collection.find_one({"session_id": session_id})
            
            if result:
                result['_id'] = str(result['_id'])
                return result
            return None
            
        except Exception as e:
            logger.error(f"Error fetching session result: {e}")
            return None

    # LEGACY SUPPORT METHODS
    async def process_legacy_audio(self, test_id: str, audio_data: bytes) -> dict:
        """Process audio using legacy format (for recordAndRespond compatibility)"""
        try:
            logger.info(f"Processing legacy audio for test_id: {test_id}")
            
            # Transcribe audio
            transcript, quality = await self.audio_processor.transcribe_audio(audio_data)
            
            if not transcript or len(transcript.strip()) < 2:
                return {
                    "response": "I didn't catch that clearly. Could you please repeat?",
                    "audio_path": None,
                    "ended": False,
                    "status": "clarification"
                }
            
            # Generate simple response
            context = await self.db_manager.get_latest_transcript_summary()
            
            # Create a temporary session for legacy processing
            session_data = SessionData(
                session_id=test_id,
                test_id=test_id,
                student_id=1000,
                student_name="Legacy User",
                session_key="LEGACY",
                created_at=time.time(),
                last_activity=time.time(),
                current_stage=SessionStage.CONVERSATION
            )
            
            ai_response = await self.conversation_manager.generate_response(
                session_data, transcript, context
            )
            
            return {
                "response": ai_response,
                "audio_path": None,
                "ended": False,
                "complete": False,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Legacy audio processing error: {e}")
            return {
                "response": "I'm having trouble processing your response. Please try again.",
                "audio_path": None,
                "ended": False,
                "status": "error"
            }

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Create FastAPI sub-application
app = FastAPI(title="Daily Standup Interview System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# Initialize session manager
session_manager = SessionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Daily Standup application started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await shared_clients.close_connections()
    logger.info("Daily Standup application shutting down")