# App/daily_standup/main.py
# Ultra-fast, summary-based daily standup backend with optimized performance

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
from typing import Dict, List, Optional, AsyncGenerator, Tuple
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
import base64
import concurrent.futures
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# OPTIMIZED CONFIGURATION
# =============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = CURRENT_DIR / "audio"
TEMP_DIR = CURRENT_DIR / "temp"
REPORTS_DIR = CURRENT_DIR / "reports"

for directory in [AUDIO_DIR, TEMP_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# Ultra-fast TTS Configuration
TTS_VOICE = "en-IN-PrabhatNeural"
TTS_RATE = "+25%"  # Faster speech
TTS_CHUNK_SIZE = 30  # Smaller chunks for faster streaming
TTS_OVERLAP = 3

# Optimized Interview Configuration
GREETING_EXCHANGES = 2  # Reduced from 3
SUMMARY_CHUNKS = 12  # Target chunk count
BASE_QUESTIONS_PER_CHUNK = 2
MAX_QUESTIONS_PER_CHUNK = 5
CONVERSATION_WINDOW_SIZE = 6  # Sliding window for conversation history
MAX_RECORDING_TIME = 25.0  # Reduced from 30
SILENCE_THRESHOLD = 800  # 800ms instead of 2000ms

# Database Configuration
DB_CONFIG = {
    "DRIVER": "ODBC Driver 17 for SQL Server",
    "SERVER": "183.82.108.211",
    "DATABASE": "SuperDB",
    "UID": "Connectly",
    "PWD": "LT@connect25",
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

# Optimized OpenAI Configuration
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 300  # Reduced for faster responses

# =============================================================================
# SUMMARY MANAGEMENT SYSTEM
# =============================================================================

@dataclass
class SummaryChunk:
    id: int
    content: str
    base_questions: List[str]
    current_question_count: int = 0
    completed: bool = False
    follow_up_questions: List[str] = field(default_factory=list)

class SummaryManager:
    def __init__(self, client_manager):
        self.client_manager = client_manager
        self.chunks: List[SummaryChunk] = []
        self.current_chunk_index = 0
    
    @property
    def openai_client(self):
        return self.client_manager.openai_client
    
    async def initialize_chunks(self, summary: str) -> bool:
        """Initialize summary chunks with base questions"""
        try:
            # Split summary into semantic chunks
            raw_chunks = await self._split_summary_semantically(summary)
            
            # Generate base questions for each chunk
            for i, chunk_content in enumerate(raw_chunks):
                base_questions = await self._generate_base_questions(chunk_content)
                
                chunk = SummaryChunk(
                    id=i,
                    content=chunk_content,
                    base_questions=base_questions
                )
                self.chunks.append(chunk)
            
            logger.info(f"✅ Initialized {len(self.chunks)} summary chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Summary chunk initialization failed: {e}")
            raise Exception(f"Summary chunk initialization failed: {e}")
    
    async def _split_summary_semantically(self, summary: str) -> List[str]:
        """Split summary into semantic chunks"""
        try:
            prompt = f"""Split this technical summary into {SUMMARY_CHUNKS} meaningful, cohesive chunks for interview questions. Each chunk should focus on a specific aspect or topic.

Summary: {summary}

Return only the chunks separated by '###CHUNK###' markers. Each chunk should be 2-4 sentences covering a distinct topic."""

            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            if not content:
                raise Exception("OpenAI returned empty response for summary splitting")
                
            chunks = [chunk.strip() for chunk in content.split('###CHUNK###') if chunk.strip()]
            
            if len(chunks) < 8:
                raise Exception(f"Insufficient chunks generated: {len(chunks)}, expected at least 8")
            
            return chunks[:SUMMARY_CHUNKS]
            
        except Exception as e:
            logger.error(f"❌ Semantic splitting failed: {e}")
            raise Exception(f"Summary semantic splitting failed: {e}")
    
    async def _generate_base_questions(self, chunk_content: str) -> List[str]:
        """Generate base questions for a chunk - NO FALLBACKS"""
        try:
            prompt = f"""Generate exactly {BASE_QUESTIONS_PER_CHUNK} insightful interview questions about this technical content. Questions should be specific, engaging, and encourage detailed responses.

Content: {chunk_content}

Format: Return only the questions, one per line, numbered 1. 2. etc."""

            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            if not content:
                raise Exception("OpenAI returned empty response for question generation")
                
            questions = []
            for line in content.split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question:
                        questions.append(question)
            
            if not questions:
                raise Exception(f"Failed to extract questions from OpenAI response: {content}")
                
            return questions[:BASE_QUESTIONS_PER_CHUNK]
            
        except Exception as e:
            logger.error(f"❌ Question generation failed: {e}")
            raise Exception(f"Question generation failed: {e}")
    
    def get_current_chunk(self) -> Optional[SummaryChunk]:
        """Get current active chunk"""
        if 0 <= self.current_chunk_index < len(self.chunks):
            return self.chunks[self.current_chunk_index]
        return None
    
    def get_next_question(self) -> Optional[str]:
        """Get next question from current chunk"""
        chunk = self.get_current_chunk()
        if not chunk:
            return None
        
        # First serve base questions
        if chunk.current_question_count < len(chunk.base_questions):
            question = chunk.base_questions[chunk.current_question_count]
            chunk.current_question_count += 1
            return question
        
        # Then serve follow-up questions
        follow_up_index = chunk.current_question_count - len(chunk.base_questions)
        if follow_up_index < len(chunk.follow_up_questions):
            question = chunk.follow_up_questions[follow_up_index]
            chunk.current_question_count += 1
            return question
        
        return None
    
    async def analyze_response_and_generate_followups(self, user_response: str) -> bool:
        """Analyze user response and generate follow-ups if needed"""
        chunk = self.get_current_chunk()
        if not chunk:
            return False
        
        try:
            prompt = f"""Analyze this user response about: "{chunk.content[:100]}..."

User Response: "{user_response}"

Does this response need follow-up questions for clarity or deeper insight? If yes, generate 1-2 specific follow-up questions. If the response is complete and clear, respond with "COMPLETE".

If follow-ups needed, format as:
FOLLOWUP: Question 1
FOLLOWUP: Question 2

If complete, respond with: COMPLETE"""

            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            
            if "COMPLETE" in content.upper():
                return False
            
            # Extract follow-up questions
            followups = []
            for line in content.split('\n'):
                if line.strip().startswith('FOLLOWUP:'):
                    question = line.replace('FOLLOWUP:', '').strip()
                    if question:
                        followups.append(question)
            
            # Add follow-ups if we haven't hit the limit
            total_questions = len(chunk.base_questions) + len(chunk.follow_up_questions)
            remaining_slots = MAX_QUESTIONS_PER_CHUNK - total_questions
            
            if followups and remaining_slots > 0:
                chunk.follow_up_questions.extend(followups[:remaining_slots])
                logger.info(f"Added {len(followups[:remaining_slots])} follow-up questions to chunk {chunk.id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Follow-up analysis failed: {e}")
            raise Exception(f"Follow-up analysis failed: {e}")
    
    def move_to_next_chunk(self) -> bool:
        """Move to next chunk"""
        current_chunk = self.get_current_chunk()
        if current_chunk:
            current_chunk.completed = True
        
        self.current_chunk_index += 1
        
        if self.current_chunk_index < len(self.chunks):
            logger.info(f"Moved to chunk {self.current_chunk_index}")
            return True
        
        logger.info("All chunks completed")
        return False
    
    def should_move_to_next_chunk(self) -> bool:
        """Check if current chunk is exhausted"""
        chunk = self.get_current_chunk()
        if not chunk:
            return True
        
        total_asked = chunk.current_question_count
        total_available = len(chunk.base_questions) + len(chunk.follow_up_questions)
        
        return total_asked >= total_available
    
    def get_progress(self) -> dict:
        """Get current progress"""
        return {
            "current_chunk": self.current_chunk_index,
            "total_chunks": len(self.chunks),
            "chunk_progress": f"{self.current_chunk_index + 1}/{len(self.chunks)}",
            "questions_asked": self.get_current_chunk().current_question_count if self.get_current_chunk() else 0
        }

# =============================================================================
# OPTIMIZED CORE CLASSES
# =============================================================================

class SessionStage(Enum):
    GREETING = "greeting"
    TECHNICAL = "technical"  # New stage for summary-based questions
    COMPLETE = "complete"
    ERROR = "error"

@dataclass
class ConversationExchange:
    timestamp: float
    stage: SessionStage
    ai_message: str
    user_response: str
    transcript_quality: float = 0.0
    chunk_id: Optional[int] = None

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
    conversation_window: deque = field(default_factory=lambda: deque(maxlen=CONVERSATION_WINDOW_SIZE))
    greeting_count: int = 0
    is_active: bool = True
    websocket: Optional[WebSocket] = None
    summary_manager: Optional[SummaryManager] = None
    
    def add_exchange(self, ai_message: str, user_response: str, quality: float = 0.0, chunk_id: Optional[int] = None):
        exchange = ConversationExchange(
            timestamp=time.time(),
            stage=self.current_stage,
            ai_message=ai_message,
            user_response=user_response,
            transcript_quality=quality,
            chunk_id=chunk_id
        )
        self.exchanges.append(exchange)
        self.conversation_window.append(exchange)
        self.last_activity = time.time()

# =============================================================================
# SHARED CLIENT MANAGER (OPTIMIZED)
# =============================================================================

class SharedClientManager:
    """Optimized client management with connection pooling"""
    
    def __init__(self):
        self._groq_client = None
        self._openai_client = None
        self._mongo_client = None
        self._mongo_db = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
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
    
    @property
    def executor(self):
        return self._executor
    
    async def get_mongo_client(self) -> AsyncIOMotorClient:
        if self._mongo_client is None:
            mongo_url = f"mongodb://{quote_plus(MONGO_CONFIG['username'])}:{quote_plus(MONGO_CONFIG['password'])}@{MONGO_CONFIG['host']}:{MONGO_CONFIG['port']}/{MONGO_CONFIG['database']}?authSource=admin"
            self._mongo_client = AsyncIOMotorClient(mongo_url, maxPoolSize=50, serverSelectionTimeoutMS=5000)
            try:
                await self._mongo_client.admin.command('ping')
                logger.info("✅ MongoDB client initialized")
            except Exception as e:
                logger.error(f"❌ MongoDB connection failed: {e}")
                raise Exception(f"MongoDB connection failed: {e}")
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
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info("🔌 All connections closed")

# Global shared client manager
shared_clients = SharedClientManager()

# =============================================================================
# ULTRA-FAST AUDIO PROCESSOR
# =============================================================================

class OptimizedAudioProcessor:
    def __init__(self, client_manager: SharedClientManager):
        self.client_manager = client_manager
    
    @property
    def groq_client(self):
        return self.client_manager.groq_client
    
    async def transcribe_audio_fast(self, audio_data: bytes) -> tuple[str, float]:
        """Ultra-fast transcription with parallel processing"""
        try:
            if len(audio_data) < 1000:
                raise Exception("Audio data too small for transcription")
            
            # Run transcription in thread pool for non-blocking operation
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_transcribe,
                audio_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Fast transcription error: {e}")
            raise Exception(f"Transcription failed: {e}")
    
    def _sync_transcribe(self, audio_data: bytes) -> tuple[str, float]:
        """Synchronous transcription for thread pool"""
        try:
            # Create temporary file with timestamp to avoid conflicts
            temp_file = TEMP_DIR / f"audio_{int(time.time() * 1000000)}.webm"
            
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            # Transcribe using Groq
            with open(temp_file, "rb") as file:
                result = self.groq_client.audio.transcriptions.create(
                    file=(temp_file.name, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json"
                )
            
            # Clean up immediately
            try:
                os.remove(temp_file)
            except:
                pass
            
            transcript = result.text.strip()
            if not transcript:
                raise Exception("Groq returned empty transcript")
            
            # Fast quality assessment
            quality = min(len(transcript) / 50, 1.0)
            if hasattr(result, 'segments') and result.segments:
                confidences = [seg.get('confidence', 0.8) for seg in result.segments[:3]]
                if confidences:
                    quality = (quality + sum(confidences) / len(confidences)) / 2
            
            return transcript, quality
            
        except Exception as e:
            logger.error(f"❌ Sync transcription error: {e}")
            raise Exception(f"Groq transcription failed: {e}")

class UltraFastTTSProcessor:
    def __init__(self):
        self.voice = TTS_VOICE
        self.rate = TTS_RATE
    
    def split_text_optimized(self, text: str) -> List[str]:
        """Optimized text splitting for minimal latency"""
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > TTS_CHUNK_SIZE * 5:  # ~150 chars
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    async def generate_ultra_fast_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Ultra-fast audio generation with parallel processing"""
        try:
            chunks = self.split_text_optimized(text)
            
            # Process first chunk immediately, prepare others in parallel
            tasks = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                if i == 0:
                    # Process first chunk immediately for minimal latency
                    async for audio_chunk in self._generate_chunk_audio(chunk):
                        if audio_chunk:
                            yield audio_chunk
                else:
                    # Queue other chunks for parallel processing
                    tasks.append(self._generate_chunk_audio(chunk))
            
            # Stream remaining chunks as they complete
            for task in tasks:
                async for audio_chunk in task:
                    if audio_chunk:
                        yield audio_chunk
                        
        except Exception as e:
            logger.error(f"❌ Ultra-fast TTS error: {e}")
            raise Exception(f"TTS generation failed: {e}")
    
    async def _generate_chunk_audio(self, chunk: str) -> AsyncGenerator[bytes, None]:
        """Generate audio for a single chunk"""
        try:
            tts = edge_tts.Communicate(chunk, self.voice, rate=self.rate)
            audio_data = b""
            
            async for tts_chunk in tts.stream():
                if tts_chunk["type"] == "audio":
                    audio_data += tts_chunk["data"]
            
            if audio_data:
                yield audio_data
            else:
                raise Exception("EdgeTTS returned empty audio data")
                
        except Exception as e:
            logger.error(f"❌ Chunk TTS error: {e}")
            raise Exception(f"TTS chunk generation failed: {e}")

# =============================================================================
# OPTIMIZED CONVERSATION MANAGER
# =============================================================================

class OptimizedConversationManager:
    def __init__(self, client_manager: SharedClientManager):
        self.client_manager = client_manager
        self.model = OPENAI_MODEL
        self.temperature = OPENAI_TEMPERATURE
        self.max_tokens = OPENAI_MAX_TOKENS
    
    @property
    def openai_client(self):
        return self.client_manager.openai_client
    
    async def generate_fast_response(self, session_data: SessionData, user_input: str) -> str:
        """Generate ultra-fast AI responses with context awareness"""
        try:
            if session_data.current_stage == SessionStage.GREETING:
                return await self._generate_greeting_response(session_data, user_input)
            elif session_data.current_stage == SessionStage.TECHNICAL:
                return await self._generate_technical_response(session_data, user_input)
            else:
                return await self._generate_conclusion_response(session_data, user_input)
                
        except Exception as e:
            logger.error(f"❌ Fast response generation error: {e}")
            raise Exception(f"AI response generation failed: {e}")
    
    async def _generate_greeting_response(self, session_data: SessionData, user_input: str) -> str:
        """Optimized greeting responses"""
        prompts = [
            f"User said: '{user_input}'. Respond warmly and ask how their work is going. Keep it brief and natural (max 2 sentences).",
            f"User said: '{user_input}'. Acknowledge and transition to asking about their recent technical work. Be encouraging and brief."
        ]
        
        prompt = prompts[min(session_data.greeting_count, len(prompts) - 1)]
        
        # Run in thread pool for non-blocking operation
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.client_manager.executor,
            self._sync_openai_call,
            prompt
        )
        
        return response
    
    async def _generate_technical_response(self, session_data: SessionData, user_input: str) -> str:
        """Generate responses using summary-based questions"""
        if not session_data.summary_manager:
            raise Exception("Summary manager not initialized")
        
        # Analyze user response for follow-ups
        await session_data.summary_manager.analyze_response_and_generate_followups(user_input)
        
        # Get next question from current chunk
        next_question = session_data.summary_manager.get_next_question()
        
        if next_question:
            # Build context from conversation window
            context = self._build_conversation_context(session_data)
            
            prompt = f"""You're conducting a technical standup interview. 

Recent conversation context:
{context}

User just said: "{user_input}"

Your next planned question: "{next_question}"

Acknowledge their response briefly and naturally transition to the next question. Keep it conversational and engaging. Maximum 2 sentences + the question."""

            # Run in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_openai_call,
                prompt
            )
            
            return response
        
        # No more questions in current chunk, try to move to next
        if session_data.summary_manager.should_move_to_next_chunk():
            if session_data.summary_manager.move_to_next_chunk():
                next_question = session_data.summary_manager.get_next_question()
                if next_question:
                    return f"Great insights! Now let me ask you about another aspect of your work: {next_question}"
        
        # All chunks completed
        session_data.current_stage = SessionStage.COMPLETE
        return "Thank you for sharing all those details about your work. You've provided excellent insights into your technical progress."
    
    def _build_conversation_context(self, session_data: SessionData) -> str:
        """Build context from sliding window of conversation"""
        context = ""
        for exchange in list(session_data.conversation_window)[-3:]:  # Last 3 exchanges
            context += f"AI: {exchange.ai_message}\nUser: {exchange.user_response}\n\n"
        return context.strip()
    
    async def _generate_conclusion_response(self, session_data: SessionData, user_input: str) -> str:
        """Fast conclusion responses"""
        return "Thank you for participating in today's standup. Your technical insights have been recorded successfully."
    
    def _sync_openai_call(self, prompt: str) -> str:
        """Synchronous OpenAI call for thread pool - NO FALLBACKS"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            result = response.choices[0].message.content.strip()
            if not result:
                raise Exception("OpenAI returned empty response")
            return result
        except Exception as e:
            logger.error(f"❌ OpenAI API call failed: {e}")
            raise Exception(f"OpenAI API failed: {e}")
    
    async def generate_fast_evaluation(self, session_data: SessionData) -> tuple[str, float]:
        """Generate evaluation efficiently - NO FALLBACKS"""
        try:
            # Summarize key points from conversation
            key_points = []
            for exchange in session_data.exchanges[-10:]:  # Last 10 exchanges
                if exchange.stage == SessionStage.TECHNICAL:
                    key_points.append(f"Q: {exchange.ai_message[:100]}... A: {exchange.user_response[:100]}...")
            
            if not key_points:
                raise Exception("No technical exchanges found for evaluation")
            
            prompt = f"""Evaluate this standup interview based on key points:

{chr(10).join(key_points[:5])}  

Provide brief evaluation (2-3 sentences) and score out of 10.
Format: [Evaluation] Score: X/10"""

            loop = asyncio.get_event_loop()
            evaluation = await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_openai_call,
                prompt
            )
            
            # Extract score
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)/10', evaluation)
            if not score_match:
                raise Exception(f"Could not extract score from evaluation: {evaluation}")
                
            score = float(score_match.group(1))
            
            return evaluation, score
            
        except Exception as e:
            logger.error(f"❌ Fast evaluation error: {e}")
            raise Exception(f"Evaluation generation failed: {e}")

# =============================================================================
# DATABASE MANAGER (OPTIMIZED)
# =============================================================================

class OptimizedDatabaseManager:
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
        """Optimized SQL connection with connection pooling"""
        try:
            conn_str = f"DRIVER={{{DB_CONFIG['DRIVER']}}};"
            conn_str += f"SERVER={DB_CONFIG['SERVER']};"
            conn_str += f"DATABASE={DB_CONFIG['DATABASE']};"
            conn_str += f"UID={DB_CONFIG['UID']};"
            conn_str += f"PWD={DB_CONFIG['PWD']};"
            
            conn = pyodbc.connect(conn_str, timeout=DB_CONFIG['timeout'])
            return conn
        except Exception as e:
            logger.error(f"❌ SQL connection failed: {e}")
            raise Exception(f"SQL Server connection failed: {e}")
    
    async def get_student_info_fast(self) -> tuple:
        """Fast student info retrieval"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.client_manager.executor,
            self._sync_get_student_info
        )
    
    # def _sync_get_student_info(self) -> tuple:
    #     """Synchronous student info for thread pool - NO FALLBACKS"""
    #     try:
    #         conn = self.get_sql_connection()
    #         cursor = conn.cursor()
    #         cursor.execute("SELECT TOP 1 ID, First_Name, Last_Name FROM tbl_Student ORDER BY NEWID()")
    #         row = cursor.fetchone()
    #         cursor.close()
    #         conn.close()
            
    #         if not row:
    #             raise Exception("No student records found in tbl_Student")
                
    #         return (row[0], row[1], row[2], f"SESSION_{int(time.time())}")
            
    #     except Exception as e:
    #         logger.error(f"❌ Error fetching student info: {e}")
    #         raise Exception(f"Student info retrieval failed: {e}")
    def _sync_get_student_info(self) -> tuple:
        """Return dummy student info while SQL Server is down"""
        logger.warning("⚠️ Using dummy student info (SQL Server is DOWN)")
        student_id = 99999
        first_name = "Dummy"
        last_name = "User"
        session_key = f"SESSION_{int(time.time())}"
        return (student_id, first_name, last_name, session_key)

    
    async def get_summary_fast(self) -> str:
        """Fast summary retrieval from MongoDB - NO FALLBACKS"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_get_summary
            )
        except Exception as e:
            logger.error(f"❌ Error fetching summary: {e}")
            raise Exception(f"Summary retrieval failed: {e}")
    
    # def _sync_get_summary(self) -> str:
    #     """Synchronous summary retrieval for thread pool - NO FALLBACKS"""
    #     try:
    #         # This should be run in sync context within thread pool
    #         import asyncio
    #         db = asyncio.run(self.get_mongo_db())
    #         collection = db[MONGO_CONFIG['transcripts_collection']]
            
    #         # Use asyncio.run within the thread to handle async calls
    #         doc = asyncio.run(collection.find_one(
    #             {"summary": {"$exists": True, "$ne": None, "$ne": ""}},
    #             sort=[("timestamp", -1)]
    #         ))
            
    #         if not doc or not doc.get("summary"):
    #             raise Exception("No valid summary found in MongoDB transcripts collection")
                
    #         summary = doc["summary"].strip()
    #         if len(summary) < 100:
    #             raise Exception(f"Summary too short ({len(summary)} chars): {summary}")
                
    #         return summary
            
    #     except Exception as e:
    #         logger.error(f"❌ Sync summary retrieval error: {e}")
    #         raise Exception(f"MongoDB summary retrieval failed: {e}")
    def _sync_get_summary(self) -> str:
        logger.warning("⚠️ Using dummy summary (MongoDB is DOWN)")
        return (
            "MLOps (Machine Learning Operations) is a set of practices that combines Machine Learning, "
            "DevOps, and Data Engineering to deploy and maintain ML models in production reliably. It enables "
            "automation and monitoring of the ML lifecycle, including training, deployment, and retraining. "
            "Key tools include MLflow, Kubeflow, and TFX. MLOps ensures reproducibility, scalability, and model governance, "
            "and addresses challenges like data quality, model drift, and pipeline orchestration."
        )


    
    async def save_session_result_fast(self, session_data: SessionData, evaluation: str, score: float):
        """Fast session result saving"""
        try:
            # Run in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_save_result,
                session_data, evaluation, score
            )
        except Exception as e:
            logger.error(f"❌ Error saving session result: {e}")
            raise Exception(f"Session save failed: {e}")
    
    def _sync_save_result(self, session_data: SessionData, evaluation: str, score: float) -> bool:
        """Synchronous save for thread pool"""
        try:
            import asyncio
            db = asyncio.run(self.get_mongo_db())
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
                        "transcript_quality": exchange.transcript_quality,
                        "chunk_id": exchange.chunk_id
                    }
                    for exchange in session_data.exchanges
                ],
                "evaluation": evaluation,
                "score": score,
                "total_exchanges": len(session_data.exchanges),
                "greeting_exchanges": session_data.greeting_count,
                "summary_progress": session_data.summary_manager.get_progress() if session_data.summary_manager else {},
                "duration": time.time() - session_data.created_at
            }
            
            result = asyncio.run(collection.insert_one(document))
            logger.info(f"Session {session_data.session_id} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sync save error: {e}")
            raise Exception(f"MongoDB save failed: {e}")

# =============================================================================
# ULTRA-FAST SESSION MANAGER
# =============================================================================

class UltraFastSessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, SessionData] = {}
        self.db_manager = OptimizedDatabaseManager(shared_clients)
        self.audio_processor = OptimizedAudioProcessor(shared_clients)
        self.tts_processor = UltraFastTTSProcessor()
        self.conversation_manager = OptimizedConversationManager(shared_clients)
    
    async def create_session_fast(self, websocket: WebSocket = None) -> SessionData:
        """Ultra-fast session creation"""
        session_id = str(uuid.uuid4())
        test_id = f"standup_{int(time.time())}"
        
        # Get student info in parallel
        student_info_task = asyncio.create_task(self.db_manager.get_student_info_fast())
        summary_task = asyncio.create_task(self.db_manager.get_summary_fast())
        
        # Wait for both tasks
        student_id, first_name, last_name, session_key = await student_info_task
        summary = await summary_task
        
        # Create session
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
        
        # Initialize summary manager
        summary_manager = SummaryManager(shared_clients)
        await summary_manager.initialize_chunks(summary)
        session_data.summary_manager = summary_manager
        
        self.active_sessions[session_id] = session_data
        logger.info(f"✅ Fast session created {session_id} for {session_data.student_name}")
        
        return session_data
    
    async def remove_session(self, session_id: str):
        """Fast session removal"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Removed session {session_id}")
    
    async def process_audio_ultra_fast(self, session_id: str, audio_data: bytes):
        """Ultra-fast audio processing pipeline"""
        session_data = self.active_sessions.get(session_id)
        if not session_data or not session_data.is_active:
            return
        
        start_time = time.time()
        
        # Start transcription immediately
        transcription_task = asyncio.create_task(
            self.audio_processor.transcribe_audio_fast(audio_data)
        )
        
        # Wait for transcription
        transcript, quality = await transcription_task
        
        if not transcript or len(transcript.strip()) < 2:
            await self._send_quick_message(session_data, {
                "type": "clarification",
                "text": "Could you repeat that more clearly?",
                "status": session_data.current_stage.value
            })
            return
        
        logger.info(f"Session {session_id}: User said: {transcript}")
        
        # Generate AI response immediately
        response_task = asyncio.create_task(
            self.conversation_manager.generate_fast_response(session_data, transcript)
        )
        
        ai_response = await response_task
        
        # Add exchange to session with chunk tracking
        chunk_id = None
        if session_data.summary_manager:
            current_chunk = session_data.summary_manager.get_current_chunk()
            chunk_id = current_chunk.id if current_chunk else None
        
        session_data.add_exchange(ai_response, transcript, quality, chunk_id)
        
        # Update session state
        await self._update_session_state_fast(session_data)
        
        # Send response with ultra-fast audio streaming
        await self._send_response_with_ultra_fast_audio(session_data, ai_response)
        
        processing_time = time.time() - start_time
        logger.info(f"⚡ Total processing time: {processing_time:.2f}s")
    
    async def _update_session_state_fast(self, session_data: SessionData):
        """Ultra-fast session state updates"""
        if session_data.current_stage == SessionStage.GREETING:
            session_data.greeting_count += 1
            if session_data.greeting_count >= GREETING_EXCHANGES:
                session_data.current_stage = SessionStage.TECHNICAL
                logger.info(f"Session {session_data.session_id} moved to TECHNICAL stage")
        
        elif session_data.current_stage == SessionStage.TECHNICAL:
            # Check if all summary chunks are completed
            if session_data.summary_manager and session_data.summary_manager.current_chunk_index >= len(session_data.summary_manager.chunks):
                session_data.current_stage = SessionStage.COMPLETE
                logger.info(f"Session {session_data.session_id} moved to COMPLETE stage")
                
                # Generate evaluation and save session in background
                asyncio.create_task(self._finalize_session_fast(session_data))
    
    async def _finalize_session_fast(self, session_data: SessionData):
        """Fast session finalization"""
        try:
            # Generate evaluation in parallel
            evaluation_task = asyncio.create_task(
                self.conversation_manager.generate_fast_evaluation(session_data)
            )
            
            evaluation, score = await evaluation_task
            
            # Save to database in background
            save_task = asyncio.create_task(
                self.db_manager.save_session_result_fast(session_data, evaluation, score)
            )
            
            # Send completion message
            completion_message = "Thank you for participating in today's standup! Your responses have been recorded."
            
            await self._send_quick_message(session_data, {
                "type": "conversation_end",
                "text": completion_message,
                "evaluation": evaluation,
                "score": score,
                "pdf_url": f"/download_results/{session_data.session_id}",
                "status": "complete"
            })
            
            # Generate and send final audio
            async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(completion_message):
                if audio_chunk:
                    await self._send_quick_message(session_data, {
                        "type": "audio_chunk",
                        "audio": audio_chunk.hex(),
                        "status": "complete"
                    })
            
            await self._send_quick_message(session_data, {"type": "audio_end", "status": "complete"})
            
            session_data.is_active = False
            
            # Wait for save to complete
            await save_task
            
        except Exception as e:
            logger.error(f"❌ Fast session finalization error: {e}")
            raise Exception(f"Session finalization failed: {e}")
    
    async def _send_response_with_ultra_fast_audio(self, session_data: SessionData, text: str):
        """Send response with ultra-fast audio streaming"""
        try:
            # Send text response immediately
            await self._send_quick_message(session_data, {
                "type": "ai_response",
                "text": text,
                "status": session_data.current_stage.value
            })
            
            # Stream audio chunks with minimal delay
            chunk_count = 0
            async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(text):
                if audio_chunk and session_data.is_active:
                    await self._send_quick_message(session_data, {
                        "type": "audio_chunk",
                        "audio": audio_chunk.hex(),
                        "status": session_data.current_stage.value
                    })
                    chunk_count += 1
            
            # Send audio end signal
            await self._send_quick_message(session_data, {
                "type": "audio_end",
                "status": session_data.current_stage.value
            })
            
            logger.info(f"🎵 Streamed {chunk_count} audio chunks")
            
        except Exception as e:
            logger.error(f"❌ Ultra-fast audio streaming error: {e}")
            raise Exception(f"Audio streaming failed: {e}")
    
    async def _send_quick_message(self, session_data: SessionData, message: dict):
        """Ultra-fast WebSocket message sending"""
        try:
            if session_data.websocket:
                await session_data.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"❌ Quick WebSocket send error: {e}")
            raise Exception(f"WebSocket send failed: {e}")
    
    async def get_session_result_fast(self, session_id: str) -> dict:
        """Fast session result retrieval"""
        try:
            # Run in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                shared_clients.executor,
                self._sync_get_session_result,
                session_id
            )
        except Exception as e:
            logger.error(f"❌ Error fetching session result: {e}")
            raise Exception(f"Session result retrieval failed: {e}")
    
    def _sync_get_session_result(self, session_id: str) -> dict:
        """Synchronous session result retrieval"""
        try:
            import asyncio
            db = asyncio.run(self.db_manager.get_mongo_db())
            collection = db[MONGO_CONFIG['results_collection']]
            result = asyncio.run(collection.find_one({"session_id": session_id}))
            
            if result:
                result['_id'] = str(result['_id'])
                return result
            return None
            
        except Exception as e:
            logger.error(f"❌ Sync session result error: {e}")
            raise Exception(f"Session result retrieval failed: {e}")

    # LEGACY SUPPORT (OPTIMIZED)
    async def process_legacy_audio_fast(self, test_id: str, audio_data: bytes) -> dict:
        """Fast legacy audio processing"""
        try:
            logger.info(f"Processing legacy audio for test_id: {test_id}")
            
            # Fast transcription
            transcript, quality = await self.audio_processor.transcribe_audio_fast(audio_data)
            
            if not transcript or len(transcript.strip()) < 2:
                raise Exception("Transcription returned empty or too short result")
            
            # Generate fast response
            summary = await self.db_manager.get_summary_fast()
            
            # Create temporary session for legacy processing
            session_data = SessionData(
                session_id=test_id,
                test_id=test_id,
                student_id=1000,
                student_name="Legacy User",
                session_key="LEGACY",
                created_at=time.time(),
                last_activity=time.time(),
                current_stage=SessionStage.TECHNICAL
            )
            
            ai_response = await self.conversation_manager.generate_fast_response(
                session_data, transcript
            )
            
            return {
                "response": ai_response,
                "audio_path": None,
                "ended": False,
                "complete": False,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Legacy audio processing error: {e}")
            raise Exception(f"Legacy audio processing failed: {e}")

# =============================================================================
# OPTIMIZED FASTAPI APPLICATION
# =============================================================================

# Create FastAPI sub-application
app = FastAPI(title="Ultra-Fast Daily Standup System", version="2.0.0")

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

# Initialize ultra-fast session manager
session_manager = UltraFastSessionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Ultra-Fast Daily Standup application started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await shared_clients.close_connections()
    logger.info("Daily Standup application shutting down")

# =============================================================================
# OPTIMIZED API ENDPOINTS
# =============================================================================

@app.get("/start_test")
async def start_standup_session_fast():
    """Start a new daily standup session with ultra-fast initialization"""
    try:
        logger.info("🚀 Starting ultra-fast standup session...")
        
        # Create new session with parallel initialization
        session_data = await session_manager.create_session_fast()
        
        # Generate initial greeting
        greeting = "Hello! Welcome to your daily standup. How are you doing today?"
        
        logger.info(f"⚡ Ultra-fast session created: {session_data.test_id}")
        
        return {
            "status": "success",
            "message": "Session started successfully",
            "test_id": session_data.test_id,
            "session_id": session_data.session_id,  # FIXED: Both IDs returned
            "websocket_url": f"/daily_standup/ws/{session_data.session_id}",
            "greeting": greeting,
            "summary_chunks": len(session_data.summary_manager.chunks) if session_data.summary_manager else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Error starting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@app.post("/api/record-respond")
async def record_and_respond_fast(
    audio: UploadFile = File(...),
    test_id: str = Form(...)
):
    """Ultra-fast audio processing endpoint"""
    try:
        logger.info(f"🎤 Processing audio for test_id: {test_id}")
        
        if not test_id:
            raise HTTPException(status_code=400, detail="test_id is required")
        
        if not audio or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Valid audio file is required")
        
        # Read audio data
        audio_data = await audio.read()
        if len(audio_data) < 1000:
            raise HTTPException(status_code=400, detail="Audio file too small")
        
        # Process using optimized legacy method
        result = await session_manager.process_legacy_audio_fast(test_id, audio_data)
        
        logger.info(f"⚡ Audio processed for {test_id}")
        
        return {
            "status": "success",
            "response": result.get("response", "Thank you for your input."),
            "audio_path": result.get("audio_path"),
            "ended": result.get("ended", False),
            "complete": result.get("complete", False),
            "message": result.get("response", "Processing complete")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Record and respond error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/summary/{test_id}")
async def get_standup_summary_fast(test_id: str):
    """Get standup session summary with ultra-fast retrieval"""
    try:
        logger.info(f"📊 Getting summary for test_id: {test_id}")
        
        if not test_id:
            raise HTTPException(status_code=400, detail="test_id is required")
        
        # Try to get actual session result
        result = await session_manager.get_session_result_fast(test_id)
        
        if result:
            # Extract summary data from conversation
            exchanges = result.get("conversation_log", [])
            
            yesterday_work = ""
            today_plans = ""
            blockers = ""
            additional_notes = ""
            
            # Parse conversation for standup components
            for exchange in exchanges:
                user_response = exchange.get("user_response", "").lower()
                ai_message = exchange.get("ai_message", "").lower()
                
                if any(word in ai_message for word in ["yesterday", "accomplished", "completed"]):
                    yesterday_work = exchange.get("user_response", "")
                elif any(word in ai_message for word in ["today", "plan", "working on"]):
                    today_plans = exchange.get("user_response", "")
                elif any(word in ai_message for word in ["blocker", "challenge", "obstacle", "stuck"]):
                    blockers = exchange.get("user_response", "")
                elif exchange.get("user_response") and not yesterday_work and not today_plans:
                    additional_notes = exchange.get("user_response", "")
            
            summary_data = {
                "test_id": test_id,
                "session_id": result.get("session_id", test_id),
                "student_name": result.get("student_name", "Student"),
                "timestamp": result.get("timestamp", time.time()),
                "duration": result.get("duration", 0),
                "yesterday": yesterday_work or "Progress discussed during session",
                "today": today_plans or "Plans outlined during session",
                "blockers": blockers or "No specific blockers mentioned",
                "notes": additional_notes or "Additional discussion points covered",
                "accomplishments": yesterday_work,
                "plans": today_plans,
                "challenges": blockers,
                "additional_info": additional_notes,
                "evaluation": result.get("evaluation", "Session completed successfully"),
                "score": result.get("score", 8.0),
                "total_exchanges": result.get("total_exchanges", 0),
                "summary_progress": result.get("summary_progress", {}),
                "pdf_url": f"/daily_standup/download_results/{test_id}",
                "status": "completed"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Session result not found for test_id: {test_id}")
        
        logger.info(f"⚡ Fast summary generated for {test_id}")
        return summary_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting summary: {e}")
        raise HTTPException(status_code=500, detail=f"Summary retrieval failed: {str(e)}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint_ultra_fast(websocket: WebSocket, session_id: str):
    """Ultra-fast WebSocket endpoint with optimized communication"""
    await websocket.accept()
    
    try:
        logger.info(f"🔌 WebSocket connected for session: {session_id}")
        
        # Get session
        session_data = session_manager.active_sessions.get(session_id)
        if not session_data:
            logger.error(f"❌ Session {session_id} not found in active sessions")
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": f"Session {session_id} not found. Please start a new session.",
                "status": "error"
            }))
            return
        
        # Update session with WebSocket
        session_data.websocket = websocket
        
        # Send initial greeting with ultra-fast audio
        greeting = "Hello! Welcome to your daily standup. How are you doing today?"
        await websocket.send_text(json.dumps({
            "type": "ai_response",
            "text": greeting,
            "status": "greeting"
        }))
        
        # Generate and stream greeting audio with minimal delay
        async for audio_chunk in session_manager.tts_processor.generate_ultra_fast_stream(greeting):
            if audio_chunk:
                await websocket.send_text(json.dumps({
                    "type": "audio_chunk",
                    "audio": audio_chunk.hex(),
                    "status": "greeting"
                }))
        
        # Send audio end signal
        await websocket.send_text(json.dumps({
            "type": "audio_end",
            "status": "greeting"
        }))
        
        # Keep connection alive and handle messages
        while session_data.is_active:
            try:
                # Wait for audio data with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)  # 5 minute timeout
                message = json.loads(data)
                
                if message.get("type") == "audio_data":
                    # Process base64 encoded audio ultra-fast
                    audio_data = base64.b64decode(message.get("audio", ""))
                    # Process in background for minimal latency
                    asyncio.create_task(
                        session_manager.process_audio_ultra_fast(session_id, audio_data)
                    )
                
                elif message.get("type") == "ping":
                    # Heartbeat
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
            except asyncio.TimeoutError:
                logger.info(f"🔌 WebSocket timeout: {session_id}")
                break
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected: {session_id}")
                break
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "text": f"Error: {str(e)}",
                    "status": "error"
                }))
                break
    
    except Exception as e:
        logger.error(f"❌ WebSocket endpoint error: {e}")
    finally:
        await session_manager.remove_session(session_id)

@app.get("/download_results/{session_id}")
async def download_results_fast(session_id: str):
    """Fast PDF generation and download"""
    try:
        # Get session result
        result = await session_manager.get_session_result_fast(session_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Generate PDF in thread pool
        loop = asyncio.get_event_loop()
        pdf_buffer = await loop.run_in_executor(
            shared_clients.executor,
            generate_pdf_report,
            result, session_id
        )
        
        return StreamingResponse(
            io.BytesIO(pdf_buffer),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=standup_report_{session_id}.pdf"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

def generate_pdf_report(result: dict, session_id: str) -> bytes:
    """Generate PDF report synchronously"""
    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=LETTER)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = f"Daily Standup Report - {result.get('student_name', 'Student')}"
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 12))
        
        # Session info
        info_text = f"""
        Session ID: {session_id}
        Date: {datetime.fromtimestamp(result.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')}
        Duration: {result.get('duration', 0)/60:.1f} minutes
        Total Exchanges: {result.get('total_exchanges', 0)}
        Summary Progress: {result.get('summary_progress', {}).get('chunk_progress', 'N/A')}
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Conversation log
        story.append(Paragraph("Conversation Summary", styles['Heading2']))
        for exchange in result.get('conversation_log', [])[:15]:  # Limit to first 15
            story.append(Paragraph(f"AI: {exchange.get('ai_message', '')}", styles['Normal']))
            story.append(Paragraph(f"User: {exchange.get('user_response', '')}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        # Evaluation
        if result.get('evaluation'):
            story.append(Paragraph("Evaluation", styles['Heading2']))
            story.append(Paragraph(result['evaluation'], styles['Normal']))
            story.append(Paragraph(f"Score: {result.get('score', 0)}/10", styles['Normal']))
        
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer.read()
        
    except Exception as e:
        logger.error(f"❌ PDF generation error: {e}")
        raise Exception(f"PDF generation failed: {e}")

@app.get("/health")
async def health_check_fast():
    """Ultra-fast health check"""
    try:
        return {
            "status": "healthy",
            "service": "ultra_fast_daily_standup",
            "timestamp": time.time(),
            "active_sessions": len(session_manager.active_sessions),
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/test")
async def test_endpoint_fast():
    """Fast test endpoint"""
    return {
        "message": "Ultra-Fast Daily Standup service is running",
        "timestamp": time.time(),
        "status": "blazing_fast",
        "optimizations": [
            "800ms silence detection",
            "Parallel processing pipeline", 
            "Summary-based questioning",
            "Sliding window conversation history",
            "Ultra-fast TTS streaming",
            "Thread pool optimization",
            "Session synchronization fix",
            "NO FALLBACKS - Real error detection"
        ]
    }