"""
AI Services module for Daily Standup application
Handles all AI-related operations: LLM calls, TTS, STT, and conversation management
"""

import os
import time
import logging
import asyncio
import edge_tts
import openai
import re
import uuid
from groq import Groq
from typing import List, AsyncGenerator, Tuple, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures

from .config import config
from .prompts import prompts

logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS
# =============================================================================

class SessionStage(Enum):
    GREETING = "greeting"
    TECHNICAL = "technical"
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
    conversation_window: deque = field(default_factory=lambda: deque(maxlen=config.CONVERSATION_WINDOW_SIZE))
    greeting_count: int = 0
    is_active: bool = True
    websocket: Optional[Any] = field(default=None)
    summary_manager: Optional[Any] = field(default=None)
    clarification_attempts: int = field(default=0)
    
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

@dataclass
class SummaryChunk:
    id: int
    content: str
    base_questions: List[str]
    current_question_count: int = 0
    completed: bool = False
    follow_up_questions: List[str] = field(default_factory=list)

# =============================================================================
# SHARED CLIENT MANAGER
# =============================================================================

class SharedClientManager:
    """Optimized client management with connection pooling"""
    
    def __init__(self):
        self._groq_client = None
        self._openai_client = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.THREAD_POOL_MAX_WORKERS)
        
    @property
    def groq_client(self) -> Groq:
        if self._groq_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise Exception("GROQ_API_KEY not found in environment variables")
            self._groq_client = Groq(api_key=api_key)
            logger.info("✅ Groq client initialized")
        return self._groq_client
    
    @property 
    def openai_client(self) -> openai.OpenAI:
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OPENAI_API_KEY not found in environment variables")
            self._openai_client = openai.OpenAI(api_key=api_key)
            logger.info("✅ OpenAI client initialized")
        return self._openai_client
    
    @property
    def executor(self):
        return self._executor
    
    async def close_connections(self):
        """Cleanup method for graceful shutdown"""
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info("🔌 AI client connections closed")

# Global shared client manager
shared_clients = SharedClientManager()

# =============================================================================
# SUMMARY MANAGEMENT SYSTEM
# =============================================================================

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
            raw_chunks = await self._split_summary_semantically(summary)
            
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
            prompt = prompts.summary_splitting_prompt(summary)
            
            response = self.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL,
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
            
            return chunks[:config.SUMMARY_CHUNKS]
            
        except Exception as e:
            logger.error(f"❌ Semantic splitting failed: {e}")
            raise Exception(f"Summary semantic splitting failed: {e}")
    
    async def _generate_base_questions(self, chunk_content: str) -> List[str]:
        """Generate base questions for a chunk"""
        try:
            prompt = prompts.base_questions_prompt(chunk_content)
            
            response = self.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL,
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
                
            return questions[:config.BASE_QUESTIONS_PER_CHUNK]
            
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
            prompt = prompts.followup_analysis_prompt(chunk.content, user_response)
            
            response = self.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL,
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
            remaining_slots = config.MAX_QUESTIONS_PER_CHUNK - total_questions
            
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
# AUDIO PROCESSING
# =============================================================================

class OptimizedAudioProcessor:
    def __init__(self, client_manager: SharedClientManager):
        self.client_manager = client_manager
    
    @property
    def groq_client(self):
        return self.client_manager.groq_client
    
    async def transcribe_audio_fast(self, audio_data: bytes) -> Tuple[str, float]:
        """Ultra-fast transcription with parallel processing"""
        try:
            if len(audio_data) < 1000:
                raise Exception("Audio data too small for transcription")
            
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
    
    def _sync_transcribe(self, audio_data: bytes) -> Tuple[str, float]:
        """Synchronous transcription for thread pool"""
        try:
            temp_file = config.TEMP_DIR / f"audio_{int(time.time() * 1000000)}.webm"
            
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            with open(temp_file, "rb") as file:
                result = self.groq_client.audio.transcriptions.create(
                    file=(temp_file.name, file.read()),
                    model=config.GROQ_TRANSCRIPTION_MODEL,
                    response_format="verbose_json"
                )
            
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
        self.voice = config.TTS_VOICE
        self.rate = config.TTS_RATE
    
    def split_text_optimized(self, text: str) -> List[str]:
        """Optimized text splitting for minimal latency"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > config.TTS_CHUNK_SIZE * 5:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    async def generate_ultra_fast_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Ultra-fast audio generation with parallel processing"""
        try:
            chunks = self.split_text_optimized(text)
            
            tasks = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                if i == 0:
                    async for audio_chunk in self._generate_chunk_audio(chunk):
                        if audio_chunk:
                            yield audio_chunk
                else:
                    tasks.append(self._generate_chunk_audio(chunk))
            
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
# CONVERSATION MANAGEMENT
# =============================================================================

class OptimizedConversationManager:
    def __init__(self, client_manager: SharedClientManager):
        self.client_manager = client_manager
    
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
        """Dynamic greeting responses with context awareness"""
        context = {
            'recent_exchanges': [
                f"AI: {ex.ai_message}, User: {ex.user_response}" 
                for ex in list(session_data.conversation_window)[-2:]
            ]
        }
        
        prompt = prompts.dynamic_greeting_response(user_input, session_data.greeting_count, context)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.client_manager.executor,
            self._sync_openai_call,
            prompt
        )
        
        return response
    
    async def _generate_technical_response(self, session_data: SessionData, user_input: str) -> str:
        """Generate dynamic technical responses using context"""
        if not session_data.summary_manager:
            raise Exception("Summary manager not initialized")
        
        await session_data.summary_manager.analyze_response_and_generate_followups(user_input)
        
        next_question = session_data.summary_manager.get_next_question()
        
        if next_question:
            context = self._build_conversation_context(session_data)
            session_state = {
                'questions_asked': sum(1 for ex in session_data.exchanges if ex.stage == SessionStage.TECHNICAL),
                'current_topic': session_data.summary_manager.get_current_chunk().content[:50] if session_data.summary_manager.get_current_chunk() else 'technical work'
            }
            
            prompt = prompts.dynamic_technical_response(context, user_input, next_question, session_state)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_openai_call,
                prompt
            )
            
            return response
        
        if session_data.summary_manager.should_move_to_next_chunk():
            if session_data.summary_manager.move_to_next_chunk():
                next_question = session_data.summary_manager.get_next_question()
                if next_question:
                    progress_info = {
                        'current_chunk': session_data.summary_manager.current_chunk_index,
                        'total_chunks': len(session_data.summary_manager.chunks)
                    }
                    
                    prompt = prompts.dynamic_chunk_transition(user_input, next_question, progress_info)
                    
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        self.client_manager.executor,
                        self._sync_openai_call,
                        prompt
                    )
                    
                    return response
        
        # Session completion with dynamic response
        session_data.current_stage = SessionStage.COMPLETE
        conversation_summary = {
            'topics_covered': list(set(ex.chunk_id for ex in session_data.exchanges if ex.chunk_id)),
            'total_exchanges': len(session_data.exchanges)
        }
        
        prompt = prompts.dynamic_session_completion(conversation_summary)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.client_manager.executor,
            self._sync_openai_call,
            prompt
        )
        
        return response
    
    def _build_conversation_context(self, session_data: SessionData) -> str:
        """Build context from sliding window of conversation"""
        context = ""
        for exchange in list(session_data.conversation_window)[-3:]:
            context += f"AI: {exchange.ai_message}\nUser: {exchange.user_response}\n\n"
        return context.strip()
    
    async def _generate_conclusion_response(self, session_data: SessionData, user_input: str) -> str:
        """Dynamic conclusion responses with session context"""
        session_context = {
            'key_topics': list(set(ex.chunk_id for ex in session_data.exchanges if ex.chunk_id))[:3],
            'total_exchanges': len(session_data.exchanges)
        }
        
        prompt = prompts.dynamic_conclusion_response(user_input, session_context)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.client_manager.executor,
            self._sync_openai_call,
            prompt
        )
        
        return response
    
    def _sync_openai_call(self, prompt: str) -> str:
        """Synchronous OpenAI call for thread pool"""
        try:
            response = self.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.OPENAI_TEMPERATURE,
                max_tokens=config.OPENAI_MAX_TOKENS
            )
            result = response.choices[0].message.content.strip()
            if not result:
                raise Exception("OpenAI returned empty response")
            return result
        except Exception as e:
            logger.error(f"❌ OpenAI API call failed: {e}")
            raise Exception(f"OpenAI API failed: {e}")
    
    async def generate_fast_evaluation(self, session_data: SessionData) -> Tuple[str, float]:
        """Generate dynamic evaluation based on actual conversation"""
        try:
            conversation_exchanges = []
            for exchange in session_data.exchanges[-10:]:
                if exchange.stage == SessionStage.TECHNICAL:
                    conversation_exchanges.append({
                        'ai_message': exchange.ai_message,
                        'user_response': exchange.user_response,
                        'chunk_id': exchange.chunk_id,
                        'quality': exchange.transcript_quality
                    })
            
            if not conversation_exchanges:
                raise Exception("No technical exchanges found for evaluation")
            
            session_stats = {
                'duration_minutes': round((time.time() - session_data.created_at) / 60, 1),
                'avg_response_length': sum(len(ex['user_response']) for ex in conversation_exchanges) // len(conversation_exchanges)
            }
            
            prompt = prompts.dynamic_evaluation_prompt(conversation_exchanges, session_stats)
            
            loop = asyncio.get_event_loop()
            evaluation = await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_openai_call,
                prompt
            )
            
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)/10', evaluation)
            if not score_match:
                raise Exception(f"Could not extract score from evaluation: {evaluation}")
                
            score = float(score_match.group(1))
            
            return evaluation, score
            
        except Exception as e:
            logger.error(f"❌ Fast evaluation error: {e}")
            raise Exception(f"Evaluation generation failed: {e}")