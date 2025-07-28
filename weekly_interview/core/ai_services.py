# weekly_interview/core/ai_services.py
"""
AI Services module for Enhanced Mock Interview System
Handles LLM interactions, TTS, STT, and interview session management
"""

import os
import time
import logging
import asyncio
import edge_tts
import openai
import re
import uuid
import base64
from groq import Groq
from typing import List, AsyncGenerator, Tuple, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures

from .config import config
from .content_service import ContentService

logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS
# =============================================================================

class InterviewStage(Enum):
    GREETING = "greeting"
    TECHNICAL = "technical" 
    COMMUNICATION = "communication"
    HR = "hr"
    COMPLETE = "complete"
    ERROR = "error"

class InterviewState(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ROUND_TRANSITION = "round_transition"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ConversationExchange:
    timestamp: float
    stage: InterviewStage
    ai_message: str
    user_response: str = ""
    transcript_quality: float = 0.0
    round_number: int = 1
    question_number: int = 1
    is_followup: bool = False
    processing_time: float = 0.0

@dataclass
class InterviewSession:
    session_id: str
    test_id: str
    student_id: int
    student_name: str
    session_key: str
    created_at: float
    last_activity: float
    current_stage: InterviewStage
    current_state: InterviewState
    exchanges: List[ConversationExchange] = field(default_factory=list)
    conversation_window: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # Round management
    questions_per_round: Dict[str, int] = field(default_factory=lambda: {
        "greeting": 0, "technical": 0, "communication": 0, "hr": 0
    })
    round_start_times: Dict[str, float] = field(default_factory=dict)
    current_round_number: int = 1
    
    # Session state
    is_active: bool = True
    websocket: Optional[Any] = field(default=None)
    tts_voice: str = field(default=config.TTS_VOICE)
    
    # Interview content
    interview_content: str = ""
    total_questions_asked: int = 0
    followup_questions: int = 0
    
    # Quality metrics
    avg_response_time: float = 0.0
    audio_quality_scores: List[float] = field(default_factory=list)
    
    def add_exchange(self, ai_message: str, user_response: str = "", quality: float = 0.0,
                    is_followup: bool = False, processing_time: float = 0.0):
        """Add conversation exchange with enhanced tracking"""
        exchange = ConversationExchange(
            timestamp=time.time(),
            stage=self.current_stage,
            ai_message=ai_message,
            user_response=user_response,
            transcript_quality=quality,
            round_number=self.current_round_number,
            question_number=self.questions_per_round[self.current_stage.value] + 1,
            is_followup=is_followup,
            processing_time=processing_time
        )
        
        self.exchanges.append(exchange)
        self.conversation_window.append(exchange)
        self.last_activity = time.time()
        
        # Update counters
        self.questions_per_round[self.current_stage.value] += 1
        self.total_questions_asked += 1
        
        if is_followup:
            self.followup_questions += 1
        
        # Update quality metrics
        if quality > 0:
            self.audio_quality_scores.append(quality)
    
    def update_last_response(self, user_response: str, quality: float = 0.0):
        """Update the last exchange with user response"""
        if self.exchanges:
            self.exchanges[-1].user_response = user_response
            self.exchanges[-1].transcript_quality = quality
            if quality > 0:
                self.audio_quality_scores.append(quality)

# =============================================================================
# SHARED CLIENT MANAGER
# =============================================================================

class SharedClientManager:
    """Optimized client management with connection pooling"""
    
    def __init__(self):
        self._groq_client = None
        self._openai_client = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.THREAD_POOL_MAX_WORKERS
        )
        
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
        logger.info("✅ AI client connections closed")

# Global shared client manager
shared_clients = SharedClientManager()

# =============================================================================
# INTERVIEW SESSION MANAGER
# =============================================================================

class InterviewSessionManager:
    """Enhanced session management with WebSocket support"""
    
    def __init__(self, db_manager):
        self.active_sessions: Dict[str, InterviewSession] = {}
        self.db_manager = db_manager
        self.content_service = ContentService(db_manager)
        
    async def create_session_fast(self, websocket: Optional[Any] = None) -> InterviewSession:
        """Ultra-fast session creation with real database connections"""
        session_id = str(uuid.uuid4())
        test_id = f"interview_{int(time.time())}"
        
        try:
            # Get student info and content in parallel
            student_info_task = asyncio.create_task(
                self.db_manager.get_student_info_fast()
            )
            content_task = asyncio.create_task(
                self.content_service.get_interview_content_context()
            )
            
            student_id, first_name, last_name, session_key = await student_info_task
            interview_content = await content_task
            
            # Validate data
            if not interview_content or len(interview_content.strip()) < config.MIN_CONTENT_LENGTH:
                raise Exception("Invalid interview content retrieved")
            
            if not first_name or not last_name:
                raise Exception("Invalid student data retrieved")
            
            # Create session
            session_data = InterviewSession(
                session_id=session_id,
                test_id=test_id,
                student_id=student_id,
                student_name=f"{first_name} {last_name}",
                session_key=session_key,
                created_at=time.time(),
                last_activity=time.time(),
                current_stage=InterviewStage.GREETING,
                current_state=InterviewState.NOT_STARTED,
                websocket=websocket,
                interview_content=interview_content
            )
            
            self.active_sessions[session_id] = session_data
            
            logger.info(f"✅ Interview session created: {session_id} for {session_data.student_name}")
            
            return session_data
            
        except Exception as e:
            logger.error(f"❌ Failed to create interview session: {e}")
            raise Exception(f"Session creation failed: {e}")
    
    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """Get session by ID with activity update"""
        session = self.active_sessions.get(session_id)
        if session:
            session.last_activity = time.time()
        return session
    
    def validate_session(self, session_id: str) -> InterviewSession:
        """Validate session and check timeout"""
        session = self.get_session(session_id)
        if not session:
            raise Exception("Interview session not found")
        
        if time.time() > session.last_activity + config.SESSION_TIMEOUT:
            self.cleanup_session(session_id)
            raise Exception("Interview session timed out")
        
        return session
    
    def cleanup_session(self, session_id: str):
        """Remove session from active sessions"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"🗑️ Session cleaned up: {session_id}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = [
            sid for sid, session in self.active_sessions.items()
            if current_time > session.last_activity + config.SESSION_TIMEOUT
        ]
        
        for sid in expired_sessions:
            self.cleanup_session(sid)
        
        return len(expired_sessions)

# =============================================================================
# AUDIO PROCESSING
# =============================================================================

class OptimizedAudioProcessor:
    """Enhanced audio processing with real-time capabilities"""
    
    def __init__(self, client_manager: SharedClientManager):
        self.client_manager = client_manager
    
    @property
    def groq_client(self):
        return self.client_manager.groq_client
    
    async def transcribe_audio_fast(self, audio_data: bytes) -> Tuple[str, float]:
        """Ultra-fast transcription with enhanced error handling"""
        try:
            audio_size = len(audio_data)
            logger.info(f"🎙️ Transcribing {audio_size} bytes of audio")
            
            if audio_size < 100:
                raise Exception(f"Audio data too small for transcription ({audio_size} bytes)")
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_transcribe,
                audio_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
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
                    model=config.GROQ_MODEL,
                    response_format="verbose_json",
                    prompt="Please transcribe this interview response clearly."
                )
            
            # Cleanup
            try:
                temp_file.unlink()
            except:
                pass
            
            transcript = result.text.strip() if result.text else ""
            
            if not transcript:
                logger.warning(f"⚠️ Groq returned empty transcript")
                return "", 0.0
            
            # Quality assessment
            quality = min(len(transcript) / 30, 1.0)
            if hasattr(result, 'segments') and result.segments:
                confidences = [seg.get('confidence', 0.8) for seg in result.segments[:3]]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    quality = (quality + avg_confidence) / 2
            
            logger.info(f"✅ Transcription: '{transcript}' (quality: {quality:.2f})")
            return transcript, quality
            
        except Exception as e:
            logger.error(f"❌ Sync transcription error: {e}")
            raise Exception(f"Groq transcription failed: {e}")

class UltraFastTTSProcessor:
    """Enhanced TTS processor with streaming and optimization"""
    
    def __init__(self):
        self.voice = config.TTS_VOICE
        self.speed = config.TTS_SPEED
    
    def split_text_for_streaming(self, text: str) -> List[str]:
        """Split text into optimal chunks for streaming"""
        if not text:
            return []
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > config.TTS_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    async def generate_audio_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate audio stream with enhanced optimization"""
        try:
            chunks = self.split_text_for_streaming(text)
            logger.info(f"🔊 Generating TTS for {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                try:
                    # Generate audio for chunk
                    async for audio_data in self._generate_chunk_audio(chunk):
                        if audio_data:
                            yield audio_data
                            
                except Exception as e:
                    logger.error(f"❌ TTS chunk {i} failed: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ TTS stream generation failed: {e}")
            raise Exception(f"TTS generation failed: {e}")
    
    async def _generate_chunk_audio(self, chunk: str) -> AsyncGenerator[bytes, None]:
        """Generate audio for a single text chunk"""
        try:
            # Configure TTS with speed adjustment
            rate_modifier = f"+{int((self.speed - 1) * 100)}%" if self.speed > 1 else f"{int((self.speed - 1) * 100)}%"
            
            tts = edge_tts.Communicate(chunk, self.voice, rate=rate_modifier)
            audio_buffer = b""
            
            async for tts_chunk in tts.stream():
                if tts_chunk["type"] == "audio":
                    audio_buffer += tts_chunk["data"]
            
            if audio_buffer:
                yield audio_buffer
            else:
                logger.warning(f"⚠️ Empty audio buffer for chunk: {chunk[:50]}...")
                
        except Exception as e:
            logger.error(f"❌ Chunk TTS generation failed: {e}")
            raise Exception(f"TTS chunk failed: {e}")

# =============================================================================
# CONVERSATION MANAGEMENT
# =============================================================================

class OptimizedConversationManager:
    """Enhanced conversation management with interview-specific logic"""
    
    def __init__(self, client_manager: SharedClientManager):
        self.client_manager = client_manager
    
    @property
    def openai_client(self):
        return self.client_manager.openai_client
    
    async def generate_interview_response(self, session: InterviewSession, user_input: str = "") -> str:
        """Generate interview response based on current stage"""
        try:
            start_time = time.time()
            
            if session.current_stage == InterviewStage.GREETING:
                response = await self._generate_greeting_response(session, user_input)
            elif session.current_stage == InterviewStage.TECHNICAL:
                response = await self._generate_technical_response(session, user_input)
            elif session.current_stage == InterviewStage.COMMUNICATION:
                response = await self._generate_communication_response(session, user_input)
            elif session.current_stage == InterviewStage.HR:
                response = await self._generate_hr_response(session, user_input)
            else:
                response = "Thank you for completing the interview. Your evaluation is being prepared."
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Generated {session.current_stage.value} response in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            raise Exception(f"AI response generation failed: {e}")
    
    async def _generate_greeting_response(self, session: InterviewSession, user_input: str) -> str:
        """Generate greeting and introduction"""
        if session.questions_per_round["greeting"] == 0:
            # Initial greeting
            return f"Hello {session.student_name}! Welcome to your mock interview session. I'm excited to learn more about your technical skills and experience. How are you feeling today, and are you ready to begin?"
        else:
            # Transition to technical round
            prompt = f"""
You are a professional interviewer. The candidate just responded: "{user_input}"

Acknowledge their response warmly and transition smoothly to the technical assessment.

Keep it brief and professional. Example: "That's great to hear! Let's begin with some technical questions to assess your skills."

Response:"""

            return await self._call_openai(prompt, max_tokens=150)
    
    async def _generate_technical_response(self, session: InterviewSession, user_input: str) -> str:
        """Generate technical interview questions and responses"""
        
        # Check if we should transition to next round
        if session.questions_per_round["technical"] >= config.MAX_QUESTIONS_PER_ROUND:
            return "Thank you for the technical discussion. Now let's evaluate your communication and presentation skills."
        
        # Get conversation context
        context = self._build_conversation_context(session, "technical")
        
        prompt = f"""
You are conducting a technical interview. Generate the next appropriate question based on the conversation flow.

INTERVIEW CONTENT FOR REFERENCE:
{session.interview_content[:1500]}

CONVERSATION CONTEXT:
{context}

CANDIDATE'S LAST RESPONSE: "{user_input}"

CURRENT QUESTION NUMBER: {session.questions_per_round["technical"] + 1}

INSTRUCTIONS:
1. If this is the first technical question, start with a foundational question based on the content
2. If the candidate gave a good answer, ask a more challenging follow-up or related question
3. If the candidate struggled, ask a simpler related question or provide gentle guidance
4. Focus on practical application, problem-solving, and real-world scenarios
5. Keep questions clear, specific, and interview-appropriate
6. Make it conversational but professional

Generate your response:"""

        return await self._call_openai(prompt, max_tokens=300)
    
    async def _generate_communication_response(self, session: InterviewSession, user_input: str) -> str:
        """Generate communication round questions"""
        
        if session.questions_per_round["communication"] >= config.MAX_QUESTIONS_PER_ROUND:
            return "Excellent communication skills demonstrated. Let's move to our final HR assessment to discuss your experiences and cultural fit."
        
        context = self._build_conversation_context(session, "communication")
        
        prompt = f"""
You are assessing communication skills in an interview. Generate the next appropriate question.

CONVERSATION CONTEXT:
{context}

CANDIDATE'S LAST RESPONSE: "{user_input}"

CURRENT QUESTION NUMBER: {session.questions_per_round["communication"] + 1}

FOCUS AREAS FOR COMMUNICATION ROUND:
- Explanation and presentation skills
- Clarity of communication
- Ability to articulate complex ideas
- Professional communication style
- Persuasion and storytelling abilities

INSTRUCTIONS:
1. Ask questions that require clear explanation or presentation
2. Test their ability to communicate technical concepts to non-technical audiences
3. Assess storytelling and persuasion skills
4. Focus on real scenarios they might face in a professional environment
5. Keep it conversational and engaging

Generate your communication assessment question:"""

        return await self._call_openai(prompt, max_tokens=250)
    
    async def _generate_hr_response(self, session: InterviewSession, user_input: str) -> str:
        """Generate HR/behavioral interview questions"""
        
        if session.questions_per_round["hr"] >= config.MAX_QUESTIONS_PER_ROUND:
            session.current_stage = InterviewStage.COMPLETE
            return "Thank you for completing all rounds of the interview. Your comprehensive evaluation will be prepared shortly. You've done an excellent job today!"
        
        context = self._build_conversation_context(session, "hr")
        
        prompt = f"""
You are conducting the HR/behavioral round of an interview. Generate the next appropriate question.

CONVERSATION CONTEXT:
{context}

CANDIDATE'S LAST RESPONSE: "{user_input}"

CURRENT QUESTION NUMBER: {session.questions_per_round["hr"] + 1}

FOCUS AREAS FOR HR ROUND:
- Past experiences and achievements
- Teamwork and collaboration
- Conflict resolution and problem-solving
- Leadership potential and initiative
- Cultural fit and values alignment
- Career goals and motivation

INSTRUCTIONS:
1. Use behavioral interview techniques (STAR method encouragement)
2. Ask about specific past experiences and situations
3. Assess cultural fit and soft skills
4. Focus on teamwork, leadership, and adaptability
5. Keep questions open-ended to encourage detailed responses
6. Be supportive and encouraging

Generate your behavioral assessment question:"""

        return await self._call_openai(prompt, max_tokens=250)
    
    def _build_conversation_context(self, session: InterviewSession, round_type: str) -> str:
        """Build conversation context for specific round"""
        round_exchanges = [
            ex for ex in session.exchanges 
            if ex.stage.value == round_type and ex.user_response
        ]
        
        if not round_exchanges:
            return "No previous conversation in this round."
        
        # Get last few exchanges
        recent_exchanges = round_exchanges[-3:]
        context_parts = []
        
        for ex in recent_exchanges:
            context_parts.append(f"Interviewer: {ex.ai_message}")
            if ex.user_response:
                context_parts.append(f"Candidate: {ex.user_response}")
        
        return "\n".join(context_parts)
    
    async def _call_openai(self, prompt: str, max_tokens: int = None) -> str:
        """Call OpenAI API with retry logic"""
        if max_tokens is None:
            max_tokens = config.OPENAI_MAX_TOKENS
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.client_manager.executor,
            self._sync_openai_call,
            prompt,
            max_tokens
        )
    
    def _sync_openai_call(self, prompt: str, max_tokens: int) -> str:
        """Synchronous OpenAI call for thread pool"""
        try:
            response = self.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.OPENAI_TEMPERATURE,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            if not result:
                raise Exception("OpenAI returned empty response")
            return result
            
        except Exception as e:
            logger.error(f"❌ OpenAI API call failed: {e}")
            raise Exception(f"OpenAI API failed: {e}")
    
    async def generate_comprehensive_evaluation(self, session: InterviewSession) -> Tuple[str, Dict[str, float]]:
        """Generate comprehensive interview evaluation with strict scoring"""
        try:
            # Separate exchanges by round
            technical_exchanges = [ex for ex in session.exchanges if ex.stage == InterviewStage.TECHNICAL]
            communication_exchanges = [ex for ex in session.exchanges if ex.stage == InterviewStage.COMMUNICATION]
            hr_exchanges = [ex for ex in session.exchanges if ex.stage == InterviewStage.HR]
            
            # Prepare analytics
            session_analytics = {
                "total_duration_minutes": round((time.time() - session.created_at) / 60, 1),
                "total_questions": session.total_questions_asked,
                "technical_questions": len(technical_exchanges),
                "communication_questions": len(communication_exchanges),
                "hr_questions": len(hr_exchanges),
                "followup_questions": session.followup_questions,
                "avg_audio_quality": round(sum(session.audio_quality_scores) / len(session.audio_quality_scores), 2) if session.audio_quality_scores else 0,
                "response_consistency": len([ex for ex in session.exchanges if len(ex.user_response) > 20]) / max(len(session.exchanges), 1)
            }
            
            # Generate evaluation
            evaluation_prompt = self._create_evaluation_prompt(
                technical_exchanges, communication_exchanges, hr_exchanges, session_analytics
            )
            
            evaluation = await self._call_openai(evaluation_prompt, max_tokens=1000)
            
            # Extract scores
            scores = self._extract_scores_from_evaluation(evaluation)
            
            logger.info(f"✅ Evaluation generated for {session.test_id}")
            return evaluation, scores
            
        except Exception as e:
            logger.error(f"❌ Evaluation generation failed: {e}")
            raise Exception(f"Evaluation generation failed: {e}")
    
    def _create_evaluation_prompt(self, technical_exchanges: List, communication_exchanges: List, 
                                hr_exchanges: List, analytics: Dict) -> str:
        """Create comprehensive evaluation prompt with strict criteria"""
        
        def format_exchanges(exchanges: List) -> str:
            if not exchanges:
                return "No exchanges recorded for this round."
            
            formatted = []
            for ex in exchanges:
                formatted.append(f"Q: {ex.ai_message}")
                if ex.user_response:
                    formatted.append(f"A: {ex.user_response}")
            return "\n".join(formatted)
        
        prompt = f"""
You are evaluating a comprehensive mock interview with STRICT professional standards. Provide detailed, honest assessment.

INTERVIEW ANALYTICS:
- Duration: {analytics['total_duration_minutes']} minutes
- Total Questions: {analytics['total_questions']}
- Technical Questions: {analytics['technical_questions']}
- Communication Questions: {analytics['communication_questions']}
- HR Questions: {analytics['hr_questions']}
- Follow-up Questions: {analytics['followup_questions']}
- Audio Quality: {analytics['avg_audio_quality']}/1.0
- Response Consistency: {analytics['response_consistency']:.2f}

TECHNICAL ROUND:
{format_exchanges(technical_exchanges)}

COMMUNICATION ROUND:
{format_exchanges(communication_exchanges)}

HR/BEHAVIORAL ROUND:
{format_exchanges(hr_exchanges)}

EVALUATION CRITERIA (STRICT STANDARDS):
- Technical Assessment (35%): Knowledge depth, problem-solving, accuracy, practical application
- Communication Skills (30%): Clarity, presentation, articulation, professional communication
- Behavioral/Cultural Fit (25%): Teamwork, leadership, adaptability, cultural alignment
- Overall Presentation (10%): Confidence, engagement, professionalism, interview presence

SCORING SCALE:
- 9.0-10.0: Exceptional - Top 5% candidate, immediate hire recommendation
- 8.0-8.9: Excellent - Strong candidate, clear hire recommendation  
- 7.0-7.9: Good - Solid candidate, hire with confidence
- 6.0-6.9: Acceptable - Meets basic requirements, conditional hire
- 5.0-5.9: Below Average - Significant concerns, additional screening needed
- Below 5.0: Poor - Does not meet standards, no hire recommendation

Generate evaluation with these sections:
1. **Technical Assessment** - Score: X.X/10 (Detailed analysis of technical competency)
2. **Communication Skills** - Score: X.X/10 (Assessment of presentation and clarity)
3. **Behavioral/Cultural Fit** - Score: X.X/10 (Teamwork, adaptability, cultural alignment)
4. **Overall Presentation** - Score: X.X/10 (Interview presence and professionalism)
5. **Key Strengths** (2-3 specific points with examples)
6. **Areas for Development** (2-3 constructive areas for improvement)
7. **Final Recommendation** (Strong Hire/Hire/Conditional Hire/No Hire with justification)

Be thorough, honest, and constructive. Use specific examples from their responses.
"""
        return prompt
    
    def _extract_scores_from_evaluation(self, evaluation: str) -> Dict[str, float]:
        """Extract scores from evaluation with enhanced parsing"""
        scores = {
            "technical_score": 0.0,
            "communication_score": 0.0,
            "behavioral_score": 0.0,
            "overall_score": 0.0
        }
        
        # Enhanced score extraction patterns
        patterns = {
            "technical_score": r"Technical Assessment.*?Score:\s*(\d+(?:\.\d+)?)/10",
            "communication_score": r"Communication Skills.*?Score:\s*(\d+(?:\.\d+)?)/10",
            "behavioral_score": r"Behavioral.*?Score:\s*(\d+(?:\.\d+)?)/10",
            "overall_score": r"Overall Presentation.*?Score:\s*(\d+(?:\.\d+)?)/10"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, evaluation, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    score = float(match.group(1))
                    scores[key] = min(max(score, 0.0), 10.0)  # Ensure score is between 0-10
                except ValueError:
                    logger.warning(f"⚠️ Could not parse {key} from evaluation")
        
        # Calculate weighted overall score if individual scores are available
        if any(scores.values()):
            weighted_score = (
                scores["technical_score"] * config.EVALUATION_CRITERIA["technical_weight"] +
                scores["communication_score"] * config.EVALUATION_CRITERIA["communication_weight"] +
                scores["behavioral_score"] * config.EVALUATION_CRITERIA["behavioral_weight"] +
                scores["overall_score"] * config.EVALUATION_CRITERIA["overall_presentation"]
            )
            scores["weighted_overall"] = round(weighted_score, 1)
        
        return scores