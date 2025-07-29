# weekly_interview/core/ai_services.py
"""
UPGRADED AI Services - Daily Standup Style with Enhanced 7-Day Content Processing
Ultra-fast streaming with intelligent multi-summary fragment management
TTS functionality moved to separate tts_processor.py
UTF-8 CLEANED VERSION
"""

import os
import time
import logging
import asyncio
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
import random

from .config import config

logger = logging.getLogger(__name__)

# Import TTS processor from separate file
try:
    from .tts_processor import UltraFastTTSProcessor
except ImportError:
    logger.error("Could not import TTS processor. Make sure tts_processor.py exists.")
    # Create fallback TTS processor
    class UltraFastTTSProcessor:
        async def generate_ultra_fast_stream(self, text: str):
            logger.warning("Using fallback TTS (no audio)")
            yield b'\x00' * 1024  # Silent audio

# =============================================================================
# ENHANCED FRAGMENT PARSING - 7-Day Summary Processing (UNCHANGED)
# =============================================================================

def parse_multi_summaries_into_fragments(summaries: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Enhanced fragment parsing from multiple 7-day summaries with intelligent slicing
    Returns rich, diverse fragments from all summaries combined
    """
    if not summaries:
        return {"General": "No content available"}
    
    logger.info(f"Processing {len(summaries)} summaries for fragment creation")
    
    all_fragments = {}
    fragment_counter = 1
    
    # Process each summary and extract meaningful chunks
    for i, summary_doc in enumerate(summaries, 1):
        summary_text = summary_doc.get("summary", "")
        if not summary_text or len(summary_text.strip()) < config.MIN_CONTENT_LENGTH:
            continue
            
        doc_id = str(summary_doc.get("_id", f"doc_{i}"))[:8]
        
        # Extract structured sections from this summary
        individual_fragments = _extract_structured_sections(summary_text)
        
        if individual_fragments:
            # Add structured fragments with source tracking
            for section_title, content in individual_fragments.items():
                if len(content.strip()) > 100:  # Meaningful content only
                    fragment_key = f"{fragment_counter}. {section_title} (Day {i})"
                    all_fragments[fragment_key] = content
                    fragment_counter += 1
        else:
            # If no structured sections, apply intelligent slicing
            sliced_content = _apply_intelligent_content_slicing(summary_text)
            if sliced_content and len(sliced_content.strip()) > config.MIN_CONTENT_LENGTH:
                fragment_key = f"{fragment_counter}. Technical Work Summary (Day {i})"
                all_fragments[fragment_key] = sliced_content
                fragment_counter += 1
    
    # If we have too many fragments, apply selection strategy
    if len(all_fragments) > config.MAX_INTERVIEW_FRAGMENTS:
        all_fragments = _apply_fragment_selection_strategy(all_fragments)
    
    # Ensure minimum fragments
    if len(all_fragments) < config.MIN_INTERVIEW_FRAGMENTS:
        all_fragments = _ensure_minimum_fragments(summaries, all_fragments)
    
    logger.info(f"Created {len(all_fragments)} interview fragments from {len(summaries)} summaries")
    return all_fragments

def _extract_structured_sections(text: str) -> Dict[str, str]:
    """Extract structured sections from individual summary"""
    patterns = [
        # Numbered sections: 1. 2. 3.
        r'^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.|$)',
        # Bullet points: - * •
        r'^\s*[-*•]\s+(.+?)(?=^\s*[-*•]|$)',
        # Headers: ## ### Topic:
        r'^#+\s+(.+?)(?=^#+|\Z)',
        r'^([A-Z][^:]*):(.+?)(?=^[A-Z][^:]*:|\Z)'
    ]
    
    sections = {}
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            if isinstance(matches[0], tuple):
                # Numbered/titled sections
                for match in matches:
                    if len(match) >= 2:
                        title = match[0] if match[0].isdigit() else match[0]
                        content = match[1].strip().replace('\n', ' ')
                        if len(content) > 50:
                            sections[f"Topic {title}"] = content
            else:
                # Header sections
                for i, match in enumerate(matches, 1):
                    content = match.strip().replace('\n', ' ')
                    if len(content) > 50:
                        sections[f"Section {i}"] = content
            
            if sections:
                break  # Use first successful pattern
    
    return sections

def _apply_intelligent_content_slicing(content: str) -> str:
    """Apply intelligent slicing to preserve meaningful content"""
    if not content:
        return ""
    
    # Target length based on configuration
    target_length = int(len(content) * config.CONTENT_SLICE_FRACTION)
    target_length = max(target_length, config.MIN_CONTENT_LENGTH)
    target_length = min(target_length, len(content))
    
    if target_length >= len(content):
        return content
    
    # Try to find natural break points (sentences)
    sentences = re.split(r'[.!?]+', content)
    if len(sentences) > 1:
        selected_sentences = []
        current_length = 0
        
        # Strategic starting position for variety
        start_idx = random.randint(0, max(0, len(sentences) // 4))
        
        for i in range(start_idx, len(sentences)):
            sentence = sentences[i].strip()
            if sentence and current_length + len(sentence) <= target_length:
                selected_sentences.append(sentence)
                current_length += len(sentence)
            elif selected_sentences:
                break
        
        if selected_sentences:
            return '. '.join(selected_sentences) + '.'
    
    # Fallback: word boundary slicing
    slice_pos = content.rfind(' ', 0, target_length)
    if slice_pos > target_length * 0.8:
        return content[:slice_pos] + '...'
    
    return content[:target_length] + '...'

def _apply_fragment_selection_strategy(fragments: Dict[str, str]) -> Dict[str, str]:
    """Select most diverse and relevant fragments for interview"""
    # Score fragments by diversity and technical relevance
    technical_keywords = [
        'development', 'programming', 'algorithm', 'code', 'system',
        'architecture', 'database', 'api', 'framework', 'implementation',
        'optimization', 'performance', 'security', 'testing', 'deployment'
    ]
    
    scored_fragments = []
    for key, content in fragments.items():
        score = 0
        content_lower = content.lower()
        
        # Technical relevance
        score += sum(2 for keyword in technical_keywords if keyword in content_lower)
        
        # Length bonus
        score += len(content) / 100
        
        # Day diversity bonus (prefer different days)
        day_match = re.search(r'Day (\d+)', key)
        if day_match:
            day_num = int(day_match.group(1))
            score += day_num * 0.5  # Slight preference for later days
        
        scored_fragments.append((key, content, score))
    
    # Sort by score and select top fragments
    scored_fragments.sort(key=lambda x: x[2], reverse=True)
    
    selected_fragments = {}
    for key, content, _ in scored_fragments[:config.MAX_INTERVIEW_FRAGMENTS]:
        selected_fragments[key] = content
    
    return selected_fragments

def _ensure_minimum_fragments(summaries: List[Dict[str, Any]], current_fragments: Dict[str, str]) -> Dict[str, str]:
    """Ensure minimum number of fragments by creating additional ones if needed"""
    if len(current_fragments) >= config.MIN_INTERVIEW_FRAGMENTS:
        return current_fragments
    
    # Create additional fragments from remaining content
    additional_needed = config.MIN_INTERVIEW_FRAGMENTS - len(current_fragments)
    
    for i, summary_doc in enumerate(summaries[:additional_needed], len(current_fragments) + 1):
        summary_text = summary_doc.get("summary", "")
        if summary_text and len(summary_text.strip()) > 50:
            # Create basic fragment
            sliced_content = _apply_intelligent_content_slicing(summary_text)
            if sliced_content:
                current_fragments[f"{i}. Additional Technical Content"] = sliced_content
    
    return current_fragments

# =============================================================================
# DATA MODELS - SIMPLIFIED (Daily Standup Style) (UNCHANGED)
# =============================================================================

class InterviewStage(Enum):
    GREETING = "greeting"
    TECHNICAL = "technical"
    COMMUNICATION = "communication"
    HR = "hr"
    COMPLETE = "complete"
    ERROR = "error"

@dataclass
class ConversationExchange:
    timestamp: float
    stage: InterviewStage
    ai_message: str
    user_response: str = ""
    transcript_quality: float = 0.0
    concept: Optional[str] = None
    is_followup: bool = False

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
    exchanges: List[ConversationExchange] = field(default_factory=list)
    conversation_window: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # Simplified round tracking (keep for interview structure)
    questions_per_round: Dict[str, int] = field(default_factory=lambda: {
        "greeting": 0, "technical": 0, "communication": 0, "hr": 0
    })
    
    # Fragment-based attributes (like daily_standup)
    fragments: Dict[str, str] = field(default_factory=dict)
    fragment_keys: List[str] = field(default_factory=list)
    concept_question_counts: Dict[str, int] = field(default_factory=dict)
    questions_per_concept: int = 2
    current_concept: str = ""
    question_index: int = 0
    followup_questions: int = 0
    
    # Session state
    is_active: bool = True
    websocket: Optional[Any] = field(default=None)
    
    def add_exchange(self, ai_message: str, user_response: str = "", quality: float = 0.0,
                    concept: Optional[str] = None, is_followup: bool = False):
        exchange = ConversationExchange(
            timestamp=time.time(),
            stage=self.current_stage,
            ai_message=ai_message,
            user_response=user_response,
            transcript_quality=quality,
            concept=concept,
            is_followup=is_followup
        )
        self.exchanges.append(exchange)
        self.conversation_window.append(exchange)
        self.last_activity = time.time()
    
    def update_last_response(self, user_response: str, quality: float = 0.0):
        if self.exchanges:
            self.exchanges[-1].user_response = user_response
            self.exchanges[-1].transcript_quality = quality

# =============================================================================
# SHARED CLIENT MANAGER (Same as Daily Standup) (UNCHANGED)
# =============================================================================

class SharedClientManager:
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
            logger.info("Groq client initialized")
        return self._groq_client
    
    @property 
    def openai_client(self) -> openai.OpenAI:
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OPENAI_API_KEY not found in environment variables")
            self._openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        return self._openai_client
    
    @property
    def executor(self):
        return self._executor
    
    async def close_connections(self):
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info("AI client connections closed")

# Global shared client manager
shared_clients = SharedClientManager()

# =============================================================================
# ENHANCED FRAGMENT MANAGER (Daily Standup Style + Interview Rounds) (UNCHANGED)
# =============================================================================

class EnhancedInterviewFragmentManager:
    """Fragment-based interview management with round awareness"""
    
    def __init__(self, client_manager, session_data: InterviewSession):
        self.client_manager = client_manager
        self.session_data = session_data
    
    @property
    def openai_client(self):
        return self.client_manager.openai_client
    
    def initialize_fragments(self, summaries: List[Dict[str, Any]]) -> bool:
        """Initialize fragments from 7-day summaries with intelligent processing"""
        try:
            # Parse multi-summaries into interview fragments
            self.session_data.fragments = parse_multi_summaries_into_fragments(summaries)
            self.session_data.fragment_keys = list(self.session_data.fragments.keys())
            
            # Initialize concept question counts
            self.session_data.concept_question_counts = {
                key: 0 for key in self.session_data.fragment_keys
            }
            
            # Calculate dynamic questions per concept (interview-aware)
            total_interview_questions = config.QUESTIONS_PER_ROUND * len(config.ROUND_NAMES[1:])  # Exclude greeting
            self.session_data.questions_per_concept = max(
                config.MIN_QUESTIONS_PER_CONCEPT,
                min(config.MAX_QUESTIONS_PER_CONCEPT,
                    total_interview_questions // len(self.session_data.fragment_keys) 
                    if self.session_data.fragment_keys else 1)
            )
            
            logger.info(f"Initialized {len(self.session_data.fragment_keys)} fragments from 7-day summaries, "
                       f"target {self.session_data.questions_per_concept} questions per concept")
            return True
            
        except Exception as e:
            logger.error(f"Fragment initialization failed: {e}")
            raise Exception(f"Fragment initialization failed: {e}")
    
    def get_active_fragment_for_round(self, round_stage: InterviewStage) -> Tuple[str, str]:
        """Get active fragment considering interview round context"""
        if not self.session_data.fragment_keys:
            return "General", self.session_data.fragments.get("General", "No content available")
        
        # Round-aware fragment selection
        if round_stage == InterviewStage.TECHNICAL:
            # Prioritize technical fragments
            tech_fragments = [k for k in self.session_data.fragment_keys 
                            if any(tech_word in k.lower() for tech_word in 
                                  ['technical', 'development', 'code', 'system', 'algorithm'])]
            
            if tech_fragments:
                selected_fragments = tech_fragments
            else:
                selected_fragments = self.session_data.fragment_keys
                
        elif round_stage == InterviewStage.COMMUNICATION:
            # Prioritize communication/presentation fragments
            comm_fragments = [k for k in self.session_data.fragment_keys 
                            if any(comm_word in k.lower() for comm_word in 
                                  ['presentation', 'documentation', 'meeting', 'communication'])]
            
            if comm_fragments:
                selected_fragments = comm_fragments
            else:
                selected_fragments = self.session_data.fragment_keys
                
        elif round_stage == InterviewStage.HR:
            # Prioritize experience/project fragments
            exp_fragments = [k for k in self.session_data.fragment_keys 
                           if any(exp_word in k.lower() for exp_word in 
                                 ['project', 'experience', 'challenge', 'team', 'collaboration'])]
            
            if exp_fragments:
                selected_fragments = exp_fragments
            else:
                selected_fragments = self.session_data.fragment_keys
        else:
            selected_fragments = self.session_data.fragment_keys
        
        # Smart selection from filtered fragments
        min_questions = min(self.session_data.concept_question_counts.get(k, 0) for k in selected_fragments)
        underutilized_concepts = [
            concept for concept in selected_fragments 
            if self.session_data.concept_question_counts.get(concept, 0) == min_questions
        ]
        
        if underutilized_concepts:
            for concept in selected_fragments:
                if concept in underutilized_concepts:
                    return concept, self.session_data.fragments[concept]
        
        # Fallback to first available
        selected_concept = selected_fragments[0]
        return selected_concept, self.session_data.fragments[selected_concept]
    
    def should_continue_round(self, current_stage: InterviewStage) -> bool:
        """Determine if current round should continue"""
        stage_name = current_stage.value
        questions_in_stage = self.session_data.questions_per_round.get(stage_name, 0)
        
        # Stage-specific continuation logic
        if current_stage == InterviewStage.GREETING:
            return questions_in_stage < 2
        elif current_stage in [InterviewStage.TECHNICAL, InterviewStage.COMMUNICATION, InterviewStage.HR]:
            return questions_in_stage < config.QUESTIONS_PER_ROUND
        
        return False
    
    def should_continue_interview(self) -> bool:
        """Determine if entire interview should continue"""
        # Check if we've completed all rounds
        completed_rounds = sum(1 for stage in ["technical", "communication", "hr"] 
                             if self.session_data.questions_per_round.get(stage, 0) >= config.QUESTIONS_PER_ROUND)
        
        if completed_rounds >= 3:  # All main rounds completed
            return False
        
        # Check for fragment coverage if in active round
        if self.session_data.current_stage != InterviewStage.GREETING:
            uncovered_concepts = [
                concept for concept, count in self.session_data.concept_question_counts.items() 
                if count == 0
            ]
            
            # If too many uncovered concepts in current stage, continue
            if len(uncovered_concepts) > len(self.session_data.fragment_keys) * 0.5:
                return True
        
        return completed_rounds < 3
    
    def get_concept_conversation_history(self, concept: str, window_size: int = 3) -> str:
        """Get conversation history for specific concept in current round"""
        current_stage = self.session_data.current_stage.value
        
        entries = [
            exchange for exchange in reversed(self.session_data.exchanges)
            if (exchange.concept == concept and 
                exchange.stage.value == current_stage and 
                exchange.user_response)
        ]
        last_entries = list(reversed(entries[:window_size]))

        history = []
        for entry in last_entries:
            q = f"Q: {entry.ai_message}"
            a = f"A: {entry.user_response}"
            history.append(f"{q}\n{a}")
        return "\n\n".join(history)
    
    def add_question(self, question: str, concept: str = None, is_followup: bool = False):
        """Add question with round and concept tracking"""
        # Track round usage
        stage_name = self.session_data.current_stage.value
        if stage_name in self.session_data.questions_per_round:
            self.session_data.questions_per_round[stage_name] += 1
        
        # Track concept usage (skip for greeting)
        if concept and concept in self.session_data.concept_question_counts and stage_name != 'greeting':
            self.session_data.concept_question_counts[concept] += 1
        
        # Track follow-ups
        if is_followup and stage_name != 'greeting':
            self.session_data.followup_questions += 1
        
        self.session_data.current_concept = concept
        
        if stage_name != 'greeting':
            self.session_data.question_index += 1
        
        logger.info(f"Added question to {stage_name} round (concept: '{concept}', followup: {is_followup})")

# =============================================================================
# ULTRA-FAST AUDIO PROCESSING (Identical to Daily Standup) (UNCHANGED)
# =============================================================================

class OptimizedAudioProcessor:
    def __init__(self, client_manager: SharedClientManager):
        self.client_manager = client_manager
    
    @property
    def groq_client(self):
        return self.client_manager.groq_client
    
    async def transcribe_audio_fast(self, audio_data: bytes) -> Tuple[str, float]:
        """Ultra-fast transcription (identical to daily_standup)"""
        try:
            audio_size = len(audio_data)
            logger.info(f"Transcribing {audio_size} bytes of audio")
            
            if audio_size < 50:
                raise Exception(f"Audio data too small for transcription ({audio_size} bytes)")
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_transcribe,
                audio_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Fast transcription error: {e}")
            raise Exception(f"Transcription failed: {e}")
    
    def _sync_transcribe(self, audio_data: bytes) -> Tuple[str, float]:
        """Synchronous transcription for thread pool"""
        try:
            audio_size = len(audio_data)
            temp_file = config.TEMP_DIR / f"audio_{int(time.time() * 1000000)}.webm"
            
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            logger.info(f"Sending {audio_size} bytes to Groq for transcription")
            
            with open(temp_file, "rb") as file:
                result = self.groq_client.audio.transcriptions.create(
                    file=(temp_file.name, file.read()),
                    model=config.GROQ_MODEL,
                    response_format="verbose_json",
                    prompt="Please transcribe this interview response clearly and accurately."
                )
            
            try:
                os.remove(temp_file)
            except:
                pass
            
            transcript = result.text.strip() if result.text else ""
            
            if not transcript:
                logger.warning(f"Groq returned empty transcript for {audio_size} bytes")
                return "", 0.0
            
            # Quality assessment
            quality = min(len(transcript) / 30, 1.0)
            if hasattr(result, 'segments') and result.segments:
                confidences = [seg.get('confidence', 0.8) for seg in result.segments[:3]]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    quality = (quality + avg_confidence) / 2
            
            logger.info(f"Transcription: '{transcript}' (quality: {quality:.2f})")
            return transcript, quality
            
        except Exception as e:
            logger.error(f"Sync transcription error: {e}")
            raise Exception(f"Groq transcription failed: {e}")

# =============================================================================
# ENHANCED CONVERSATION MANAGER (Daily Standup Style + Interview Rounds) (UNCHANGED)
# =============================================================================

class OptimizedConversationManager:
    def __init__(self, client_manager: SharedClientManager):
        self.client_manager = client_manager
    
    @property
    def openai_client(self):
        return self.client_manager.openai_client
    
    async def generate_fast_response(self, session_data: InterviewSession, user_input: str) -> str:
        """Generate ultra-fast AI responses with round-aware context"""
        try:
            if session_data.current_stage == InterviewStage.GREETING:
                return await self._generate_greeting_response(session_data, user_input)
            elif session_data.current_stage in [InterviewStage.TECHNICAL, InterviewStage.COMMUNICATION, InterviewStage.HR]:
                return await self._generate_round_response(session_data, user_input)
            else:
                return await self._generate_conclusion_response(session_data, user_input)
                
        except Exception as e:
            logger.error(f"Fast response generation error: {e}")
            raise Exception(f"AI response generation failed: {e}")
    
    async def _generate_greeting_response(self, session_data: InterviewSession, user_input: str) -> str:
        """Generate greeting responses (similar to daily_standup)"""
        if session_data.questions_per_round["greeting"] == 0:
            return f"Hello {session_data.student_name}! Welcome to your mock interview. I'm excited to learn about your technical skills and experience. How are you feeling today?"
        else:
            # Transition to technical round
            prompt = f"""You are a professional interviewer. The candidate just responded: "{user_input}"

Acknowledge their response warmly and transition smoothly to the technical assessment round.

Keep it brief and professional. Focus on starting the technical discussion based on their recent work.

Response:"""

            return await self._call_openai_fast(prompt, 200)
    
    async def _generate_round_response(self, session_data: InterviewSession, user_input: str) -> str:
        """Generate round-specific responses using fragment-based approach"""
        if not hasattr(session_data, 'fragment_manager') or not session_data.fragment_manager:
            raise Exception("Fragment manager not initialized")
        
        fragment_manager = session_data.fragment_manager
        
        # Get current fragment for this round
        current_concept_title, current_concept_content = fragment_manager.get_active_fragment_for_round(session_data.current_stage)
        
        # Get conversation history for this concept
        history = fragment_manager.get_concept_conversation_history(current_concept_title)
        
        # Get the last question asked
        last_question = session_data.exchanges[-1].ai_message if session_data.exchanges else ""
        
        # Round-specific prompt generation
        round_context = self._get_round_specific_context(session_data.current_stage)
        
        prompt = f"""You are conducting a {session_data.current_stage.value} interview round. Generate the next appropriate question based on the conversation flow.

ROUND FOCUS: {round_context}

CURRENT CONCEPT CONTEXT:
Title: {current_concept_title}
Content: {current_concept_content[:800]}...

CONVERSATION HISTORY FOR THIS CONCEPT:
{history}

CANDIDATE'S LAST RESPONSE: "{user_input}"
PREVIOUS QUESTION: "{last_question}"

CURRENT QUESTION NUMBER IN {session_data.current_stage.value.upper()} ROUND: {session_data.questions_per_round[session_data.current_stage.value] + 1}

INSTRUCTIONS:
1. Analyze if the candidate's response shows understanding (YES/NO)
2. If YES and we should move to next concept, suggest a new concept-based question
3. If NO or needs elaboration, generate a follow-up question for the same concept
4. Keep questions challenging but fair for {session_data.current_stage.value} assessment
5. Make it conversational and professional

FORMAT:
UNDERSTANDING: [YES/NO]
CONCEPT: [{current_concept_title}]
QUESTION: [Your next question]"""

        return await self._process_round_response(session_data, prompt, current_concept_title, fragment_manager)
    
    async def _process_round_response(self, session_data: InterviewSession, prompt: str, 
                                     current_concept_title: str, fragment_manager) -> str:
        """Process round response with fragment logic"""
        
        response = await self._call_openai_fast(prompt, 400)
        
        # Parse the LLM response
        parsed_response = self._parse_llm_response(response, ["UNDERSTANDING", "CONCEPT", "QUESTION"])
        
        understanding = parsed_response.get("understanding", "NO").upper()
        next_question = parsed_response.get("question", "Can you elaborate more on that?")
        suggested_concept = parsed_response.get("concept", current_concept_title)
        
        # Determine if this is a follow-up question
        is_followup = (understanding == "NO" and suggested_concept == current_concept_title)
        
        # Check if round should continue
        if understanding == "YES" and not fragment_manager.should_continue_round(session_data.current_stage):
            # Move to next round
            session_data.current_stage = self._get_next_stage(session_data.current_stage)
            
            if session_data.current_stage == InterviewStage.COMPLETE:
                return "Thank you for completing the interview! You've done an excellent job across all rounds. Your evaluation will be prepared shortly."
            
            # Generate transition message
            transition_message = self._generate_round_transition(session_data.current_stage, next_question)
            fragment_manager.add_question(transition_message, suggested_concept, is_followup)
            return transition_message
        
        # Continue current round
        fragment_manager.add_question(next_question, suggested_concept, is_followup)
        return next_question
    
    def _get_round_specific_context(self, stage: InterviewStage) -> str:
        """Get context for specific interview round"""
        contexts = {
            InterviewStage.TECHNICAL: "Technical skills, problem-solving, coding concepts, system design, and implementation details",
            InterviewStage.COMMUNICATION: "Communication skills, explanation ability, presentation skills, and clarity of expression",
            InterviewStage.HR: "Behavioral questions, teamwork, leadership, cultural fit, and past experiences"
        }
        return contexts.get(stage, "General interview assessment")
    
    def _get_next_stage(self, current_stage: InterviewStage) -> InterviewStage:
        """Get next interview stage"""
        stage_progression = {
            InterviewStage.GREETING: InterviewStage.TECHNICAL,
            InterviewStage.TECHNICAL: InterviewStage.COMMUNICATION,
            InterviewStage.COMMUNICATION: InterviewStage.HR,
            InterviewStage.HR: InterviewStage.COMPLETE
        }
        return stage_progression.get(current_stage, InterviewStage.COMPLETE)
    
    def _generate_round_transition(self, next_stage: InterviewStage, base_question: str) -> str:
        """Generate smooth transition between rounds"""
        transitions = {
            InterviewStage.TECHNICAL: f"Great! Now let's dive into the technical assessment. {base_question}",
            InterviewStage.COMMUNICATION: f"Excellent technical discussion! Now I'd like to assess your communication skills. {base_question}",
            InterviewStage.HR: f"Wonderful! For our final round, let's discuss your experiences and cultural fit. {base_question}"
        }
        return transitions.get(next_stage, base_question)
    
    def _parse_llm_response(self, response: str, keys: List[str]) -> Dict[str, str]:
        """Parse structured responses from the LLM"""
        result = {}
        lines = response.strip().split('\n')
        for line in lines:
            for key in keys:
                prefix = f"{key.upper()}:"
                if line.upper().startswith(prefix):
                    result[key.lower()] = line[len(prefix):].strip()
                    break
        return result
    
    async def _generate_conclusion_response(self, session_data: InterviewSession, user_input: str) -> str:
        """Generate conclusion response"""
        return "Thank you for completing all rounds of the interview. Your comprehensive evaluation will be prepared shortly."
    
    async def _call_openai_fast(self, prompt: str, max_tokens: int = None) -> str:
        """Fast OpenAI call with optimization"""
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
            logger.error(f"OpenAI API call failed: {e}")
            raise Exception(f"OpenAI API failed: {e}")
    
    async def generate_fast_evaluation(self, session_data: InterviewSession) -> Tuple[str, Dict[str, float]]:
        """Generate comprehensive interview evaluation"""
        try:
            # Separate exchanges by round
            round_exchanges = {stage.value: [] for stage in InterviewStage}
            
            for exchange in session_data.exchanges:
                if exchange.user_response:
                    round_exchanges[exchange.stage.value].append(exchange)
            
            # Calculate session analytics
            session_analytics = {
                "total_duration_minutes": round((time.time() - session_data.created_at) / 60, 1),
                "questions_per_round": dict(session_data.questions_per_round),
                "total_concepts_covered": len([c for c, count in session_data.concept_question_counts.items() if count > 0]),
                "followup_questions": session_data.followup_questions,
                "fragment_coverage": round(
                    (len([c for c, count in session_data.concept_question_counts.items() if count > 0]) 
                     / len(session_data.fragment_keys) * 100) 
                    if session_data.fragment_keys else 0, 1
                )
            }
            
            # Generate evaluation prompt
            evaluation_prompt = self._create_comprehensive_evaluation_prompt(
                round_exchanges, session_analytics, session_data.fragment_keys
            )
            
            evaluation = await self._call_openai_fast(evaluation_prompt, 1000)
            
            # Extract scores
            scores = self._extract_scores_from_evaluation(evaluation)
            
            logger.info(f"Evaluation generated for {session_data.test_id}")
            return evaluation, scores
            
        except Exception as e:
            logger.error(f"Evaluation generation failed: {e}")
            raise Exception(f"Evaluation generation failed: {e}")
    
    def _create_comprehensive_evaluation_prompt(self, round_exchanges: Dict, 
                                               analytics: Dict, fragment_keys: List[str]) -> str:
        """Create comprehensive evaluation prompt"""
        
        def format_round_exchanges(exchanges: List) -> str:
            if not exchanges:
                return "No meaningful exchanges in this round."
            
            formatted = []
            for ex in exchanges[:8]:  # Limit for prompt size
                formatted.append(f"Q: {ex.ai_message}")
                if ex.user_response:
                    formatted.append(f"A: {ex.user_response}")
            return "\n".join(formatted)
        
        prompt = f"""You are evaluating a comprehensive mock interview with STRICT professional standards.

INTERVIEW ANALYTICS:
- Duration: {analytics['total_duration_minutes']} minutes
- Questions per Round: {analytics['questions_per_round']}
- Concepts Covered: {analytics['total_concepts_covered']} from {len(fragment_keys)} available
- Fragment Coverage: {analytics['fragment_coverage']}%
- Follow-up Questions: {analytics['followup_questions']}

TECHNICAL ROUND:
{format_round_exchanges(round_exchanges.get('technical', []))}

COMMUNICATION ROUND:
{format_round_exchanges(round_exchanges.get('communication', []))}

HR/BEHAVIORAL ROUND:
{format_round_exchanges(round_exchanges.get('hr', []))}

EVALUATION CRITERIA:
- Technical Assessment (35%): Knowledge depth, problem-solving, accuracy
- Communication Skills (30%): Clarity, presentation, articulation
- Behavioral/Cultural Fit (25%): Teamwork, leadership, adaptability  
- Overall Presentation (10%): Confidence, engagement, professionalism

SCORING SCALE (STRICT):
- 9.0-10.0: Exceptional - Top 5% candidate
- 8.0-8.9: Excellent - Strong hire recommendation
- 7.0-7.9: Good - Solid candidate
- 6.0-6.9: Acceptable - Meets basic requirements
- 5.0-5.9: Below Average - Needs improvement
- Below 5.0: Poor - Does not meet standards

Generate evaluation with these sections:
1. **Technical Assessment** - Score: X.X/10
2. **Communication Skills** - Score: X.X/10  
3. **Behavioral/Cultural Fit** - Score: X.X/10
4. **Overall Presentation** - Score: X.X/10
5. **Key Strengths** (2-3 specific points)
6. **Areas for Development** (2-3 areas)
7. **Final Recommendation** (Strong Hire/Hire/Conditional/No Hire)

Be thorough, honest, and constructive."""
        
        return prompt
    
    def _extract_scores_from_evaluation(self, evaluation: str) -> Dict[str, float]:
        """Extract scores from evaluation text"""
        scores = {
            "technical_score": 0.0,
            "communication_score": 0.0,
            "behavioral_score": 0.0,
            "overall_score": 0.0
        }
        
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
                    scores[key] = min(max(score, 0.0), 10.0)
                except ValueError:
                    logger.warning(f"Could not parse {key} from evaluation")
        
        # Calculate weighted overall score
        if any(scores.values()):
            weighted_score = (
                scores["technical_score"] * 0.35 +
                scores["communication_score"] * 0.30 +
                scores["behavioral_score"] * 0.25 +
                scores["overall_score"] * 0.10
            )
            scores["weighted_overall"] = round(weighted_score, 1)
        
        return scores