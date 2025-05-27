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
SAMPLE_RATE = 16000 # Sample rate for audio recording
BLOCK_SIZE = 4096 # Block size for audio processing
SILENCE_THRESHOLD = 0.005 # Threshold for silence detection
SILENCE_DURATION = 3 # Duration of silence to consider end of recording
MAX_RECORDING_DURATION = 15.0 # Maximum recording duration is 15 seconds
MIN_RECORDING_DURATION = 1.0  # Minimum duration to consider a valid recording is 1 second

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
        try:
            self.client = pymongo.MongoClient(
                "mongodb://sa:L%40nc%5Eere%400012@192.168.48.200:27017/?authSource=admin"
            )
            self.db = self.client["test"]
            self.transcripts = self.db["drive"]
            self.interviews = self.db["interviews"]
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.client = None
    
    def save_interview_data(self, session_id: str, conversations: Dict[RoundType, List[ConversationEntry]], evaluation: str, scores: Dict[str, Optional[float]]) -> bool:
        try:
            if not self.client:
                logger.warning("No database connection available")
                return False
                
            # Flatten conversations
            conv_data = {}
            for round_type, conv_list in conversations.items():
                conv_data[round_type.value] = [{
                    "user_input": entry.user_input,
                    "ai_response": entry.ai_response,
                    "timestamp": entry.timestamp
                } for entry in conv_list]
            
            doc = {
                "session_id": session_id,
                "timestamp": time.time(),
                "conversations": conv_data,
                "evaluation": evaluation,
                "scores": scores
            }
            result = self.interviews.insert_one(doc)
            logger.info(f"Interview saved for session {session_id}, id {result.inserted_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving interview data: {e}")
            return False

    def get_technical_content(self) -> str:
        """Get and summarize technical content from MongoDB"""
        try:
            if not self.client:
                return "No technical content available due to database connection issues."
                
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
        # Fixed model name and increased max_tokens for longer responses
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, max_tokens=200)
        self.parser = StrOutputParser()
        
        # Improved prompts for each round
        self.prompts = {
            RoundType.TECHNICAL: PromptTemplate.from_template("""
            You are conducting a technical interview round (20 minutes duration). Your goal is to ask comprehensive technical questions.
            If the candidate goes off-topic, tell them explicitly that they are going off-topic and ask a redirected next technical question.
            
            Technical content for reference:
            {technical_content}
            
            Previous conversation:
            {history}
            
            Current question number: {question_count}
            
            Instructions:
            - Questions 1-4: Ask fundamental technical concepts and basic programming knowledge
            - Questions 5-8: Ask application-based questions and problem-solving scenarios  
            - Questions 9+: Ask advanced problem-solving and system design questions
            - If question count >= 12: Say "Thank you for the technical round. Let's move to the communication round."
            
            Ask specific, practical technical question. Keep it focused and clear.
            
            """),
            
            RoundType.COMMUNICATION: PromptTemplate.from_template("""
            You are testing communication skills in this round (20 minutes duration). Focus on verbal communication, clarity, and presentation abilities.
            If the candidate goes off-topic, tell them explicitly that they are going off-topic and ask a redirected next communication question.
            Previous conversation:
            {history}
            
            Current question number: {question_count}
            
            Instructions:
            - Questions 1-4: Test clarity of explanation and articulation skills
            - Questions 5-8: Test persuasion, storytelling, and explanation skills
            - Questions 9-11: Test confidence, presentation skills, and leadership communication
            - If question count >= 12: Say "Thank you for the communication round. Let's proceed to the HR round."

            Ask question that evaluates verbal communication, confidence, or presentation skills.
            """),
            
            RoundType.HR: PromptTemplate.from_template("""
            You are an HR interviewer conducting behavioral assessment (20 minutes duration). Focus on cultural fit, teamwork, and soft skills.
            If the candidate goes off-topic, tell them explicitly that they are going off-topic and ask a redirected next HR question.
            Previous conversation:
            {history}
            
            Current question number: {question_count}
            
            Instructions:
            - Questions 1-4: Ask about past experiences, teamwork, and collaboration
            - Questions 5-8: Ask situational and behavioral questions (STAR method encouraged)
            - Questions 9-11: Ask about conflict resolution, adaptability, and growth mindset
            - If question count >= 12: Say "Thank you for completing the interview. We'll now prepare your evaluation."
            Ask behavioral or situational question that reveals personality and cultural fit.
            """)
        }

        self.evaluation_prompt = PromptTemplate.from_template("""
        Evaluate this interview performance across all rounds. Provide specific scores and constructive feedback.
        
        Technical Round Conversations:
        {technical_conversation}
        
        Communication Round Conversations:
        {communication_conversation}
        
        HR Round Conversations:
        {hr_conversation}
        
        Provide a structured evaluation with:
        1. Technical competency (2-3 sentences) - Score: X/10
        2. Communication skills (2-3 sentences) - Score: X/10  
        3. Cultural fit and soft skills (2-3 sentences) - Score: X/10
        4. Overall recommendation and areas for improvement (2-3 sentences) - Overall Score: X/10
        
        Keep the evaluation professional, constructive, and under 250 words.
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
        """Improved fallback questions if LLM fails"""
        fallbacks = {
            RoundType.TECHNICAL: [
                "this is a fallback question for technical round.",
                "Check for errors in your code and try again."
                # "Can you explain your experience with databases?",
                # "How do you approach debugging complex issues?",
                # "Let's move to the communication round."
            ],
            RoundType.COMMUNICATION: [
                "this is a fallback question for communication round.",
                "Check for errors in your code and try again."
                # "How would you explain a technical concept to a non-technical person?",
                # "Tell me about a time you had to present to a group.",
                # "Let's proceed to the HR round."
            ],
            RoundType.HR: [
                "this is a fallback question for HR round.",
                "Check for errors in your code and try again."
                # "Tell me about a challenging situation you handled at work.",
                # "How do you handle feedback and criticism?",
                # "Thank you for completing the interview."
            ]
        }
        
        questions = fallbacks[round_type]
        return questions[min(question_count - 1, len(questions) - 1)]
    
    async def evaluate_interview(self, conversations: Dict[RoundType, List[ConversationEntry]]) -> str:
        """Generate evaluation based on all conversations"""
        try:
            def format_round_conversations(conv_list):
                if not conv_list:
                    return "No conversations recorded for this round."
                
                formatted = []
                for entry in conv_list:
                    formatted.append(f"Interviewer: {entry.ai_response}")
                    formatted.append(f"Candidate: {entry.user_input}")
                return "\n".join(formatted)
            
            technical_conv = format_round_conversations(conversations.get(RoundType.TECHNICAL, []))
            communication_conv = format_round_conversations(conversations.get(RoundType.COMMUNICATION, []))
            hr_conv = format_round_conversations(conversations.get(RoundType.HR, []))
            
            chain = self.evaluation_prompt | self.llm | self.parser
            
            evaluation = await chain.ainvoke({
                "technical_conversation": technical_conv,
                "communication_conversation": communication_conv,
                "hr_conversation": hr_conv
            })
            
            return evaluation.strip()
            
        except Exception as e:
            logger.error(f"Evaluation generation error: {e}")
            return "Unable to generate evaluation due to technical issues. Please contact support."

# Score extraction functions
import re

def extract_scores_from_evaluation(text: str) -> Dict[str, Optional[float]]:
    scores = {"technical_score": None, "communication_score": None, "hr_score": None, "overall_score": None}

    # Updated regex patterns to match the evaluation format
    scores["technical_score"] = _extract_score(text, r"Technical competency.*?Score:\s*(\d+(?:\.\d+)?)/10")
    scores["communication_score"] = _extract_score(text, r"Communication skills.*?Score:\s*(\d+(?:\.\d+)?)/10")
    scores["hr_score"] = _extract_score(text, r"Cultural fit.*?Score:\s*(\d+(?:\.\d+)?)/10")
    scores["overall_score"] = _extract_score(text, r"Overall Score:\s*(\d+(?:\.\d+)?)/10")

    return scores

def _extract_score(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

# ========================
# Audio Manager
# ========================

class AudioManager:
    def __init__(self):
        try:
            self.groq_client = Groq()
            logger.info("Audio manager initialized")
        except Exception as e:
            logger.error(f"Audio manager initialization failed: {e}")
            self.groq_client = None
    
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
            if not self.groq_client:
                logger.error("Groq client not available")
                return None
                
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
# Session Manager with Enhanced Debugging
# ========================

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, InterviewSession] = {}
        self.db_manager = DatabaseManager()
        self.llm_manager = LLMManager()
        self.audio_manager = AudioManager()
        logger.info("SessionManager initialized")
    
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
        logger.info(f"Created session {session_id}. Total sessions: {len(self.sessions)}")
        logger.info(f"Session details: Round={session.current_round.value}, State={session.state.value}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """Get session by ID with enhanced debugging"""
        logger.info(f"Looking for session: {session_id}")
        logger.info(f"Available sessions: {list(self.sessions.keys())}")
        
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = time.time()
            logger.info(f"Found session {session_id}. Round: {session.current_round.value}, State: {session.state.value}")
        else:
            logger.error(f"Session {session_id} not found! Available sessions: {list(self.sessions.keys())}")
        
        return session
    
    def list_sessions(self) -> Dict[str, Dict[str, str]]:
        """Debug method to list all active sessions"""
        session_info = {}
        for sid, session in self.sessions.items():
            session_info[sid] = {
                "round": session.current_round.value,
                "state": session.state.value,
                "last_activity": time.ctime(session.last_activity),
                "questions_asked": str(session.questions_asked)
            }
        return session_info
    
    def should_end_round(self, session: InterviewSession) -> bool:
        """Determine if current round should end based on questions asked"""
        current_round = session.current_round
        questions_asked = session.questions_asked[current_round]
        
        # Increased limits for longer sessions
        limits = {
            RoundType.TECHNICAL: 12,
            RoundType.COMMUNICATION: 12,
            RoundType.HR: 12
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
            return "No previous conversation in this round."
        
        # Include last 4 exchanges for better context
        recent_conversations = conversations[-4:]
        
        formatted = []
        for entry in recent_conversations:
            formatted.append(f"Interviewer: {entry.ai_response}")
            formatted.append(f"Candidate: {entry.user_input}")
        
        return "\n".join(formatted)
    
    async def generate_first_question(self, session_id: str) -> Dict[str, Any]:
        """Generate the first question for a round"""
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found in generate_first_question")
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
        logger.info(f"Generated question {question_count} for round {session.current_round.value}")
        
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
            logger.error(f"Session {session_id} not found in process_user_response")
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"Processing user response for session {session_id}, round {session.current_round.value}")
        
        # Record and transcribe audio
        try:
            logger.info("Starting audio recording...")
            audio_file = self.audio_manager.record_audio()
            if not audio_file:
                logger.warning("No audio recorded")
                return {"error": "No audio recorded", "retry": True}
            
            logger.info(f"Audio file recorded: {audio_file}")
            user_text = self.audio_manager.transcribe_audio(audio_file)
            
            # Cleanup audio file
            try:
                os.remove(audio_file)
                logger.info(f"Cleaned up audio file: {audio_file}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup audio file {audio_file}: {cleanup_error}")
            
            if not user_text:
                logger.warning("Could not transcribe audio")
                return {"error": "Could not understand audio", "retry": True}
            
            logger.info(f"User response transcribed: {user_text[:100]}...")
            
        except Exception as audio_error:
            logger.error(f"Audio processing error: {audio_error}")
            return {"error": "Audio processing failed", "retry": True}
        
        # Update session activity
        session.last_activity = time.time()
        
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
            "completing the interview",
            "prepare your evaluation"
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
            logger.error(f"Session {session_id} not found in generate_evaluation")
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Generate evaluation using LLMManager
        evaluation = await self.llm_manager.evaluate_interview(session.conversations)
        scores = extract_scores_from_evaluation(evaluation)
        
        # Save to database
        save_success = self.db_manager.save_interview_data(
            session_id=session_id,
            conversations=session.conversations,
            evaluation=evaluation,
            scores=scores
        )
        
        if not save_success:
            logger.warning(f"Failed to save interview data for session {session_id}")
        
        # Generate analytics
        total_questions = sum(session.questions_asked.values())
        total_responses = sum(len(convs) for convs in session.conversations.values())
        round_durations = {}
        
        # Calculate approximate round durations
        for round_type in RoundType:
            question_count = session.questions_asked[round_type]
            round_durations[round_type.value] = f"{question_count} questions"
        
        return {
            "evaluation": evaluation,
            "scores": scores,
            "analytics": {
                "total_questions": total_questions,
                "total_responses": total_responses,
                "round_breakdown": round_durations
            }
        }
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions (older than 4 hours)"""
        current_time = time.time()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session.last_activity > 14400  # 4 hours
        ]
        
        for sid in expired_sessions:
            self.sessions.pop(sid, None)
            logger.info(f"Cleaned up expired session: {sid}")

# ========================
# FastAPI Application with Lifespan
# ========================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("Interview system starting up...")
    yield
    # Shutdown
    session_manager.cleanup_expired_sessions()
    AudioManager.clean_old_audio_files()
    logger.info("Interview system shut down")

app = FastAPI(title="AI Interview System", lifespan=lifespan)

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
        logger.info("Starting new interview...")
        session_id = session_manager.create_session()
        logger.info(f"Session created: {session_id}")
        
        result = await session_manager.generate_first_question(session_id)
        result["session_id"] = session_id
        
        logger.info(f"First question generated for session {session_id}")
        return result
    except Exception as e:
        logger.error(f"Error starting interview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record_and_respond")
async def record_and_respond(request: RecordRequest):
    """Record user response and provide AI reply"""
    try:
        logger.info(f"Processing response for session: {request.session_id}")
        
        # Check if session exists first
        session = session_manager.get_session(request.session_id)
        if not session:
            logger.error(f"Session {request.session_id} not found in record_and_respond")
            # List all available sessions for debugging
            available_sessions = session_manager.list_sessions()
            logger.error(f"Available sessions: {available_sessions}")
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
        
        result = await session_manager.process_user_response(request.session_id)
        logger.info(f"Response processed successfully for session {request.session_id}")
        return result
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error processing response for session {request.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/start_next_round")
async def start_next_round(session_id: str):
    """Start the next interview round"""
    try:
        logger.info(f"Starting next round for session: {session_id}")
        session = session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found in start_next_round")
            raise HTTPException(status_code=404, detail="Session not found")
        
        session.state = InterviewState.IN_PROGRESS
        result = await session_manager.generate_first_question(session_id)
        logger.info(f"Next round started for session {session_id}")
        return result
    except Exception as e:
        logger.error(f"Error starting next round: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate")
async def get_evaluation(session_id: str):
    """Get final interview evaluation"""
    try:
        logger.info(f"Generating evaluation for session: {session_id}")
        result = await session_manager.generate_evaluation(session_id)
        logger.info(f"Evaluation generated for session {session_id}")
        return result
    except Exception as e:
        logger.error(f"Error generating evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Debug endpoints
@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to list all active sessions"""
    try:
        sessions_info = session_manager.list_sessions()
        return {
            "total_sessions": len(sessions_info),
            "sessions": sessions_info
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return {"error": str(e)}

@app.get("/debug/session/{session_id}")
async def debug_session(session_id: str):
    """Debug endpoint to get detailed session info"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return {"error": "Session not found", "session_id": session_id}
        
        return {
            "session_id": session_id,
            "current_round": session.current_round.value,
            "state": session.state.value,
            "voice": session.voice,
            "questions_asked": session.questions_asked,
            "conversations_count": {
                round_type.value: len(convs) 
                for round_type, convs in session.conversations.items()
            },
            "last_activity": time.ctime(session.last_activity),
            "round_start_time": time.ctime(session.round_start_time)
        }
    except Exception as e:
        logger.error(f"Error in session debug endpoint: {e}")
        return {"error": str(e)}

@app.get("/cleanup")
async def cleanup():
    """Clean up expired sessions and old files"""
    try:
        sessions_before = len(session_manager.sessions)
        session_manager.cleanup_expired_sessions()
        sessions_after = len(session_manager.sessions)
        
        AudioManager.clean_old_audio_files()
        
        return {
            "message": "Cleanup completed",
            "sessions_removed": sessions_before - sessions_after,
            "active_sessions": sessions_after
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(session_manager.sessions),
        "timestamp": time.time()
    }

# Note: Startup/shutdown events are now handled by the lifespan context manager above