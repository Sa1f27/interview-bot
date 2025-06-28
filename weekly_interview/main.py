import os
import time
import uuid
import random
import logging
import asyncio
import textwrap
import subprocess
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import io

import pyodbc
import pymongo
from urllib.parse import quote_plus
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from fastapi.responses import StreamingResponse

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import edge_tts
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================
# Constants and Configuration
# ========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Audio and timing constants
TTS_SPEED = 1.2
INACTIVITY_TIMEOUT = 3600  # 1 hour
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 2000  # 2 seconds
MAX_RECORDING_DURATION = 15000  # 15 seconds
PREPARATION_TIME = 5000  # 5 seconds

# Interview configuration
QUESTIONS_PER_ROUND = 6  # Target questions per round
MIN_QUESTIONS_PER_ROUND = 4  # Minimum before allowing transition
MAX_QUESTIONS_PER_ROUND = 8  # Maximum to prevent endless rounds

# Environment configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

# ========================
# Database Configuration
# ========================

# SQL Server connection
DB_CONFIG = {
    "DRIVER": "ODBC Driver 17 for SQL Server",
    "SERVER": "192.168.48.200",
    "DATABASE": "SuperDB",
    "UID": "sa",
    "PWD": "Welcome@123",
}

CONNECTION_STRING = (
    f"DRIVER={{{DB_CONFIG['DRIVER']}}};"
    f"SERVER={DB_CONFIG['SERVER']};"
    f"DATABASE={DB_CONFIG['DATABASE']};"
    f"UID={DB_CONFIG['UID']};"
    f"PWD={DB_CONFIG['PWD']}"
)

# MongoDB configuration
MONGO_USER = "LanTech"
MONGO_PASS = "L@nc^ere@0012"
MONGO_HOST = "192.168.48.201:27017"
MONGO_DB_NAME = "Api-1"
MONGO_AUTH_SOURCE = "admin"

# ========================
# Data Models and Enums
# ========================

class RoundType(Enum):
    GREETING = "Greeting"
    TECHNICAL = "Technical"
    COMMUNICATION = "Communication"
    HR = "HR"

class InterviewState(Enum):
    NOT_STARTED = "not_started"
    GREETING = "greeting"
    IN_PROGRESS = "in_progress"
    ROUND_TRANSITION = "round_transition"
    INTERVIEW_COMPLETE = "interview_complete"

@dataclass
class ConversationEntry:
    question: str
    answer: str = None
    round_type: RoundType = None
    is_followup: bool = False
    concept: str = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class InterviewSession:
    test_id: str
    current_round: RoundType
    state: InterviewState
    voice: str
    conversation_log: List[ConversationEntry] = field(default_factory=list)
    round_start_time: float = field(default_factory=time.time)
    questions_per_round: Dict[RoundType, int] = field(default_factory=lambda: {r: 0 for r in RoundType})
    followup_questions: int = 0
    last_activity: float = field(default_factory=time.time)
    
    # Fragment-based features
    technical_fragments: Dict[str, str] = field(default_factory=dict)
    fragment_coverage: Dict[str, int] = field(default_factory=dict)
    current_concept: str = None

class RecordRequest(BaseModel):
    test_id: str

class InterviewResponse(BaseModel):
    test_id: str = None
    question: str
    audio_path: str
    round: str
    duration_sec: int = None

class ConversationResponse(BaseModel):
    ended: bool = False
    response: str
    audio_path: str
    round_complete: bool = False
    interview_complete: bool = False
    next_round: str = None
    current_round: str = None

class EvaluationResponse(BaseModel):
    evaluation: str
    scores: Dict[str, Any]
    analytics: Dict[str, Any]
    pdf_url: Optional[str] = None

# ========================
# Database Managers
# ========================

def get_db_connection():
    """Get SQL Server database connection"""
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        logger.info("Successfully connected to SQL Server")
        return conn
    except pyodbc.Error as e:
        logger.error(f"SQL Server connection error: {e}")
        raise

def fetch_random_student_info():
    """Fetch random student info from SQL Server"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Fetch student records
        cursor.execute("SELECT ID, First_Name, Last_Name FROM tbl_Student WHERE ID IS NOT NULL AND First_Name IS NOT NULL AND Last_Name IS NOT NULL")
        student_records = cursor.fetchall()
        
        if not student_records:
            logger.warning("No valid student data found")
            return None

        # Fetch session IDs
        cursor.execute("SELECT DISTINCT Session_ID FROM tbl_Session WHERE Session_ID IS NOT NULL")
        session_rows = cursor.fetchall()
        session_ids = [row[0] for row in session_rows]
        
        cursor.close()
        conn.close()

        # Select random student
        selected_student = random.choice(student_records)
        student_id = selected_student[0]
        first_name = selected_student[1]
        last_name = selected_student[2]
        
        return (
            student_id,
            first_name,
            last_name,
            random.choice(session_ids) if session_ids else None
        )
    except Exception as e:
        logger.error(f"Error fetching student info: {e}")
        return None

class DatabaseManager:
    """Enhanced MongoDB database manager"""
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(
                f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASS)}@{MONGO_HOST}/{MONGO_DB_NAME}?authSource={MONGO_AUTH_SOURCE}"
            )
            self.db = self.client[MONGO_DB_NAME]
            self.transcripts = self.db["original-1"]
            self.interviews = self.db["interview_results-1"]
            logger.info("MongoDB connection established")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            self.client = None
    
    def get_latest_technical_content(self) -> str:
        """Get latest technical content summary"""
        try:
            if not self.client:
                return "No technical content available due to database connection issues."
                
            doc = self.transcripts.find_one(
                {"summary": {"$exists": True, "$ne": None, "$ne": ""}},
                sort=[("timestamp", -1)]
            )
            
            if doc and doc.get("summary"):
                return doc["summary"]
            
            # Fallback without sorting
            doc = self.transcripts.find_one({"summary": {"$exists": True, "$ne": None, "$ne": ""}})
            
            if doc and doc.get("summary"):
                return doc["summary"]
                
            raise ValueError("No technical content found")
            
        except Exception as e:
            logger.error(f"Error fetching technical content: {e}")
            return "Technical content unavailable due to database issues."
    
    def save_interview_data(self, test_id: str, session: InterviewSession, evaluation: str, scores: Dict[str, Any]) -> bool:
        """Save enhanced interview data with analytics"""
        try:
            if not self.client:
                logger.warning("No database connection available")
                return False
            
            # Fetch student info
            student_info = fetch_random_student_info()
            if not student_info:
                logger.error("Failed to fetch student info")
                return False
                
            student_id, first_name, last_name, session_id = student_info
            name = f"{first_name} {last_name}" if first_name and last_name else "Unknown Student"
            
            # Extract overall score
            score_match = re.search(r'Overall Score:\s*(\d+(?:\.\d+)?)\s*/\s*10', evaluation)
            extracted_score = float(score_match.group(1)) if score_match else None
            
            # Convert conversation log
            conversation_data = []
            for entry in session.conversation_log:
                conversation_data.append({
                    "question": entry.question,
                    "answer": entry.answer,
                    "round_type": entry.round_type.value if entry.round_type else None,
                    "is_followup": entry.is_followup,
                    "concept": entry.concept,
                    "timestamp": entry.timestamp
                })
            
            # Enhanced analytics
            document = {
                "test_id": test_id,
                "Student_ID": student_id,
                "name": name,
                "session_id": session_id,
                "timestamp": time.time(),
                "conversation_log": conversation_data,
                "evaluation": evaluation,
                "score": extracted_score,
                "scores": scores,
                "interview_analytics": {
                    "total_questions": len(session.conversation_log),
                    "questions_per_round": dict(session.questions_per_round),
                    "followup_questions": session.followup_questions,
                    "main_questions": len(session.conversation_log) - session.followup_questions,
                    "technical_fragments_covered": len(session.fragment_coverage),
                    "fragment_coverage": dict(session.fragment_coverage)
                }
            }
            
            result = self.interviews.insert_one(document)
            logger.info(f"Interview saved: test_id={test_id}, student={name}, score={extracted_score}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving interview data: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()

# ========================
# Fragment Parser for Technical Content
# ========================

def parse_technical_fragments(content: str) -> Dict[str, str]:
    """Parse technical content into concept fragments"""
    if not content or not content.strip():
        return {"General Programming": content or "Basic programming concepts"}
    
    lines = content.strip().split('\n')
    section_pattern = re.compile(r'^\s*(\d+)\.\s+(.+)')
    
    fragments = {}
    current_section = None
    current_content = []
    
    for line in lines:
        match = section_pattern.match(line)
        
        if match:
            # Save previous section
            if current_section and current_content:
                fragments[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            section_num = match.group(1)
            section_title = match.group(2).strip()
            current_section = f"{section_num}. {section_title}"
            current_content = [line]
        else:
            if current_section:
                current_content.append(line)
            else:
                # Content before any numbered section
                if "Introduction" not in fragments:
                    fragments["Introduction"] = line
                else:
                    fragments["Introduction"] += '\n' + line
    
    # Save last section
    if current_section and current_content:
        fragments[current_section] = '\n'.join(current_content).strip()
    
    # Fallback if no sections found
    if not fragments:
        fragments["General Programming"] = content
    
    logger.info(f"Parsed {len(fragments)} technical fragments: {list(fragments.keys())}")
    return fragments

# ========================
# Enhanced LLM Manager
# ========================

class LLMManager:
    """Enhanced LLM manager with adaptive questioning"""
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.8)
        self.parser = StrOutputParser()
        
        # Enhanced prompts with greeting and transitions
        self.greeting_prompt = PromptTemplate.from_template("""
        You are a professional and friendly AI interviewer conducting a structured interview. Your goal is to create a welcoming, comfortable environment.

        **Your Task:**
        Start with a warm, professional greeting that puts the candidate at ease. Use professional but friendly language.
        Examples: "Hello! Welcome to today's interview. I'm excited to learn more about you." or "Good day! Thank you for joining us today."
        Do NOT ask technical questions yet - this is just to make them feel welcome and explain the process.

        **Guidelines:**
        - Use professional, clear English
        - Be warm and encouraging
        - Keep it conversational but professional
        - Briefly explain the interview structure (Technical → Communication → HR rounds)

        **Output Format:**
        CONCEPT: greeting
        QUESTION: [Your warm, professional greeting and process explanation]
        """)
        
        self.technical_prompt = PromptTemplate.from_template("""
        You are conducting the Technical round of a professional interview. Focus on assessing technical knowledge and problem-solving skills.

        **Current Technical Concept:** {current_concept}
        **Concept Details:** {concept_content}
        **Recent Conversation History:** {history}
        **Previous Question:** {previous_question}
        **Candidate's Response:** {user_response}
        **Question Number:** {question_count} (Questions asked for this concept: {questions_for_concept})

        **Your Task:**

        **IF CURRENT CONCEPT IS 'greeting':**
        The candidate responded to your greeting. Now transition to technical assessment:
        1. Acknowledge their response professionally
        2. Introduce the technical round
        3. Ask your first technical question based on the concept content
        4. Set UNDERSTANDING to YES

        **OTHERWISE:**
        Assess the candidate's technical understanding:

        1. **If they show good technical understanding:** Give brief positive feedback and ask a related follow-up question about the SAME concept, or move to practical application.

        2. **If they struggle:** Be supportive and offer a simpler question about the same concept or provide gentle guidance.

        3. **If they demonstrate mastery:** You can signal readiness for next concept by setting UNDERSTANDING to YES.

        **After {MAX_QUESTIONS_PER_ROUND} questions total:** Say "Thank you for the technical assessment. Let's move to evaluate your communication skills."

        **Guidelines:**
        - Focus ONLY on the current technical concept
        - Ask specific, practical questions
        - Assess depth of understanding
        - Maintain professional interview tone

        **Output Format:**
        UNDERSTANDING: [YES if ready for next concept/round, NO if staying with current concept]
        CONCEPT: [{current_concept} or specify new concept if transitioning]
        QUESTION: [Your professional technical question]
        """)
        
        self.communication_prompt = PromptTemplate.from_template("""
        You are conducting the Communication round of a professional interview. Focus on verbal communication, presentation skills, and clarity.

        **Recent Conversation History:** {history}
        **Previous Question:** {previous_question}
        **Candidate's Response:** {user_response}
        **Question Number:** {question_count}

        **Your Task:**
        Evaluate communication skills through questions about:
        - Explanation and articulation abilities
        - Presentation and storytelling skills
        - Persuasion and confidence
        - Professional communication style

        **Question Guidelines by Range:**
        - Questions 1-2: Test clarity of explanation and articulation
        - Questions 3-4: Test persuasion, storytelling, and presentation skills
        - Questions 5-6: Test confidence and leadership communication

        **After {MAX_QUESTIONS_PER_ROUND} questions:** Say "Thank you for the communication round. Let's proceed to the final HR assessment."

        **Assessment Strategy:**
        1. **Good communication:** Give positive feedback and ask more challenging communication scenarios
        2. **Needs improvement:** Provide supportive guidance and ask simpler communication questions
        3. **Excellent skills:** Explore advanced communication scenarios

        **Output Format:**
        UNDERSTANDING: [YES if ready for HR round, NO if continuing communication assessment]
        CONCEPT: communication_skills
        QUESTION: [Your professional communication question]
        """)
        
        self.hr_prompt = PromptTemplate.from_template("""
        You are conducting the HR round of a professional interview. Focus on behavioral assessment, cultural fit, and soft skills.

        **Recent Conversation History:** {history}
        **Previous Question:** {previous_question}
        **Candidate's Response:** {user_response}
        **Question Number:** {question_count}

        **Your Task:**
        Assess cultural fit and soft skills through behavioral questions:
        - Past experiences and teamwork
        - Situational and behavioral scenarios (STAR method)
        - Conflict resolution and adaptability
        - Growth mindset and leadership potential

        **Question Guidelines by Range:**
        - Questions 1-2: Past experiences, teamwork, and collaboration
        - Questions 3-4: Situational and behavioral questions (encourage STAR method)
        - Questions 5-6: Conflict resolution, adaptability, and growth mindset

        **After {MAX_QUESTIONS_PER_ROUND} questions:** Say "Thank you for completing the interview. We'll now prepare your comprehensive evaluation."

        **Assessment Strategy:**
        1. **Strong cultural fit:** Explore leadership scenarios and advanced behavioral questions
        2. **Some concerns:** Ask clarifying questions to better understand motivations
        3. **Excellent fit:** Focus on growth potential and career aspirations

        **Output Format:**
        UNDERSTANDING: [YES if interview complete, NO if continuing HR assessment]
        CONCEPT: behavioral_assessment
        QUESTION: [Your professional HR/behavioral question]
        """)
        
        self.evaluation_prompt = PromptTemplate.from_template("""
        You are evaluating a comprehensive professional interview across multiple rounds. Provide detailed, constructive assessment.

        **Technical Round Conversations:**
        {technical_conversations}

        **Communication Round Conversations:**
        {communication_conversations}

        **HR Round Conversations:**
        {hr_conversations}

        **Interview Analytics:**
        - Total Questions: {total_questions}
        - Technical Concepts Covered: {technical_concepts}
        - Follow-up Questions: {followup_questions}

        Generate a professional evaluation with these sections:
        1. **Technical Assessment** - Score: X/10 (Knowledge depth, problem-solving, technical communication)
        2. **Communication Skills** - Score: X/10 (Clarity, presentation, professional communication)
        3. **Cultural Fit & Soft Skills** - Score: X/10 (Teamwork, adaptability, behavioral competencies)
        4. **Overall Recommendation** - Score: X/10 (Overall suitability and growth potential)
        5. **Key Strengths** (2-3 specific points)
        6. **Areas for Development** (2-3 constructive suggestions)
        7. **Hiring Recommendation** (Strong Hire/Hire/No Hire with brief justification)

        Be thorough but concise (under 300 words). Use professional language suitable for hiring decisions.
        Include specific examples from their responses where relevant.
        """)
    
    def _parse_llm_response(self, response: str, keys: List[str]) -> Dict[str, str]:
        """Parse structured LLM responses"""
        result = {}
        lines = response.strip().split('\n')
        for line in lines:
            for key in keys:
                prefix = f"{key}:"
                if line.startswith(prefix):
                    result[key.lower()] = line[len(prefix):].strip()
                    break
        return result
    
    async def generate_greeting(self) -> Dict[str, str]:
        """Generate initial greeting"""
        chain = self.greeting_prompt | self.llm | self.parser
        response = await chain.ainvoke({})
        return self._parse_llm_response(response, ["CONCEPT", "QUESTION"])
    
    async def generate_technical_question(self, current_concept: str, concept_content: str, 
                                        history: str, previous_question: str, user_response: str,
                                        question_count: int, questions_for_concept: int) -> Dict[str, str]:
        """Generate technical interview question"""
        chain = self.technical_prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "current_concept": current_concept,
            "concept_content": concept_content,
            "history": history,
            "previous_question": previous_question,
            "user_response": user_response,
            "question_count": question_count,
            "questions_for_concept": questions_for_concept,
            "MAX_QUESTIONS_PER_ROUND": MAX_QUESTIONS_PER_ROUND
        })
        return self._parse_llm_response(response, ["UNDERSTANDING", "CONCEPT", "QUESTION"])
    
    async def generate_communication_question(self, history: str, previous_question: str, 
                                            user_response: str, question_count: int) -> Dict[str, str]:
        """Generate communication round question"""
        chain = self.communication_prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "history": history,
            "previous_question": previous_question,
            "user_response": user_response,
            "question_count": question_count,
            "MAX_QUESTIONS_PER_ROUND": MAX_QUESTIONS_PER_ROUND
        })
        return self._parse_llm_response(response, ["UNDERSTANDING", "CONCEPT", "QUESTION"])
    
    async def generate_hr_question(self, history: str, previous_question: str, 
                                 user_response: str, question_count: int) -> Dict[str, str]:
        """Generate HR round question"""
        chain = self.hr_prompt | self.llm | self.parser
        response = await chain.ainvoke({
            "history": history,
            "previous_question": previous_question,
            "user_response": user_response,
            "question_count": question_count,
            "MAX_QUESTIONS_PER_ROUND": MAX_QUESTIONS_PER_ROUND
        })
        return self._parse_llm_response(response, ["UNDERSTANDING", "CONCEPT", "QUESTION"])
    
    async def generate_evaluation(self, technical_convs: List[ConversationEntry], 
                                communication_convs: List[ConversationEntry],
                                hr_convs: List[ConversationEntry], analytics: Dict[str, Any]) -> str:
        """Generate comprehensive interview evaluation"""
        def format_conversations(convs: List[ConversationEntry]) -> str:
            if not convs:
                return "No conversations recorded for this round."
            
            formatted = []
            for entry in convs:
                formatted.append(f"Interviewer: {entry.question}")
                if entry.answer:
                    formatted.append(f"Candidate: {entry.answer}")
            return "\n".join(formatted)
        
        chain = self.evaluation_prompt | self.llm | self.parser
        return await chain.ainvoke({
            "technical_conversations": format_conversations(technical_convs),
            "communication_conversations": format_conversations(communication_convs),
            "hr_conversations": format_conversations(hr_convs),
            "total_questions": analytics.get("total_questions", 0),
            "technical_concepts": analytics.get("technical_concepts_covered", 0),
            "followup_questions": analytics.get("followup_questions", 0)
        })

# ========================
# Enhanced Audio Manager
# ========================

class AudioManager:
    """Enhanced audio management with cleanup and optimization"""
    def __init__(self):
        try:
            self.groq_client = Groq()
            logger.info("Audio manager initialized successfully")
        except Exception as e:
            logger.error(f"Audio manager initialization failed: {e}")
            self.groq_client = None
    
    @staticmethod
    def get_random_voice() -> str:
        """Get random professional voice"""
        voices = ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural"]
        return random.choice(voices)
    
    @staticmethod
    def clean_audio_folder():
        """Clean up old audio files"""
        try:
            current_time = time.time()
            cleanup_count = 0
            
            for filename in os.listdir(AUDIO_DIR):
                if filename.endswith(".mp3"):
                    file_path = os.path.join(AUDIO_DIR, filename)
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    # Remove files older than 1 hour
                    if file_age > 3600:
                        try:
                            os.remove(file_path)
                            cleanup_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete {filename}: {e}")
            
            # Clean temp directory
            for filename in os.listdir(TEMP_DIR):
                filepath = os.path.join(TEMP_DIR, filename)
                if os.path.getmtime(filepath) < current_time - 3600:
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
            
            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} old audio files")
                
        except Exception as e:
            logger.error(f"Error during audio cleanup: {e}")
    
    async def text_to_speech(self, text: str, voice: str, speed: float = TTS_SPEED) -> Optional[str]:
        """Convert text to speech with optimization"""
        timestamp = int(time.time() * 1000)
        raw_path = os.path.join(AUDIO_DIR, f"ai_raw_{timestamp}.mp3")
        final_path = os.path.join(AUDIO_DIR, f"ai_{timestamp}.mp3")
        
        try:
            # Clean old files before generating new ones
            self.clean_audio_folder()
            
            # Generate TTS audio
            await edge_tts.Communicate(text, voice).save(raw_path)

            # Apply speed adjustment with ffmpeg
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-filter:a", f"atempo={speed}", "-vn", final_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Clean up raw file
            if os.path.exists(raw_path):
                os.remove(raw_path)
            
            # Wait for final file and return relative path
            for _ in range(10):
                if os.path.exists(final_path):
                    return f"./audio/{os.path.basename(final_path)}"
                await asyncio.sleep(0.1)

            logger.error(f"TTS final audio file missing: {final_path}")
            return None

        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            # Clean up any partial files
            for path in [raw_path, final_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            return None
    
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
# Enhanced Test Manager
# ========================

class TestManager:
    """Enhanced test manager with fragment-based interviewing"""
    def __init__(self):
        self.tests: Dict[str, InterviewSession] = {}
        self.db_manager = DatabaseManager()
        self.llm_manager = LLMManager()
        self.audio_manager = AudioManager()
        logger.info("Test manager initialized")
    
    def create_test(self) -> str:
        """Create new interview test with technical content parsing"""
        test_id = str(uuid.uuid4())
        voice = AudioManager.get_random_voice()
        
        # Get technical content and parse fragments
        technical_content = self.db_manager.get_latest_technical_content()
        technical_fragments = parse_technical_fragments(technical_content)
        
        session = InterviewSession(
            test_id=test_id,
            current_round=RoundType.GREETING,
            state=InterviewState.NOT_STARTED,
            voice=voice,
            technical_fragments=technical_fragments,
            fragment_coverage={key: 0 for key in technical_fragments.keys()}
        )
        
        self.tests[test_id] = session
        logger.info(f"Created test {test_id} with {len(technical_fragments)} technical concepts")
        return test_id
    
    def get_test(self, test_id: str) -> Optional[InterviewSession]:
        """Get test by ID and update activity"""
        test = self.tests.get(test_id)
        if test:
            test.last_activity = time.time()
        return test
    
    def validate_test(self, test_id: str) -> InterviewSession:
        """Validate test ID and check timeout"""
        test = self.get_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Interview session not found")
        
        if time.time() > test.last_activity + INACTIVITY_TIMEOUT:
            raise HTTPException(status_code=408, detail="Interview session timed out")
        
        return test
    
    def get_active_technical_concept(self, test: InterviewSession) -> tuple[str, str]:
        """Get current active technical concept for questioning"""
        if not test.technical_fragments:
            return "General Programming", "Basic programming concepts and problem-solving"
        
        # Find concepts with minimal coverage
        min_coverage = min(test.fragment_coverage.values())
        underutilized_concepts = [
            concept for concept, count in test.fragment_coverage.items() 
            if count == min_coverage
        ]
        
        # Select next underutilized concept
        fragment_keys = list(test.technical_fragments.keys())
        for concept in fragment_keys:
            if concept in underutilized_concepts:
                return concept, test.technical_fragments[concept]
        
        # Fallback to cycling through concepts
        concept_index = test.questions_per_round[RoundType.TECHNICAL] % len(fragment_keys)
        selected_concept = fragment_keys[concept_index]
        return selected_concept, test.technical_fragments[selected_concept]
    
    def get_conversation_history(self, test: InterviewSession, round_type: RoundType, window_size: int = 4) -> str:
        """Get recent conversation history for context"""
        round_conversations = [
            entry for entry in test.conversation_log 
            if entry.round_type == round_type
        ]
        
        if not round_conversations:
            return "No previous conversation in this round."
        
        recent_conversations = round_conversations[-window_size:]
        formatted = []
        
        for entry in recent_conversations:
            formatted.append(f"Interviewer: {entry.question}")
            if entry.answer:
                formatted.append(f"Candidate: {entry.answer}")
        
        return "\n".join(formatted)
    
    def should_transition_round(self, test: InterviewSession) -> bool:
        """Determine if current round should transition to next"""
        current_round = test.current_round
        questions_asked = test.questions_per_round[current_round]
        
        # Minimum questions before allowing transition
        if questions_asked < MIN_QUESTIONS_PER_ROUND:
            return False
        
        # Force transition after maximum questions
        if questions_asked >= MAX_QUESTIONS_PER_ROUND:
            return True
        
        # Allow natural transition based on LLM assessment
        return False
    
    def get_next_round(self, current_round: RoundType) -> Optional[RoundType]:
        """Get next round in interview sequence"""
        sequence = [RoundType.GREETING, RoundType.TECHNICAL, RoundType.COMMUNICATION, RoundType.HR]
        try:
            current_idx = sequence.index(current_round)
            return sequence[current_idx + 1] if current_idx < len(sequence) - 1 else None
        except ValueError:
            return None
    
    def add_conversation_entry(self, test: InterviewSession, question: str, 
                             round_type: RoundType, concept: str = None, 
                             is_followup: bool = False):
        """Add conversation entry with enhanced tracking"""
        entry = ConversationEntry(
            question=question,
            round_type=round_type,
            concept=concept,
            is_followup=is_followup
        )
        
        test.conversation_log.append(entry)
        test.questions_per_round[round_type] += 1
        test.current_concept = concept
        
        if is_followup:
            test.followup_questions += 1
        
        # Track technical concept coverage
        if round_type == RoundType.TECHNICAL and concept and concept in test.fragment_coverage:
            test.fragment_coverage[concept] += 1
        
        logger.info(f"Added {round_type.value} question (followup: {is_followup}) "
                   f"for concept '{concept}' to test {test.test_id}")
    
    def add_answer(self, test: InterviewSession, answer: str):
        """Add answer to last conversation entry"""
        if test.conversation_log:
            test.conversation_log[-1].answer = answer
    
    async def start_interview(self, test_id: str) -> Dict[str, Any]:
        """Start interview with greeting"""
        test = self.validate_test(test_id)
        test.state = InterviewState.GREETING
        
        # Generate greeting
        greeting_data = await self.llm_manager.generate_greeting()
        question = greeting_data.get("question", "Hello! Welcome to today's interview.")
        concept = greeting_data.get("concept", "greeting")
        
        # Add to conversation log
        self.add_conversation_entry(test, question, RoundType.GREETING, concept)
        
        # Generate audio
        audio_path = await self.audio_manager.text_to_speech(question, test.voice)
        
        # Estimate duration
        total_rounds = 3  # Technical, Communication, HR
        estimated_duration = total_rounds * QUESTIONS_PER_ROUND * 3 * 60  # 3 minutes per question
        
        return {
            "test_id": test_id,
            "question": question,
            "audio_path": audio_path,
            "round": test.current_round.value,
            "duration_sec": estimated_duration
        }
    
    async def process_response(self, test_id: str, audio_file: UploadFile) -> Dict[str, Any]:
        """Process user audio response and generate next question"""
        test = self.validate_test(test_id)
        
        # Save and transcribe audio
        try:
            audio_filename = os.path.join(TEMP_DIR, f"user_response_{int(time.time())}_{test_id}.webm")
            content = await audio_file.read()
            
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty audio file received")
            
            with open(audio_filename, "wb") as f:
                f.write(content)
            
            user_response = self.audio_manager.transcribe_audio(audio_filename)
            
            # Cleanup
            try:
                os.remove(audio_filename)
            except Exception:
                pass
            
            if not user_response or not user_response.strip():
                raise HTTPException(status_code=400, detail="Could not understand audio response")
            
            logger.info(f"User response: {user_response[:100]}...")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            raise HTTPException(status_code=500, detail="Audio processing failed")
        
        # Add answer to conversation log
        self.add_answer(test, user_response)
        
        # Determine next action based on current round
        if test.current_round == RoundType.GREETING:
            return await self._handle_greeting_response(test, user_response)
        elif test.current_round == RoundType.TECHNICAL:
            return await self._handle_technical_response(test, user_response)
        elif test.current_round == RoundType.COMMUNICATION:
            return await self._handle_communication_response(test, user_response)
        elif test.current_round == RoundType.HR:
            return await self._handle_hr_response(test, user_response)
        else:
            raise HTTPException(status_code=400, detail="Invalid interview state")
    
    async def _handle_greeting_response(self, test: InterviewSession, user_response: str) -> Dict[str, Any]:
        """Handle response to greeting and transition to technical round"""
        test.current_round = RoundType.TECHNICAL
        test.state = InterviewState.IN_PROGRESS
        
        # Get first technical concept
        concept_title, concept_content = self.get_active_technical_concept(test)
        
        # Generate technical question
        history = self.get_conversation_history(test, RoundType.GREETING)
        last_question = test.conversation_log[-1].question if test.conversation_log else ""
        
        question_data = await self.llm_manager.generate_technical_question(
            current_concept=concept_title,
            concept_content=concept_content,
            history=history,
            previous_question=last_question,
            user_response=user_response,
            question_count=1,
            questions_for_concept=0
        )
        
        next_question = question_data.get("question", "Let's start with a technical question.")
        concept = question_data.get("concept", concept_title)
        
        # Add to conversation log
        self.add_conversation_entry(test, next_question, RoundType.TECHNICAL, concept)
        
        # Generate audio
        audio_path = await self.audio_manager.text_to_speech(next_question, test.voice)
        
        return {
            "ended": False,
            "response": next_question,
            "audio_path": audio_path or "",
            "round_complete": False,
            "current_round": test.current_round.value
        }
    
    async def _handle_technical_response(self, test: InterviewSession, user_response: str) -> Dict[str, Any]:
        """Handle technical round responses with adaptive questioning"""
        
        # Check if should force round transition
        if self.should_transition_round(test):
            return await self._transition_to_next_round(test, user_response)
        
        # Get current concept info
        concept_title, concept_content = self.get_active_technical_concept(test)
        
        # Generate follow-up question
        history = self.get_conversation_history(test, RoundType.TECHNICAL)
        last_question = test.conversation_log[-1].question if test.conversation_log else ""
        questions_for_concept = test.fragment_coverage.get(concept_title, 0)
        
        question_data = await self.llm_manager.generate_technical_question(
            current_concept=concept_title,
            concept_content=concept_content,
            history=history,
            previous_question=last_question,
            user_response=user_response,
            question_count=test.questions_per_round[RoundType.TECHNICAL] + 1,
            questions_for_concept=questions_for_concept
        )
        
        understanding = question_data.get("understanding", "NO").upper()
        next_question = question_data.get("question", "Can you elaborate on that?")
        suggested_concept = question_data.get("concept", concept_title)
        
        # Check for transition keywords
        transition_keywords = [
            "communication skills", "communication round", 
            "move to evaluate", "communication assessment"
        ]
        
        if understanding == "YES" or any(keyword in next_question.lower() for keyword in transition_keywords):
            return await self._transition_to_next_round(test, user_response, next_question)
        
        # Continue with current or new concept
        is_followup = (suggested_concept == concept_title)
        if not is_followup:
            concept_title, concept_content = self.get_active_technical_concept(test)
            suggested_concept = concept_title
        
        # Add to conversation log
        self.add_conversation_entry(test, next_question, RoundType.TECHNICAL, suggested_concept, is_followup)
        
        # Generate audio
        audio_path = await self.audio_manager.text_to_speech(next_question, test.voice)
        
        return {
            "ended": False,
            "response": next_question,
            "audio_path": audio_path or "",
            "round_complete": False,
            "current_round": test.current_round.value
        }
    
    async def _handle_communication_response(self, test: InterviewSession, user_response: str) -> Dict[str, Any]:
        """Handle communication round responses"""
        
        if self.should_transition_round(test):
            return await self._transition_to_next_round(test, user_response)
        
        history = self.get_conversation_history(test, RoundType.COMMUNICATION)
        last_question = test.conversation_log[-1].question if test.conversation_log else ""
        
        question_data = await self.llm_manager.generate_communication_question(
            history=history,
            previous_question=last_question,
            user_response=user_response,
            question_count=test.questions_per_round[RoundType.COMMUNICATION] + 1
        )
        
        understanding = question_data.get("understanding", "NO").upper()
        next_question = question_data.get("question", "Tell me about a time you had to explain something complex.")
        
        # Check for transition
        transition_keywords = ["hr round", "hr assessment", "final hr", "behavioral assessment"]
        
        if understanding == "YES" or any(keyword in next_question.lower() for keyword in transition_keywords):
            return await self._transition_to_next_round(test, user_response, next_question)
        
        # Continue communication round
        self.add_conversation_entry(test, next_question, RoundType.COMMUNICATION, "communication_skills")
        
        audio_path = await self.audio_manager.text_to_speech(next_question, test.voice)
        
        return {
            "ended": False,
            "response": next_question,
            "audio_path": audio_path or "",
            "round_complete": False,
            "current_round": test.current_round.value
        }
    
    async def _handle_hr_response(self, test: InterviewSession, user_response: str) -> Dict[str, Any]:
        """Handle HR round responses"""
        
        if self.should_transition_round(test):
            return await self._complete_interview(test, user_response)
        
        history = self.get_conversation_history(test, RoundType.HR)
        last_question = test.conversation_log[-1].question if test.conversation_log else ""
        
        question_data = await self.llm_manager.generate_hr_question(
            history=history,
            previous_question=last_question,
            user_response=user_response,
            question_count=test.questions_per_round[RoundType.HR] + 1
        )
        
        understanding = question_data.get("understanding", "NO").upper()
        next_question = question_data.get("question", "Tell me about a challenging situation you faced.")
        
        # Check for completion
        completion_keywords = ["completing the interview", "prepare your evaluation", "comprehensive evaluation"]
        
        if understanding == "YES" or any(keyword in next_question.lower() for keyword in completion_keywords):
            return await self._complete_interview(test, user_response, next_question)
        
        # Continue HR round
        self.add_conversation_entry(test, next_question, RoundType.HR, "behavioral_assessment")
        
        audio_path = await self.audio_manager.text_to_speech(next_question, test.voice)
        
        return {
            "ended": False,
            "response": next_question,
            "audio_path": audio_path or "",
            "round_complete": False,
            "current_round": test.current_round.value
        }
    
    async def _transition_to_next_round(self, test: InterviewSession, user_response: str, 
                                     transition_message: str = None) -> Dict[str, Any]:
        """Handle transition between interview rounds"""
        current_round = test.current_round
        next_round = self.get_next_round(current_round)
        
        if not next_round:
            return await self._complete_interview(test, user_response, transition_message)
        
        test.current_round = next_round
        test.state = InterviewState.ROUND_TRANSITION
        
        if not transition_message:
            transition_message = f"Thank you for the {current_round.value.lower()} assessment. Let's now move to the {next_round.value.lower()} round."
        
        audio_path = await self.audio_manager.text_to_speech(transition_message, test.voice)
        
        return {
            "ended": False,
            "response": transition_message,
            "audio_path": audio_path or "",
            "round_complete": True,
            "next_round": next_round.value,
            "current_round": current_round.value
        }
    
    async def _complete_interview(self, test: InterviewSession, user_response: str, 
                                completion_message: str = None) -> Dict[str, Any]:
        """Complete the interview"""
        test.state = InterviewState.INTERVIEW_COMPLETE
        
        if not completion_message:
            completion_message = "Thank you for completing the interview. We'll now prepare your comprehensive evaluation report."
        
        audio_path = await self.audio_manager.text_to_speech(completion_message, test.voice)
        
        return {
            "ended": False,
            "response": completion_message,
            "audio_path": audio_path or "",
            "interview_complete": True
        }
    
    async def start_next_round(self, test_id: str) -> Dict[str, Any]:
        """Start the next round after transition"""
        test = self.validate_test(test_id)
        test.state = InterviewState.IN_PROGRESS
        
        if test.current_round == RoundType.TECHNICAL:
            concept_title, concept_content = self.get_active_technical_concept(test)
            
            question_data = await self.llm_manager.generate_technical_question(
                current_concept=concept_title,
                concept_content=concept_content,
                history="",
                previous_question="",
                user_response="",
                question_count=1,
                questions_for_concept=0
            )
            
            question = question_data.get("question", "Let's start with a technical question.")
            concept = question_data.get("concept", concept_title)
            
        elif test.current_round == RoundType.COMMUNICATION:
            question_data = await self.llm_manager.generate_communication_question(
                history="",
                previous_question="",
                user_response="",
                question_count=1
            )
            
            question = question_data.get("question", "Let's assess your communication skills.")
            concept = "communication_skills"
            
        elif test.current_round == RoundType.HR:
            question_data = await self.llm_manager.generate_hr_question(
                history="",
                previous_question="",
                user_response="",
                question_count=1
            )
            
            question = question_data.get("question", "Let's discuss your experiences and cultural fit.")
            concept = "behavioral_assessment"
        
        else:
            raise HTTPException(status_code=400, detail="Invalid round for starting")
        
        # Add to conversation log
        self.add_conversation_entry(test, question, test.current_round, concept)
        
        # Generate audio
        audio_path = await self.audio_manager.text_to_speech(question, test.voice)
        
        return {
            "question": question,
            "audio_path": audio_path,
            "round": test.current_round.value
        }
    
    async def generate_evaluation(self, test_id: str) -> Dict[str, Any]:
        """Generate comprehensive interview evaluation"""
        test = self.validate_test(test_id)
        
        # Separate conversations by round
        technical_convs = [entry for entry in test.conversation_log if entry.round_type == RoundType.TECHNICAL]
        communication_convs = [entry for entry in test.conversation_log if entry.round_type == RoundType.COMMUNICATION]
        hr_convs = [entry for entry in test.conversation_log if entry.round_type == RoundType.HR]
        
        # Prepare analytics
        analytics = {
            "total_questions": len(test.conversation_log),
            "technical_concepts_covered": len([c for c, count in test.fragment_coverage.items() if count > 0]),
            "followup_questions": test.followup_questions,
            "questions_per_round": dict(test.questions_per_round),
            "fragment_coverage": dict(test.fragment_coverage)
        }
        
        # Generate evaluation
        evaluation = await self.llm_manager.generate_evaluation(
            technical_convs, communication_convs, hr_convs, analytics
        )
        
        # Extract scores
        scores = self._extract_scores_from_evaluation(evaluation)
        
        # Save to database
        save_success = self.db_manager.save_interview_data(test_id, test, evaluation, scores)
        if not save_success:
            logger.warning(f"Failed to save interview data for test {test_id}")
        
        return {
            "evaluation": evaluation,
            "scores": scores,
            "analytics": analytics,
            "pdf_url": f"./api/download_results/{test_id}"
        }
    
    def _extract_scores_from_evaluation(self, evaluation: str) -> Dict[str, Optional[float]]:
        """Extract scores from evaluation text"""
        scores = {
            "technical_score": None,
            "communication_score": None,
            "hr_score": None,
            "overall_score": None
        }
        
        patterns = {
            "technical_score": r"Technical Assessment.*?Score:\s*(\d+(?:\.\d+)?)/10",
            "communication_score": r"Communication Skills.*?Score:\s*(\d+(?:\.\d+)?)/10",
            "hr_score": r"Cultural Fit.*?Score:\s*(\d+(?:\.\d+)?)/10",
            "overall_score": r"Overall Recommendation.*?Score:\s*(\d+(?:\.\d+)?)/10"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, evaluation, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    scores[key] = float(match.group(1))
                except ValueError:
                    pass
        
        return scores
    
    def cleanup_expired_tests(self):
        """Remove expired test sessions"""
        current_time = time.time()
        expired_tests = [
            tid for tid, test in self.tests.items()
            if current_time > test.last_activity + INACTIVITY_TIMEOUT * 2
        ]
        
        for tid in expired_tests:
            self.tests.pop(tid, None)
        
        return len(expired_tests)

# ========================
# FastAPI Application Setup
# ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Enhanced Interview System starting up...")
    yield
    # Cleanup on shutdown
    test_manager.cleanup_expired_tests()
    AudioManager.clean_audio_folder()
    test_manager.db_manager.close()
    logger.info("Enhanced Interview System shut down")

app = FastAPI(
    title="Enhanced AI Interview System", 
    description="Professional AI-powered interview system with adaptive questioning",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
def setup_cors(app: FastAPI, frontend_origin: str):
    """Setup CORS middleware"""
    if frontend_origin == "*":
        logger.warning("CORS configured for ALL origins - NOT RECOMMENDED FOR PRODUCTION")
        allowed_origins = ["*"]
    else:
        allowed_origins = [origin.strip() for origin in frontend_origin.split(",")]
        logger.info(f"CORS configured for origins: {allowed_origins}")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

setup_cors(app, FRONTEND_ORIGIN)

# Static file serving
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Initialize managers
test_manager = TestManager()

# ========================
# API Endpoints
# ========================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the interview interface"""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Enhanced AI Interview System",
        "active_tests": len(test_manager.tests),
        "timestamp": time.time()
    }

@app.get("/start_interview", response_model=InterviewResponse)
async def start_interview():
    """Start a new interview session"""
    try:
        test_id = test_manager.create_test()
        result = await test_manager.start_interview(test_id)
        logger.info(f"Interview started: {test_id}")
        return InterviewResponse(**result)
    except Exception as e:
        logger.error(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}")

@app.post("/record_and_respond", response_model=ConversationResponse)
async def record_and_respond(
    audio: UploadFile = File(...),
    test_id: str = Form(...)
):
    """Process user audio response and provide next question"""
    try:
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        result = await test_manager.process_response(test_id, audio)
        logger.info(f"Response processed for test {test_id}")
        return ConversationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process response: {str(e)}")

@app.get("/start_next_round")
async def start_next_round(test_id: str):
    """Start the next interview round"""
    try:
        result = await test_manager.start_next_round(test_id)
        logger.info(f"Next round started for test {test_id}")
        return result
    except Exception as e:
        logger.error(f"Error starting next round: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate", response_model=EvaluationResponse)
async def get_evaluation(test_id: str):
    """Generate final interview evaluation"""
    try:
        result = await test_manager.generate_evaluation(test_id)
        logger.info(f"Evaluation generated for test {test_id}")
        return EvaluationResponse(**result)
    except Exception as e:
        logger.error(f"Error generating evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download_results/{test_id}")
async def download_interview_pdf(test_id: str):
    """Download interview results as PDF"""
    try:
        if not test_manager.db_manager.client:
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        
        doc = test_manager.db_manager.interviews.find_one({"test_id": test_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Interview results not found")

        # Create PDF in memory
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=LETTER)
        width, height = LETTER
        margin = 50
        y = height - margin

        def write_line(label: str, value: str, indent: int = 0):
            nonlocal y
            if y < margin + 50:
                p.showPage()
                p.setFont("Helvetica", 12)
                y = height - margin
            p.drawString(margin + indent, y, f"{label}: {value}")
            y -= 20

        # Header
        p.setFont("Helvetica-Bold", 16)
        p.drawString(margin, y, f"Interview Evaluation Report")
        y -= 25
        p.setFont("Helvetica-Bold", 12)
        p.drawString(margin, y, f"Test ID: {test_id}")
        y -= 30

        # Basic information
        p.setFont("Helvetica", 12)
        write_line("Candidate Name", str(doc.get("name", "N/A")))
        write_line("Student ID", str(doc.get("Student_ID", "N/A")))
        write_line("Session ID", str(doc.get("session_id", "N/A")))
        
        # Timestamp
        try:
            ts = float(doc.get("timestamp", time.time()))
            timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        except:
            timestr = "N/A"
        write_line("Interview Date", timestr)

        # Scores
        y -= 10
        p.setFont("Helvetica-Bold", 12)
        if y < margin + 50:
            p.showPage()
            p.setFont("Helvetica-Bold", 12)
            y = height - margin
        p.drawString(margin, y, "Performance Scores:")
        y -= 20

        p.setFont("Helvetica", 12)
        scores = doc.get("scores", {})
        score_labels = {
            "technical_score": "Technical Assessment",
            "communication_score": "Communication Skills", 
            "hr_score": "Cultural Fit & Soft Skills",
            "overall_score": "Overall Rating"
        }
        
        for key, label in score_labels.items():
            score_val = scores.get(key, "N/A")
            if score_val != "N/A":
                score_val = f"{score_val}/10"
            write_line(label, str(score_val))

        # Analytics
        analytics = doc.get("interview_analytics", {})
        if analytics:
            y -= 10
            p.setFont("Helvetica-Bold", 12)
            if y < margin + 50:
                p.showPage()
                p.setFont("Helvetica-Bold", 12)
                y = height - margin
            p.drawString(margin, y, "Interview Analytics:")
            y -= 20
            
            p.setFont("Helvetica", 11)
            write_line("Total Questions", str(analytics.get("total_questions", "N/A")))
            write_line("Main Questions", str(analytics.get("main_questions", "N/A")))
            write_line("Follow-up Questions", str(analytics.get("followup_questions", "N/A")))
            write_line("Technical Concepts Covered", str(analytics.get("technical_fragments_covered", "N/A")))

        # Evaluation text
        y -= 10
        p.setFont("Helvetica-Bold", 12)
        if y < margin + 50:
            p.showPage()
            p.setFont("Helvetica-Bold", 12)
            y = height - margin
        p.drawString(margin, y, "Detailed Evaluation:")
        y -= 20

        # Wrap evaluation text
        p.setFont("Helvetica", 12)
        evaluation = str(doc.get("evaluation", "No evaluation available"))
        wrapped_lines = textwrap.wrap(evaluation, 80)
        
        for line in wrapped_lines:
            if y < margin + 20:
                p.showPage()
                p.setFont("Helvetica", 12)
                y = height - margin
            p.drawString(margin, y, line)
            y -= 15

        # Conversation summary by rounds
        y -= 15
        p.setFont("Helvetica-Bold", 12)
        if y < margin + 50:
            p.showPage()
            p.setFont("Helvetica-Bold", 12)
            y = height - margin
        p.drawString(margin, y, "Interview Conversation Summary:")
        y -= 20

        p.setFont("Helvetica", 11)
        conversation_log = doc.get("conversation_log", [])
        
        # Group by rounds
        rounds = {"Technical": [], "Communication": [], "HR": []}
        for entry in conversation_log:
            round_type = entry.get("round_type", "Unknown")
            if round_type in rounds:
                rounds[round_type].append(entry)

        for round_name, entries in rounds.items():
            if not entries:
                continue
                
            if y < margin + 80:
                p.showPage()
                p.setFont("Helvetica", 11)
                y = height - margin
            
            p.setFont("Helvetica-Bold", 11)
            p.drawString(margin, y, f"{round_name} Round ({len(entries)} questions):")
            y -= 20
            p.setFont("Helvetica", 10)
            
            for idx, entry in enumerate(entries[:3], 1):  # Show first 3 questions per round
                if y < margin + 60:
                    p.showPage()
                    p.setFont("Helvetica", 10)
                    y = height - margin
                
                question = entry.get("question", "N/A")
                answer = entry.get("answer", "N/A")
                
                # Truncate long texts
                if len(question) > 100:
                    question = question[:97] + "..."
                if len(answer) > 100:
                    answer = answer[:97] + "..."
                
                p.drawString(margin + 10, y, f"Q{idx}: {question}")
                y -= 15
                p.drawString(margin + 10, y, f"A{idx}: {answer}")
                y -= 20

        p.showPage()
        p.save()
        buffer.seek(0)

        filename = f"interview_evaluation_{test_id}.pdf"
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating PDF for {test_id}: {e}")
        raise HTTPException(status_code=500, detail="PDF generation failed")

@app.get("/cleanup")
async def cleanup_resources():
    """Clean up audio files and expired tests"""
    try:
        AudioManager.clean_audio_folder()
        expired_count = test_manager.cleanup_expired_tests()
        
        return {
            "message": f"Cleaned up {expired_count} expired tests and old audio files",
            "expired_tests": expired_count,
            "active_tests": len(test_manager.tests),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail="Cleanup failed")

# ========================
# Enhanced API Endpoints
# ========================

@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": "Enhanced AI Interview System",
        "version": "2.0.0",
        "description": "Professional AI-powered interview system with adaptive questioning and comprehensive evaluation",
        "features": {
            "adaptive_questioning": "Questions adapt based on candidate responses and understanding",
            "fragment_based_technical": "Technical questions based on parsed content concepts",
            "multi_round_assessment": "Technical, Communication, and HR rounds",
            "professional_evaluation": "Comprehensive scoring and detailed feedback",
            "audio_processing": "Real-time speech-to-text and text-to-speech",
            "silence_detection": "Automatic recording termination on silence",
            "preparation_countdown": "5-second preparation before recording",
            "enhanced_analytics": "Detailed performance tracking and concept coverage"
        },
        "rounds": {
            "greeting": "Professional welcome and process explanation",
            "technical": "Technical knowledge and problem-solving assessment",
            "communication": "Verbal communication and presentation skills",
            "hr": "Behavioral assessment and cultural fit"
        },
        "endpoints": {
            "start_interview": "POST /start_interview - Begin new interview session",
            "record_and_respond": "POST /record_and_respond - Process audio responses",
            "start_next_round": "GET /start_next_round - Transition to next round",
            "evaluate": "GET /evaluate - Generate final evaluation",
            "download_pdf": "GET /api/download_results/{test_id} - Download PDF report"
        }
    }

@app.get("/api/interviews", response_class=JSONResponse)
async def get_all_interviews():
    """Get all interview results"""
    try:
        results = list(test_manager.db_manager.interviews.find(
            {}, 
            {"_id": 0, "conversation_log": 0, "evaluation": 0}
        ).sort("timestamp", -1))
        
        return {
            "interviews": results,
            "count": len(results),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error fetching interviews: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interviews")

@app.get("/api/interviews/{test_id}", response_class=JSONResponse)
async def get_interview_by_id(test_id: str):
    """Get specific interview by test_id"""
    try:
        result = test_manager.db_manager.interviews.find_one(
            {"test_id": test_id}, 
            {"_id": 0, "conversation_log": 0, "evaluation": 0}
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching interview {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interview")

@app.get("/api/interview-students", response_class=JSONResponse)
async def get_unique_interview_students():
    """Get distinct students from interview records"""
    try:
        pipeline = [
            {"$group": {"_id": "$Student_ID", "name": {"$first": "$name"}}},
            {"$project": {"_id": 0, "Student_ID": "$_id", "name": 1}},
            {"$sort": {"Student_ID": 1}}
        ]
        
        students = list(test_manager.db_manager.interviews.aggregate(pipeline))
        
        return {
            "count": len(students), 
            "students": students,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error fetching interview students: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve students")

@app.get("/api/interview-students/{student_id}/interviews", response_class=JSONResponse)
async def get_interviews_for_student(student_id: str):
    """Get all interviews for a specific student"""
    try:
        student_id_int = int(student_id)
        results = list(test_manager.db_manager.interviews.find(
            {"Student_ID": student_id_int},
            {"_id": 0, "conversation_log": 0, "evaluation": 0}
        ).sort("timestamp", -1))
        
        if not results:
            raise HTTPException(status_code=404, detail="No interviews found for this student")
        
        return {
            "count": len(results), 
            "interviews": results,
            "student_id": student_id_int,
            "timestamp": time.time()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid student ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching interviews for student {student_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interviews")

@app.get("/api/stats", response_class=JSONResponse)
async def get_system_stats():
    """Get enhanced system statistics"""
    try:
        active_tests = len(test_manager.tests)
        audio_files = len([f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')])
        total_interviews = test_manager.db_manager.interviews.count_documents({})
        
        # Get recent analytics
        recent_interviews = list(test_manager.db_manager.interviews.find(
            {"interview_analytics": {"$exists": True}},
            {"interview_analytics": 1, "scores": 1, "_id": 0}
        ).limit(10))
        
        avg_score = 0
        avg_questions = 0
        avg_concepts = 0
        
        if recent_interviews:
            scores = [interview.get("scores", {}).get("overall_score", 0) for interview in recent_interviews]
            scores = [s for s in scores if s is not None]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            analytics_list = [interview.get("interview_analytics", {}) for interview in recent_interviews]
            total_questions = [a.get("total_questions", 0) for a in analytics_list]
            avg_questions = sum(total_questions) / len(total_questions) if total_questions else 0
            
            concepts_covered = [a.get("technical_fragments_covered", 0) for a in analytics_list]
            avg_concepts = sum(concepts_covered) / len(concepts_covered) if concepts_covered else 0

        return {
            "active_interview_sessions": active_tests,
            "audio_files_on_disk": audio_files,
            "total_completed_interviews": total_interviews,
            "recent_performance": {
                "avg_overall_score": round(avg_score, 1),
                "avg_questions_per_interview": round(avg_questions, 1),
                "avg_technical_concepts_covered": round(avg_concepts, 1)
            },
            "configuration": {
                "questions_per_round_target": QUESTIONS_PER_ROUND,
                "min_questions_per_round": MIN_QUESTIONS_PER_ROUND,
                "max_questions_per_round": MAX_QUESTIONS_PER_ROUND,
                "inactivity_timeout_seconds": INACTIVITY_TIMEOUT,
                "preparation_time_seconds": PREPARATION_TIME / 1000
            },
            "directories": {
                "audio_directory": AUDIO_DIR,
                "temp_directory": TEMP_DIR,
                "frontend_directory": FRONTEND_DIR
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error fetching system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@app.delete("/api/interviews/{test_id}")
async def delete_interview_result(test_id: str):
    """Delete specific interview result"""
    try:
        result = test_manager.db_manager.interviews.delete_one({"test_id": test_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        logger.info(f"Deleted interview result: {test_id}")
        return {
            "message": f"Interview {test_id} deleted successfully",
            "deleted_count": result.deleted_count,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting interview {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete interview")

# ========================
# Debug and Monitoring Endpoints
# ========================

@app.get("/debug/tests")
async def debug_active_tests():
    """Debug endpoint for active test sessions"""
    try:
        tests_info = {}
        for test_id, session in test_manager.tests.items():
            tests_info[test_id] = {
                "current_round": session.current_round.value,
                "state": session.state.value,
                "questions_per_round": dict(session.questions_per_round),
                "followup_questions": session.followup_questions,
                "technical_concepts": len(session.technical_fragments),
                "concept_coverage": dict(session.fragment_coverage),
                "last_activity": time.ctime(session.last_activity)
            }
        
        return {
            "total_active_tests": len(tests_info),
            "tests": tests_info,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return {"error": str(e)}

@app.get("/debug/test/{test_id}")
async def debug_test_details(test_id: str):
    """Debug endpoint for specific test details"""
    try:
        test = test_manager.get_test(test_id)
        if not test:
            return {"error": "Test not found", "test_id": test_id}
        
        return {
            "test_id": test_id,
            "current_round": test.current_round.value,
            "state": test.state.value,
            "voice": test.voice,
            "questions_per_round": dict(test.questions_per_round),
            "followup_questions": test.followup_questions,
            "total_conversations": len(test.conversation_log),
            "technical_fragments": list(test.technical_fragments.keys()),
            "fragment_coverage": dict(test.fragment_coverage),
            "current_concept": test.current_concept,
            "last_activity": time.ctime(test.last_activity),
            "round_start_time": time.ctime(test.round_start_time)
        }
    except Exception as e:
        logger.error(f"Error in test debug endpoint: {e}")
        return {"error": str(e)}

@app.get("/debug/database")
async def debug_database_connection():
    """Debug database connections and data"""
    try:
        # Test MongoDB
        mongo_status = "connected" if test_manager.db_manager.client else "disconnected"
        
        # Test latest technical content
        try:
            technical_content = test_manager.db_manager.get_latest_technical_content()
            technical_fragments = parse_technical_fragments(technical_content)
            content_info = {
                "available": True,
                "content_length": len(technical_content),
                "fragments_count": len(technical_fragments),
                "fragment_titles": list(technical_fragments.keys())[:5]
            }
        except Exception as e:
            content_info = {"available": False, "error": str(e)}
        
        # Test SQL Server
        try:
            student_info = fetch_random_student_info()
            sql_status = "connected" if student_info else "no_data"
            sql_info = {
                "status": sql_status,
                "sample_student": student_info if student_info else None
            }
        except Exception as e:
            sql_info = {"status": "error", "error": str(e)}
        
        return {
            "mongodb": {
                "status": mongo_status,
                "database": MONGO_DB_NAME,
                "host": MONGO_HOST
            },
            "sql_server": sql_info,
            "technical_content": content_info,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Database debug error: {e}")
        return {"error": str(e)}

# ========================
# Error Handlers
# ========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred during interview processing",
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

# ========================
# Root Information Endpoint
# ========================

@app.get("/api/root")
async def interview_root():
    """Root endpoint with comprehensive system information"""
    return {
        "service": "Enhanced AI Interview System",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "adaptive_questioning": "AI adapts questions based on candidate responses",
            "multi_round_assessment": "Technical, Communication, and HR evaluation",
            "fragment_based_technical": "Technical questions from parsed content concepts",
            "professional_flow": "Greeting → Technical → Communication → HR → Evaluation",
            "enhanced_audio": "Silence detection, preparation countdown, auto-cleanup",
            "comprehensive_evaluation": "Detailed scoring with PDF reports",
            "real_time_processing": "Live speech-to-text and text-to-speech",
            "robust_analytics": "Performance tracking and concept coverage analysis"
        },
        "configuration": {
            "questions_per_round": QUESTIONS_PER_ROUND,
            "preparation_time": f"{PREPARATION_TIME/1000}s",
            "max_recording": f"{MAX_RECORDING_DURATION/1000}s",
            "silence_threshold": SILENCE_THRESHOLD,
            "inactivity_timeout": f"{INACTIVITY_TIMEOUT/60}min"
        },
        "endpoints": {
            "interview_interface": "/",
            "health_check": "/health",
            "api_info": "/api/info",
            "start_interview": "/start_interview",
            "process_response": "/record_and_respond",
            "next_round": "/start_next_round",
            "evaluation": "/evaluate", 
            "download_pdf": "/api/download_results/{test_id}",
            "all_interviews": "/api/interviews",
            "students": "/api/interview-students",
            "system_stats": "/api/stats",
            "cleanup": "/cleanup",
            "debug_tests": "/debug/tests",
            "debug_database": "/debug/database"
        },
        "usage": {
            "start_interview": "Visit / to begin a new interview session",
            "api_documentation": "Visit /docs for interactive API documentation",
            "system_monitoring": "Visit /api/stats for system performance metrics",
            "debug_information": "Visit /debug/tests for active session debugging"
        },
        "timestamp": time.time()
    }

# ========================
# Development Server
# ========================

if __name__ == "__main__":
    import uvicorn
    import socket
    
    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    local_ip = get_local_ip()
    port = 8062  # Different port for standalone operation
    
    print(f"🚀 Starting Enhanced AI Interview System")
    print(f"📡 Server: https://{local_ip}:{port}")
    print(f"📋 API Docs: https://{local_ip}:{port}/docs")
    print(f"🎙️ Interview Interface: https://{local_ip}:{port}/")
    print(f"🔊 Audio Files: https://{local_ip}:{port}/audio/")
    print(f"🎯 Multi-Round Assessment: Technical → Communication → HR")
    print(f"🧠 Adaptive Questioning: Enabled")
    print(f"📊 Enhanced Analytics: Enabled")
    print(f"🌐 CORS Origins: {FRONTEND_ORIGIN}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
        ssl_certfile="certs/cert.pem",
        ssl_keyfile="certs/key.pem",
    )