import os
import time
import uuid
import random
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import pyodbc

import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from fastapi.responses import StreamingResponse
import textwrap

import subprocess
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
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
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# REMOVED: Audio recording parameters - now handled by frontend
TTS_SPEED = 1.3

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

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
class InterviewTest:
    test_id: str
    current_round: RoundType
    state: InterviewState
    voice: str
    conversations: Dict[RoundType, List[ConversationEntry]] = field(default_factory=dict)
    round_start_time: float = field(default_factory=time.time)
    questions_asked: Dict[RoundType, int] = field(default_factory=lambda: {r: 0 for r in RoundType})
    last_activity: float = field(default_factory=time.time)

class RecordRequest(BaseModel):
    test_id: str

# ========================
# Database Manager
# ========================

# SQL Server connection parameters
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

def get_db_connection():
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        return conn
    except pyodbc.Error as e:
        logger.error(f"Database connection error: {e}")
        return None

# Fetch a random Student_ID and First_Name, Last_Name from tbl_Student SQL Server 
def fetch_random_student_info():
    """Fetch a random ID, name from tbl_Student and session_id from session table from SQL Server"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        # Fetch all student records (ID, First_Name, Last_Name)
        cursor.execute("SELECT ID, First_Name, Last_Name FROM tbl_Student WHERE ID IS NOT NULL AND First_Name IS NOT NULL AND Last_Name IS NOT NULL")
        
        student_records = cursor.fetchall()
        
        if not student_records:
            logger.warning("No valid student data found in the database")
            return None

        # Fetch distinct Session_ID
        cursor.execute("SELECT DISTINCT Session_ID FROM tbl_Session WHERE Session_ID IS NOT NULL")
        session_rows = cursor.fetchall()
        session_ids = [row[0] for row in session_rows]

        cursor.close()
        conn.close()

        # Randomly select one student record
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
        logger.error(f"Error fetching student info: {e}") # Updated error message for clarity
        return None

    
class DatabaseManager:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(
                "mongodb://sa:L%40nc%5Eere%400012@192.168.48.200:27017/?authSource=admin"
            )
            self.db = self.client["test"]
            self.transcripts = self.db["drive"]
            self.interviews = self.db["mock_interview_results"]
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.client = None
    
    def save_interview_data(self, test_id: str, conversations: Dict[RoundType, List[ConversationEntry]], evaluation: str, scores: Dict[str, Optional[float]]) -> bool:
        try:
            if not self.client:
                logger.warning("No database connection available")
                return False
            # Fetch student ID from SQL Server
            student_id, first_name, last_name, session_id = fetch_random_student_info()
            if not student_id:  # If no student ID is fetched, log and return
                logger.warning("No valid ID fetched from SQL Server")
            
            name = first_name + " " + last_name if first_name and last_name else "Unknown Student"
            # Flatten conversations
            conv_data = {}
            for round_type, conv_list in conversations.items():
                conv_data[round_type.value] = [{
                    "user_input": entry.user_input,
                    "ai_response": entry.ai_response,
                    "timestamp": entry.timestamp
                } for entry in conv_list]
            
            doc = {
                "test_id": test_id,
                "timestamp": time.time(),
                "Student_ID": student_id,
                "name": name,
                "session_id": session_id,
                "conversations": conv_data,
                "evaluation": evaluation,
                "scores": scores
            }
            result = self.interviews.insert_one(doc)
            logger.info(f"Interview saved for test {test_id}, id {result.inserted_id}")
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
            return "\n\n".join(summaries[:5])  # Limit to 5 summaries for efficiency
        except Exception as e:
            logger.error(f"Database error: {e}")
            return "Technical content unavailable."

# ========================
# LLM Manager
# ========================

def split_into_three_parts(x):
    base = x // 3
    rem = x % 3

    parts = []
    start = 1
    for i in range(3):
        count = base + (1 if i < rem else 0)
        end = start + count - 1
        parts.append((start, end))
        start = end + 1
    return parts

total_questions = 6 # total questions for each interview round
technical_ranges = split_into_three_parts(total_questions)
communication_ranges = split_into_three_parts(total_questions)
hr_ranges = split_into_three_parts(total_questions)
        
class LLMManager:
    def __init__(self):
        # Fixed model name and increased max_tokens for longer responses
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.8)
        self.parser = StrOutputParser()
        
        # prompts for each round in shorter interviews
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
            - Questions {technical_ranges[0][0]}-{technical_ranges[0][1]}: Ask fundamental technical concepts and basic programming knowledge
            - Questions {technical_ranges[1][0]}-{technical_ranges[1][1]}: Ask application-based questions and problem-solving scenarios
            - Questions {technical_ranges[2][0]}-{technical_ranges[2][1]}: Ask advanced problem-solving and system design questions
            - If question count >= 6: Say "Thank you for the technical round. Let's move to the communication round."
            
            Ask specific, practical technical question. Keep it focused and clear.
            
            """),
            
            RoundType.COMMUNICATION: PromptTemplate.from_template("""
            You are testing communication skills in this round (20 minutes duration). Focus on verbal communication, clarity, and presentation abilities.
            If the candidate goes off-topic, tell them explicitly that they are going off-topic and ask a redirected next communication question.
            Previous conversation:
            {history}
            
            Current question number: {question_count}
            
            Instructions:
            - Questions {communication_ranges[0][0]}-{communication_ranges[0][1]}: Test clarity of explanation and articulation skills
            - Questions {communication_ranges[1][0]}-{communication_ranges[1][1]}: Test persuasion, storytelling, and explanation skills
            - Questions {communication_ranges[2][0]}-{communication_ranges[2][1]}: Test confidence, presentation skills, and leadership communication
            - If question count >= 6: Say "Thank you for the communication round. Let's proceed to the HR round."

            Ask question that evaluates verbal communication, confidence, or presentation skills.
            """),
            
            RoundType.HR: PromptTemplate.from_template("""
            You are an HR interviewer conducting behavioral assessment (20 minutes duration). Focus on cultural fit, teamwork, and soft skills.
            If the candidate goes off-topic, tell them explicitly that they are going off-topic and ask a redirected next HR question.
            Previous conversation:
            {history}
            
            Current question number: {question_count}
            
            Instructions:
            - Questions {hr_ranges[0][0]}-{hr_ranges[0][1]}: Ask about past experiences, teamwork, and collaboration
            - Questions {hr_ranges[1][0]}-{hr_ranges[1][1]}: Ask situational and behavioral questions (STAR method encouraged)
            - Questions {hr_ranges[2][0]}-{hr_ranges[2][1]}: Ask about conflict resolution, adaptability, and growth mindset
            - If question count >= 6: Say "Thank you for completing the interview. We'll now prepare your evaluation."
            Ask behavioral or situational question that reveals personality and cultural fit.
            """)
        } 

        self.evaluation_prompt = PromptTemplate.from_template("""
        Evaluate this interview performance strictly across all rounds. Provide specific scores and constructive feedback.
        
        Technical Round Conversations:
        {technical_conversation}
        
        Communication Round Conversations:
        {communication_conversation}
        
        HR Round Conversations:
        {hr_conversation}
        
        Provide a structured evaluation with:
        1. Technical round - Score: X/10
        2. Communication round - Score: X/10  
        3. HR round - Score: X/10
        4. Overall recommendation and areas for improvement (2-3 sentences) - Overall Score: X/10
        
        Keep the evaluation professional, constructive, strict and under 250 words.
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
            ],
            RoundType.COMMUNICATION: [
                "this is a fallback question for communication round.",
                "Check for errors in your code and try again."
            ],
            RoundType.HR: [
                "this is a fallback question for HR round.",
                "Check for errors in your code and try again."
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
    scores["technical_score"] = _extract_score(text, r"Technical round.*?Score:\s*(\d+(?:\.\d+)?)/10")
    scores["communication_score"] = _extract_score(text, r"Communication round.*?Score:\s*(\d+(?:\.\d+)?)/10")
    scores["hr_score"] = _extract_score(text, r"HR round.*?Score:\s*(\d+(?:\.\d+)?)/10")
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
# Audio Manager - REFACTORED
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
                if filename.startswith(("ai_", "temp_")) and filename.endswith((".mp3", ".wav", ".webm")):
                    filepath = os.path.join(AUDIO_DIR, filename)
                    if os.path.getmtime(filepath) < current_time - 3600:  # 1 hour
                        os.remove(filepath)
            # Clean temp directory
            for filename in os.listdir(TEMP_DIR):
                filepath = os.path.join(TEMP_DIR, filename)
                if os.path.getmtime(filepath) < current_time - 3600:
                    os.remove(filepath)
        except Exception as e:
            logger.warning(f"Audio cleanup error: {e}")
    
    async def text_to_speech(self, text: str, voice: str, speed: float = TTS_SPEED) -> Optional[str]:
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
    
    # REMOVED: record_audio method - now handled by frontend
    
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
# Test Manager - UPDATED FOR UPLOAD
# ========================

class TestManager:
    def __init__(self):
        self.tests: Dict[str, InterviewTest] = {}
        self.db_manager = DatabaseManager()
        self.llm_manager = LLMManager()
        self.audio_manager = AudioManager()
        logger.info("TestManager initialized")
    
    def create_test(self) -> str:
        """Create a new interview test"""
        test_id = str(uuid.uuid4())
        voice = AudioManager.get_random_voice()
        
        test = InterviewTest(
            test_id=test_id,
            current_round=RoundType.TECHNICAL,
            state=InterviewState.NOT_STARTED,
            voice=voice
        )
        
        # Initialize conversation logs for all rounds
        for round_type in RoundType:
            test.conversations[round_type] = []
        
        self.tests[test_id] = test
        logger.info(f"Created test {test_id}. Total tests: {len(self.tests)}")
        logger.info(f"Test details: Round={test.current_round.value}, State={test.state.value}")
        return test_id
    
    def get_test(self, test_id: str) -> Optional[InterviewTest]:
        """Get test by ID with enhanced debugging"""
        logger.info(f"Looking for test: {test_id}")
        logger.info(f"Available tests: {list(self.tests.keys())}")
        
        test = self.tests.get(test_id)
        if test:
            test.last_activity = time.time()
            logger.info(f"Found test {test_id}. Round: {test.current_round.value}, State: {test.state.value}")
        else:
            logger.error(f"Test {test_id} not found! Available tests: {list(self.tests.keys())}")
        
        return test
    
    def list_tests(self) -> Dict[str, Dict[str, str]]:
        """Debug method to list all active tests"""
        test_info = {}
        for tid, test in self.tests.items():
            test_info[tid] = {
                "round": test.current_round.value,
                "state": test.state.value,
                "last_activity": time.ctime(test.last_activity),
                "questions_asked": str(test.questions_asked)
            }
        return test_info
    
    def should_end_round(self, test: InterviewTest) -> bool:
        """Determine if current round should end based on questions asked"""
        current_round = test.current_round
        questions_asked = test.questions_asked[current_round]
        
        # Increased limits for longer tests
        # limits = {
        #     RoundType.TECHNICAL: 12,
        #     RoundType.COMMUNICATION: 12,
        #     RoundType.HR: 12
        # }
        
        # decreased limits for shorter tests
        limits = {
            RoundType.TECHNICAL: 3,
            RoundType.COMMUNICATION: 3,
            RoundType.HR: 3
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
    
    def format_conversation_history(self, test: InterviewTest, round_type: RoundType) -> str:
        """Format conversation history for LLM context"""
        conversations = test.conversations.get(round_type, [])
        if not conversations:
            return "No previous conversation in this round."
        
        # Include last 4 exchanges for better context
        recent_conversations = conversations[-4:]
        
        formatted = []
        for entry in recent_conversations:
            formatted.append(f"Interviewer: {entry.ai_response}")
            formatted.append(f"Candidate: {entry.user_input}")
        
        return "\n".join(formatted)
    
    async def generate_first_question(self, test_id: str) -> Dict[str, Any]:
        """Generate the first question for a round"""
        test = self.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found in generate_first_question")
            raise HTTPException(status_code=404, detail="Test not found")
        
        test.state = InterviewState.IN_PROGRESS
        test.round_start_time = time.time()
        
        # Get technical content for technical round
        technical_content = ""
        if test.current_round == RoundType.TECHNICAL:
            technical_content = self.db_manager.get_technical_content()
        
        # Generate question
        history = self.format_conversation_history(test, test.current_round)
        question_count = test.questions_asked[test.current_round] + 1
        
        question = await self.llm_manager.generate_question(
            test.current_round,
            history,
            question_count,
            technical_content
        )
        
        test.questions_asked[test.current_round] += 1
        logger.info(f"Generated question {question_count} for round {test.current_round.value}")
        
        # Generate audio
        audio_path = await self.audio_manager.text_to_speech(question, test.voice)
        
        return {
            "question": question,
            "audio_path": audio_path,
            "round": test.current_round.value
        }
    
    # NEW METHOD: Process uploaded audio instead of recording
    async def process_user_response_upload(self, test_id: str, audio_file: UploadFile) -> Dict[str, Any]:
        """Process user's uploaded audio response and generate AI reply"""
        test = self.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found in process_user_response_upload")
            raise HTTPException(status_code=404, detail="Test not found")
        
        logger.info(f"Processing uploaded audio for test {test_id}, round {test.current_round.value}")
        
        # Save uploaded audio to temp directory
        try:
            audio_filename = os.path.join(TEMP_DIR, f"user_input_{int(time.time())}_{audio_file.filename}")
            with open(audio_filename, "wb") as f:
                content = await audio_file.read()
                f.write(content)
            
            logger.info(f"Audio file saved: {audio_filename}")
            
            # Transcribe audio
            user_text = self.audio_manager.transcribe_audio(audio_filename)
            
            # Cleanup audio file
            try:
                os.remove(audio_filename)
                logger.info(f"Cleaned up audio file: {audio_filename}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup audio file {audio_filename}: {cleanup_error}")
            
            if not user_text:
                logger.warning("Could not transcribe audio")
                return {"error": "Could not understand audio", "retry": True}
            
            logger.info(f"User response transcribed: {user_text[:100]}...")
            
        except Exception as audio_error:
            logger.error(f"Audio processing error: {audio_error}")
            return {"error": "Audio processing failed", "retry": True}
        
        # Update test activity
        test.last_activity = time.time()
        
        # Check if we should end the round or continue
        if self.should_end_round(test):
            return await self._handle_round_transition(test, user_text)
        
        # Generate next question
        technical_content = ""
        if test.current_round == RoundType.TECHNICAL:
            technical_content = self.db_manager.get_technical_content()
        
        history = self.format_conversation_history(test, test.current_round)
        question_count = test.questions_asked[test.current_round] + 1
        
        ai_response = await self.llm_manager.generate_question(
            test.current_round,
            history,
            question_count,
            technical_content
        )
        
        # Save conversation entry
        entry = ConversationEntry(user_input=user_text, ai_response=ai_response)
        test.conversations[test.current_round].append(entry)
        test.questions_asked[test.current_round] += 1
        
        # Check for round transition keywords in AI response
        transition_keywords = [
            "move to the communication round",
            "proceed to the HR round", 
            "completing the interview",
            "prepare your evaluation"
        ]
        
        is_transition = any(keyword in ai_response.lower() for keyword in transition_keywords)
        
        if is_transition:
            return await self._handle_round_transition(test, user_text, ai_response)
        
        # Generate audio for next question
        audio_path = await self.audio_manager.text_to_speech(ai_response, test.voice)
        
        return {
            "response": ai_response,
            "audio_path": audio_path,
            "continue": True,
            "round": test.current_round.value
        }
    
    async def _handle_round_transition(self, test: InterviewTest, 
                                     user_text: str, ai_response: str = None) -> Dict[str, Any]:
        """Handle transition between rounds or end of interview"""
        current_round = test.current_round
        
        # Save the user's response if we have an AI response
        if ai_response:
            entry = ConversationEntry(user_input=user_text, ai_response=ai_response)
            test.conversations[current_round].append(entry)
        
        # Check if there's a next round
        next_round = self.get_next_round(current_round)
        
        if next_round:
            # Transition to next round
            test.current_round = next_round
            test.state = InterviewState.ROUND_COMPLETE
            
            transition_message = f"Thank you for the {current_round.value.lower()} round. Let's move to the {next_round.value.lower()} round."
            audio_path = await self.audio_manager.text_to_speech(transition_message, test.voice)
            
            return {
                "response": transition_message,
                "audio_path": audio_path,
                "round_complete": True,
                "next_round": next_round.value,
                "current_round": current_round.value
            }
        else:
            # Interview complete
            test.state = InterviewState.INTERVIEW_COMPLETE
            
            completion_message = "Thank you for completing all interview rounds. Generating your evaluation report..."
            audio_path = await self.audio_manager.text_to_speech(completion_message, test.voice)
            
            return {
                "response": completion_message,
                "audio_path": audio_path,
                "interview_complete": True
            }
    
    async def generate_evaluation(self, test_id: str) -> Dict[str, Any]:
        """Generate final interview evaluation"""
        test = self.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found in generate_evaluation")
            raise HTTPException(status_code=404, detail="Test not found")
        
        # Generate evaluation using LLMManager
        evaluation = await self.llm_manager.evaluate_interview(test.conversations)
        scores = extract_scores_from_evaluation(evaluation)
        
        # Save to database
        save_success = self.db_manager.save_interview_data(
            test_id=test_id,
            conversations=test.conversations,
            evaluation=evaluation,
            scores=scores
        )
        
        if not save_success:
            logger.warning(f"Failed to save interview data for test {test_id}")
        
        # Generate analytics
        total_questions = sum(test.questions_asked.values())
        total_responses = sum(len(convs) for convs in test.conversations.values())
        round_durations = {}
        
        # Calculate approximate round durations
        for round_type in RoundType:
            question_count = test.questions_asked[round_type]
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
    
    def cleanup_expired_tests(self):
        """Remove expired tests (older than 4 hours)"""
        current_time = time.time()
        expired_tests = [
            tid for tid, test in self.tests.items()
            if current_time - test.last_activity > 14400  # 4 hours
        ]
        
        for tid in expired_tests:
            self.tests.pop(tid, None)
            logger.info(f"Cleaned up expired test: {tid}")

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
    test_manager.cleanup_expired_tests()
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
test_manager = TestManager()

# ========================
# API Endpoints - UPDATED
# ========================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the interview interface"""
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

@app.get("/start_interview")
async def start_interview():
    """Start a new interview test"""
    try:
        logger.info("Starting new interview...")
        test_id = test_manager.create_test()
        logger.info(f"Test created: {test_id}")
        
        result = await test_manager.generate_first_question(test_id)
        result["test_id"] = test_id
        
        logger.info(f"First question generated for test {test_id}")
        return result
    except Exception as e:
        logger.error(f"Error starting interview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# UPDATED: New endpoint for audio upload instead of recording
@app.post("/record_and_respond")
async def record_and_respond(
    audio: UploadFile = File(...),
    test_id: str = Form(...)
):
    """Process uploaded audio response and provide AI reply"""
    try:
        logger.info(f"Processing uploaded audio for test: {test_id}")
        
        # Check if test exists first
        test = test_manager.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found in record_and_respond")
            # List all available tests for debugging
            available_tests = test_manager.list_tests()
            logger.error(f"Available tests: {available_tests}")
            raise HTTPException(status_code=404, detail=f"Test {test_id} not found")
        
        result = await test_manager.process_user_response_upload(test_id, audio)
        logger.info(f"Response processed successfully for test {test_id}")
        return result
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error processing response for test {test_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/start_next_round")
async def start_next_round(test_id: str):
    """Start the next interview round"""
    try:
        logger.info(f"Starting next round for test: {test_id}")
        test = test_manager.get_test(test_id)
        if not test:
            logger.error(f"Test {test_id} not found in start_next_round")
            raise HTTPException(status_code=404, detail="Test not found")
        
        test.state = InterviewState.IN_PROGRESS
        result = await test_manager.generate_first_question(test_id)
        logger.info(f"Next round started for test {test_id}")
        return result
    except Exception as e:
        logger.error(f"Error starting next round: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate")
async def get_evaluation(test_id: str):
    """Get final interview evaluation"""
    try:
        logger.info(f"Generating evaluation for test: {test_id}")
        result = await test_manager.generate_evaluation(test_id)
        logger.info(f"Evaluation generated for test {test_id}")
        result["pdf_url"] = f"./download_results/{test_id}"
        return result
    except Exception as e:
        logger.error(f"Error generating evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Debug endpoints
@app.get("/debug/tests")
async def debug_tests():
    """Debug endpoint to list all active tests"""
    try:
        tests_info = test_manager.list_tests()
        return {
            "total_tests": len(tests_info),
            "tests": tests_info
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return {"error": str(e)}

@app.get("/debug/test/{test_id}")
async def debug_test(test_id: str):
    """Debug endpoint to get detailed test info"""
    try:
        test = test_manager.get_test(test_id)
        if not test:
            return {"error": "Test not found", "test_id": test_id}
        
        return {
            "test_id": test_id,
            "current_round": test.current_round.value,
            "state": test.state.value,
            "voice": test.voice,
            "questions_asked": test.questions_asked,
            "conversations_count": {
                round_type.value: len(convs) 
                for round_type, convs in test.conversations.items()
            },
            "last_activity": time.ctime(test.last_activity),
            "round_start_time": time.ctime(test.round_start_time)
        }
    except Exception as e:
        logger.error(f"Error in test debug endpoint: {e}")
        return {"error": str(e)}

@app.get("/cleanup")
async def cleanup():
    """Clean up expired tests and old files"""
    try:
        tests_before = len(test_manager.tests)
        test_manager.cleanup_expired_tests()
        tests_after = len(test_manager.tests)
        
        AudioManager.clean_old_audio_files()
        
        return {
            "message": "Cleanup completed",
            "tests_removed": tests_before - tests_after,
            "active_tests": tests_after
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
        "active_tests": len(test_manager.tests),
        "timestamp": time.time()
    }
    
@app.get("/api/download_results/{test_id}")
async def download_interview_pdf(test_id: str):
    """
    Fetch the saved test document from MongoDB and stream it as a PDF.
    """
    # 1) Check MongoDB connection / fetch document
    if not test_manager.db_manager.client:
        raise HTTPException(status_code=500, detail="Database connection unavailable")
    doc = test_manager.db_manager.interviews.find_one({"test_id": test_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Test ID not found")

    try:
        # 2) Create PDF in memory
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=LETTER)
        width, height = LETTER
        margin = 50
        y = height - margin

        # Helper function: writes one line, handles page breaks, resets font
        def write_line(label, value, indent=0):
            nonlocal y
            if y < margin + 50:
                p.showPage()
                p.setFont("Helvetica", 12)  # Reset font on new page
                y = height - margin
            p.drawString(margin + indent, y, f"{label}: {value}")
            y -= 20
            return y

        # 3) Header
        p.setFont("Helvetica-Bold", 14)
        p.drawString(margin, y, f"Interview Summary – Test ID: {test_id}")
        y -= 30

        # 4) Basic fields
        p.setFont("Helvetica", 12)
        name_val = doc.get("name", "N/A")
        y = write_line("Name", str(name_val))

        student_val = doc.get("Student_ID", "N/A")
        y = write_line("Student ID", str(student_val))

        session_val = doc.get("session_id", "N/A")
        y = write_line("Session ID", str(session_val))

        # Timestamp
        try:
            ts = float(doc.get("timestamp", time.time()))
            timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        except:
            timestr = "N/A"
        y = write_line("Saved At", timestr)

        # Scores block
        scores = doc.get("scores", {})
        for key in ["technical_score", "communication_score", "hr_score", "overall_score"]:
            score_val = scores.get(key, "N/A")
            # Capitalize for label (e.g., "Technical Score")
            label = key.replace("_", " ").title()
            y = write_line(label, str(score_val))

        # 5) Evaluation text (splitlines; no wrapping)
        y -= 10
        p.setFont("Helvetica-Bold", 12)
        if y < margin + 50:
            p.showPage()
            p.setFont("Helvetica-Bold", 12)
            y = height - margin
        p.drawString(margin, y, "Evaluation:")
        y -= 20

        p.setFont("Helvetica", 12)
        for line in str(doc.get("evaluation", "")).splitlines():
            y = write_line(" ", line, indent=10)

        # 6) Conversations header
        y -= 10
        p.setFont("Helvetica-Bold", 12)
        if y < margin + 30:
            p.showPage()
            p.setFont("Helvetica-Bold", 12)
            y = height - margin
        p.drawString(margin, y, "Conversations")
        y -= 20

        # 7) Iterate over each round’s conversation entries
        p.setFont("Helvetica", 11)
        convs = doc.get("conversations", {})
        for round_name, entries in convs.items():
            # Round header
            if y < margin + 50:
                p.showPage()
                p.setFont("Helvetica", 11)
                y = height - margin
            y = write_line("Round", round_name)

            for idx, entry in enumerate(entries, start=1):
                # Each entry: Q and A
                if y < margin + 80:
                    p.showPage()
                    p.setFont("Helvetica", 11)
                    y = height - margin
                ai_resp = entry.get("ai_response", "N/A")
                user_inp = entry.get("user_input", "N/A")
                y = write_line(f"  Q{idx}", str(ai_resp), indent=10)
                y = write_line(f"  A{idx}", str(user_inp), indent=10)
                # Optionally show entry timestamp if available
                try:
                    ets = float(entry.get("timestamp", time.time()))
                    etimestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ets))
                except:
                    etimestr = "N/A"
                y = write_line(f"  Time{idx}", etimestr, indent=10)
                y -= 5  # small gap

        # 8) Finalize PDF
        p.showPage()
        p.save()
        buffer.seek(0)

        filename = f"interview_{test_id}.pdf"
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Error while generating PDF for {test_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while generating PDF")


@app.get("/api/interviews", response_class=JSONResponse)
async def get_all_interviews():
    """
    Retrieve all interview documents, excluding conversations and evaluation.
    """
    try:
        results = list(test_manager.db_manager.interviews.find(
            {}, 
            {"_id": 0, "conversations": 0, "evaluation": 0}
        ))
        return {"interviews": results}
    except Exception as e:
        logger.error(f"Error fetching interview results: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interviews")


@app.get("/api/interviews/{test_id}", response_class=JSONResponse)
async def get_interview_by_id(test_id: str):
    """
    Retrieve a specific interview document by test_id, excluding conversations and evaluation.
    """
    try:
        result = test_manager.db_manager.interviews.find_one(
            {"test_id": test_id},
            {"_id": 0, "conversations": 0, "evaluation": 0}
        )
        if not result:
            raise HTTPException(status_code=404, detail="Test not found")
        return result
    except Exception as e:
        logger.error(f"Error fetching interview {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interview")


@app.get("/api/interview-students", response_class=JSONResponse)
async def get_unique_interview_students():
    """
    Return distinct Student_ID and name from interview records.
    """
    try:
        pipeline = [
            {"$group": {"_id": "$Student_ID", "name": {"$first": "$name"}}},
            {"$project": {"_id": 0, "Student_ID": "$_id", "name": 1}}
        ]
        students = list(test_manager.db_manager.interviews.aggregate(pipeline))
        return {"count": len(students), "students": students}
    except Exception as e:
        logger.error(f"Error fetching unique interview students: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interview students")


@app.get("/api/interview-students/{student_id}/interviews", response_class=JSONResponse)
async def get_interviews_for_student(student_id: str):
    """
    Get all interview summaries for a given Student_ID.
    """
    try:
        student_id_int = int(student_id)
        results = list(test_manager.db_manager.interviews.find(
            {"Student_ID": student_id_int},
            {"_id": 0, "conversations": 0, "evaluation": 0}
        ))
        if not results:
            raise HTTPException(status_code=404, detail="No interviews found for this student")
        return {"count": len(results), "interviews": results}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid student ID format")
    except Exception as e:
        logger.error(f"Error fetching interviews for student ID {student_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve interviews")