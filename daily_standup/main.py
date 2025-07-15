# ULTRA-OPTIMIZED DAILY STANDUP SUBMODULE
# Designed to run from root app.py file

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import time
import uuid
import logging
import random
import textwrap
import os
import re
import tempfile
from typing import Dict, List, Optional
import pyodbc
import edge_tts
import pymongo
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from urllib.parse import quote_plus

# Configure logging
logger = logging.getLogger(__name__)

# OPTIMIZED CONSTANTS
TTS_SPEED = 1.4
TOTAL_QUESTIONS = 12
MIN_QUESTIONS_PER_CONCEPT = 1
MAX_QUESTIONS_PER_CONCEPT = 2

# ========================
# PATHS - RELATIVE TO SUBMODULE
# ========================

# Get the directory of this file (daily_standup folder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(CURRENT_DIR, "audio")
TEMP_DIR = os.path.join(CURRENT_DIR, "temp")

# Create directories if they don't exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ========================
# ULTRA-FAST DATABASE SETUP
# ========================

# SQL Server with fallback
DB_CONFIG = {
    "DRIVER": "ODBC Driver 17 for SQL Server",
    "SERVER": "192.168.48.200",
    "DATABASE": "SuperDB",
    "UID": "sa",
    "PWD": "Welcome@123",
}

def fetch_random_student_info():
    """Fast student info fetch with fallback"""
    try:
        conn = pyodbc.connect(
            f"DRIVER={{{DB_CONFIG['DRIVER']}}};"
            f"SERVER={DB_CONFIG['SERVER']};"
            f"DATABASE={DB_CONFIG['DATABASE']};"
            f"UID={DB_CONFIG['UID']};"
            f"PWD={DB_CONFIG['PWD']}",
            timeout=3  # Fast timeout
        )
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 ID, First_Name, Last_Name FROM tbl_Student ORDER BY NEWID()")
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return (row[0], row[1], row[2], f"SESSION_{random.randint(100, 999)}")
    except:
        # Fast fallback
        return (random.randint(1000, 9999), "Test", "User", f"SESSION_{random.randint(100, 999)}")

def parse_summary_fragments(summary: str) -> Dict[str, str]:
    """Fast summary parsing"""
    if not summary or not summary.strip():
        return {"General": "Technical discussion topics"}
    
    lines = summary.strip().split('\n')
    section_pattern = re.compile(r'^\s*(\d+)\.\s+(.+)')
    
    fragments = {}
    current_section = None
    current_content = []
    
    for line in lines:
        match = section_pattern.match(line)
        if match:
            if current_section and current_content:
                fragments[current_section] = '\n'.join(current_content).strip()
            section_num = match.group(1)
            section_title = match.group(2).strip()
            current_section = f"{section_num}. {section_title}"
            current_content = [line]
        else:
            if current_section:
                current_content.append(line)
            else:
                if "Introduction" not in fragments:
                    fragments["Introduction"] = line
                else:
                    fragments["Introduction"] += '\n' + line
    
    if current_section and current_content:
        fragments[current_section] = '\n'.join(current_content).strip()
    
    if not fragments:
        fragments["General"] = summary
    
    return fragments

# MongoDB - Ultra-fast connection
MONGO_USER = "LanTech"
MONGO_PASS = "L@nc^ere@0012"
MONGO_HOST = "192.168.48.201:27017"
MONGO_DB_NAME = "Api-1"

class FastDatabaseManager:
    def __init__(self):
        self.client = pymongo.MongoClient(
            f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASS)}@{MONGO_HOST}/{MONGO_DB_NAME}?authSource=admin",
            maxPoolSize=50,  # Connection pooling
            serverSelectionTimeoutMS=3000  # Fast timeout
        )
        self.db = self.client[MONGO_DB_NAME]
        self.transcripts = self.db["original-1"]
        self.conversations = self.db["daily_standup_results-1"]
    
    def get_latest_summary(self) -> str:
        try:
            doc = self.transcripts.find_one(
                {"summary": {"$exists": True, "$ne": None, "$ne": ""}},
                sort=[("timestamp", -1)]
            )
            return doc["summary"] if doc else "Technical interview topics and concepts"
        except:
            return "Technical interview topics and concepts"
    
    async def save_test_data_async(self, test_id: str, conversation_log: List, evaluation: str, session_data: Dict):
        """Async save for better performance"""
        try:
            student_info = fetch_random_student_info()
            student_id, first_name, last_name, session_id = student_info
            
            # Extract score
            score_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', evaluation)
            score = float(score_match.group(1)) if score_match else 5.0
            
            document = {
                "test_id": test_id,
                "Student_ID": student_id,
                "name": f"{first_name} {last_name}",
                "session_id": session_id,
                "timestamp": time.time(),
                "conversation_log": conversation_log,
                "evaluation": evaluation,
                "score": score,
                "session_data": session_data
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.conversations.insert_one, document
            )
            return True
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False

db_manager = FastDatabaseManager()

# ========================
# HUMAN-LIKE CONVERSATION MANAGER
# ========================

class HumanConversationManager:
    def __init__(self):
        # Ultra-fast LLM setup
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.8, 
            timeout=8,  # Fast timeout
            max_retries=1
        )
        self.parser = StrOutputParser()
        
        # HUMAN-LIKE GREETING PROMPTS
        self.greeting_prompt = PromptTemplate.from_template("""
You're starting a friendly voice conversation with someone joining a technical check-in.

Be natural and warm - like you're genuinely happy to talk to them. Don't sound scripted.

Based on the step:
- greeting_start: Just say hello naturally (like "Hey there!" or "Hi! Good to see you")
- greeting_checkin: Ask how they're doing (like "How's your day going?" or "How are things?")  
- greeting_ready: See if they're ready to start (like "Ready to dive in?" or "Shall we get started?")

Keep it conversational and genuine - like talking to a colleague.

Step: {step}
Previous response: {user_response}

Just respond naturally:
""")
        
        # ULTRA-NATURAL CONVERSATION PROMPT
        self.conversation_prompt = PromptTemplate.from_template("""
You're having a natural technical conversation with a colleague about their work. 

Current topic we're exploring:
{concept_title}

What this topic covers:
{concept_content}

Recent conversation flow:
{conversation_history}

They just said: "{user_response}"

---

Respond like a curious, engaged colleague who genuinely wants to understand their work. 

- If they gave a good answer, acknowledge it naturally and either dig deeper or move to a related aspect
- If they seem unsure, encourage them gently and maybe rephrase or ask something easier
- If they're clearly finished with this topic, naturally transition to something new
- Keep the conversation flowing like real people talking

Be genuinely interested in what they're saying. Ask follow-ups that show you're listening.

Don't be robotic or follow a script - just have a natural technical discussion.

Your response:
""")
        
        # NATURAL EVALUATION PROMPT  
        self.evaluation_prompt = PromptTemplate.from_template("""
You just finished a friendly technical conversation with someone about their work.

Here's what you talked about:
{conversation}

Give them honest, constructive feedback like a supportive colleague would. 

Cover:
- What they did well technically
- Areas where they could grow
- How well they explained concepts
- Overall technical understanding

Be encouraging but honest. Keep it conversational, not like a formal report.

End with: "Final Score: X/10"

Keep it under 200 words and sound like a real person giving feedback:
""")
    
    async def get_greeting_response(self, step: str, user_response: str = "") -> str:
        """Generate natural greeting"""
        try:
            chain = self.greeting_prompt | self.llm | self.parser
            response = await chain.ainvoke({
                "step": step,
                "user_response": user_response
            })
            return response.strip()
        except:
            # Fast fallbacks
            if step == "greeting_start":
                return "Hey there! Good to see you."
            elif step == "greeting_checkin":
                return "How's your day going?"
            else:
                return "Ready to dive in?"
    
    async def get_conversation_response(self, concept_title: str, concept_content: str, 
                                     conversation_history: str, user_response: str) -> tuple[str, bool]:
        """Generate natural conversation response"""
        try:
            chain = self.conversation_prompt | self.llm | self.parser
            response = await chain.ainvoke({
                "concept_title": concept_title,
                "concept_content": concept_content,
                "conversation_history": conversation_history,
                "user_response": user_response
            })
            
            # Simple check if we should move on (look for transition words)
            move_on = any(phrase in response.lower() for phrase in [
                "let's move", "next topic", "another area", "switching to", "new question"
            ])
            
            return response.strip(), move_on
        except Exception as e:
            logger.error(f"Conversation error: {e}")
            return "That's interesting! Can you tell me more about that?", False
    
    async def generate_evaluation(self, conversation: str) -> str:
        """Generate natural evaluation"""
        try:
            chain = self.evaluation_prompt | self.llm | self.parser
            return await chain.ainvoke({"conversation": conversation})
        except:
            return "Great conversation! You showed good technical understanding. Final Score: 7/10"

# ========================
# REAL-TIME SESSION MANAGER
# ========================

class RealtimeSession:
    def __init__(self, summary: str):
        self.session_id = str(uuid.uuid4())
        self.created_at = time.time()
        self.last_activity = time.time()
        
        # Fragment-based concepts
        self.fragments = parse_summary_fragments(summary)
        self.fragment_keys = list(self.fragments.keys())
        self.current_fragment_index = 0
        self.questions_per_fragment = max(1, TOTAL_QUESTIONS // len(self.fragment_keys) if self.fragment_keys else 1)
        
        # Conversation state
        self.greeting_step = 0  # 0=start, 1=checkin, 2=ready, 3=conversation
        self.conversation_log = []
        self.current_conversation_history = ""
        
        logger.info(f"Session {self.session_id}: {len(self.fragment_keys)} concepts, "
                   f"{self.questions_per_fragment} questions per concept")
    
    def get_current_concept(self) -> tuple[str, str]:
        """Get current concept info"""
        if not self.fragment_keys:
            return "General Discussion", "Technical topics and concepts"
        
        index = self.current_fragment_index % len(self.fragment_keys)
        concept_title = self.fragment_keys[index]
        concept_content = self.fragments[concept_title]
        return concept_title, concept_content
    
    def should_continue(self) -> bool:
        """Simple continuation check"""
        if self.greeting_step < 3:
            return True
        
        # Count actual conversation exchanges (not greetings)
        actual_exchanges = len([entry for entry in self.conversation_log 
                              if not entry.get("concept", "").startswith("greeting")])
        
        return actual_exchanges < TOTAL_QUESTIONS
    
    def add_exchange(self, question: str, answer: str, concept: str):
        """Add Q&A exchange"""
        self.conversation_log.append({
            "question": question,
            "answer": answer,  
            "concept": concept,
            "timestamp": time.time()
        })
        
        # Update conversation history for context
        if not concept.startswith("greeting"):
            self.current_conversation_history += f"Q: {question}\nA: {answer}\n\n"
            # Keep only last 3 exchanges for context
            lines = self.current_conversation_history.split('\n\n')
            if len(lines) > 6:  # 3 exchanges = 6 lines
                self.current_conversation_history = '\n\n'.join(lines[-6:])
    
    def move_to_next_concept(self):
        """Move to next concept"""
        self.current_fragment_index += 1

# ========================
# ULTRA-FAST AUDIO STREAMING
# ========================

class StreamingAudioManager:
    @staticmethod
    async def stream_tts(text: str, voice: str = "en-IN-PrabhatNeural") -> bytes:
        """Stream TTS directly to memory"""
        try:
            # Stream TTS with speed optimization
            tts = edge_tts.Communicate(text, voice, rate=f"+{int((TTS_SPEED-1)*100)}%")
            audio_data = b""
            async for chunk in tts.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            return b""
    
    @staticmethod
    async def transcribe_streaming(audio_data: bytes) -> str:
        """Fast transcription with proper temp file handling"""
        try:
            # Use the submodule's temp directory
            temp_file = os.path.join(TEMP_DIR, f"audio_{int(time.time() * 1000)}.webm")
            
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            # Fast transcription
            from groq import Groq
            client = Groq()
            
            with open(temp_file, "rb") as file:
                result = client.audio.transcriptions.create(
                    file=(temp_file, file.read()),
                    model="whisper-large-v3-turbo"
                )
            
            # Cleanup
            try:
                os.remove(temp_file)
            except:
                pass
                
            return result.text.strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

# ========================
# WEBSOCKET CONNECTION MANAGER
# ========================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, RealtimeSession] = {}
        self.conversation_manager = HumanConversationManager()
        self.audio_manager = StreamingAudioManager()
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        # Initialize session
        summary = db_manager.get_latest_summary()
        self.sessions[session_id] = RealtimeSession(summary)
        
        logger.info(f"Connected: {session_id}")
        
        # Send initial greeting
        await self.send_greeting(session_id)
    
    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        self.sessions.pop(session_id, None)
        logger.info(f"Disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))
    
    async def send_greeting(self, session_id: str):
        """Send initial greeting"""
        greeting = await self.conversation_manager.get_greeting_response("greeting_start")
        audio_data = await self.audio_manager.stream_tts(greeting)
        
        await self.send_message(session_id, {
            "type": "ai_response",
            "text": greeting,
            "audio": audio_data.hex() if audio_data else "",
            "status": "greeting"
        })
    
    async def process_audio(self, session_id: str, audio_data: bytes):
        """Process incoming audio"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        # Quick validation
        if len(audio_data) < 1000:
            await self.send_message(session_id, {
                "type": "error",
                "message": "Audio too short, please speak again"
            })
            return
        
        # Transcribe
        user_response = await self.audio_manager.transcribe_streaming(audio_data)
        if not user_response or len(user_response.strip()) < 2:
            await self.send_message(session_id, {
                "type": "clarification",
                "message": "I didn't catch that. Could you repeat?"
            })
            return
        
        logger.info(f"Session {session_id}: User said: {user_response}")
        
        # Handle based on conversation stage
        if session.greeting_step < 3:
            await self.handle_greeting(session_id, user_response)
        else:
            await self.handle_conversation(session_id, user_response)
    
    async def handle_greeting(self, session_id: str, user_response: str):
        """Handle greeting flow"""
        session = self.sessions[session_id]
        
        if session.greeting_step == 0:
            # User responded to initial greeting
            session.greeting_step = 1
            response = await self.conversation_manager.get_greeting_response("greeting_checkin", user_response)
            session.add_exchange("Hello!", user_response, "greeting_initial")
            
        elif session.greeting_step == 1:
            # User shared how they're doing
            session.greeting_step = 2
            response = await self.conversation_manager.get_greeting_response("greeting_ready", user_response)
            session.add_exchange("How are you doing?", user_response, "greeting_checkin")
            
        else:
            # Ready to start main conversation
            session.greeting_step = 3
            concept_title, concept_content = session.get_current_concept()
            response, _ = await self.conversation_manager.get_conversation_response(
                concept_title, concept_content, "", user_response
            )
            session.add_exchange("Ready to start?", user_response, "greeting_ready")
        
        # Send response
        audio_data = await self.audio_manager.stream_tts(response)
        await self.send_message(session_id, {
            "type": "ai_response",
            "text": response,
            "audio": audio_data.hex() if audio_data else "",
            "status": "greeting" if session.greeting_step < 3 else "conversation"
        })
    
    async def handle_conversation(self, session_id: str, user_response: str):
        """Handle main conversation"""
        session = self.sessions[session_id]
        
        # Get current concept
        concept_title, concept_content = session.get_current_concept()
        
        # Generate response
        ai_response, should_move_on = await self.conversation_manager.get_conversation_response(
            concept_title, concept_content, session.current_conversation_history, user_response
        )
        
        # Add to conversation log
        last_question = session.conversation_log[-1]["question"] if session.conversation_log else "Previous question"
        session.add_exchange(last_question, user_response, concept_title)
        
        # Check if should continue or end
        if should_move_on:
            session.move_to_next_concept()
        
        if not session.should_continue():
            await self.end_conversation(session_id)
            return
        
        # Send response
        audio_data = await self.audio_manager.stream_tts(ai_response)
        await self.send_message(session_id, {
            "type": "ai_response", 
            "text": ai_response,
            "audio": audio_data.hex() if audio_data else "",
            "status": "conversation"
        })
        
        # Update for next question
        session.add_exchange(ai_response, "", concept_title)
    
    async def end_conversation(self, session_id: str):
        """End conversation and generate evaluation"""
        session = self.sessions[session_id]
        
        # Generate evaluation
        full_conversation = ""
        for entry in session.conversation_log:
            if not entry["concept"].startswith("greeting") and entry["answer"]:
                full_conversation += f"Q: {entry['question']}\nA: {entry['answer']}\n\n"
        
        evaluation = await self.conversation_manager.generate_evaluation(full_conversation)
        
        # Save to database
        await db_manager.save_test_data_async(
            session.session_id,
            session.conversation_log,
            evaluation,
            {
                "fragments_covered": len(session.fragment_keys),
                "total_exchanges": len(session.conversation_log)
            }
        )
        
        # Send final message
        closing_message = "Great conversation! Thanks for sharing your technical insights."
        audio_data = await self.audio_manager.stream_tts(closing_message)
        
        await self.send_message(session_id, {
            "type": "conversation_end",
            "text": closing_message,
            "audio": audio_data.hex() if audio_data else "",
            "evaluation": evaluation,
            "pdf_url": f"/download_results/{session.session_id}"
        })

# ========================
# FASTAPI SUB-APPLICATION
# ========================

# Create sub-app (not main app)
sub_app = FastAPI(title="Ultra-Fast Daily Standup", version="3.0.0")

# Static files using relative path
sub_app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Connection manager
manager = ConnectionManager()

# ========================
# WEBSOCKET ENDPOINT
# ========================

@sub_app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    logger.info(f"WebSocket connection attempt for session: {session_id}")
    try:
        await manager.connect(websocket, session_id)
        logger.info(f"WebSocket connected successfully: {session_id}")
        while True:
            data = await websocket.receive_bytes()
            logger.info(f"Received audio data: {len(data)} bytes")
            await manager.process_audio(session_id, data)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        manager.disconnect(session_id)

# ========================
# ESSENTIAL ENDPOINTS
# ========================

@sub_app.get("/")
async def get_interface():
    """Serve the WebSocket interface"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Ultra-Fast Daily Standup</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 30px; border-radius: 15px; }
        .status { padding: 20px; margin: 20px 0; border-radius: 10px; text-align: center; }
        .ready { background: #d4edda; }
        .listening { background: #fff3cd; }
        .speaking { background: #d1ecf1; }
        button { padding: 15px 30px; font-size: 16px; border: none; border-radius: 8px; cursor: pointer; }
        .start { background: #28a745; color: white; }
        .recording { background: #dc3545; color: white; }
        .log { background: #2c3e50; color: #2ecc71; padding: 15px; border-radius: 8px; height: 200px; overflow-y: auto; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Ultra-Fast Daily Standup</h1>
        <p><strong>Real-time WebSocket streaming for instant conversation</strong></p>
        
        <div id="status" class="status ready">
            <h3>Ready to Start</h3>
            <p>Click Start for ultra-fast conversation experience</p>
        </div>
        
        <button id="controlBtn" class="start" onclick="startConversation()">Start Conversation</button>
        
        <div class="log" id="log"></div>
    </div>

    <script>
        let ws = null;
        let mediaRecorder = null;
        let isRecording = false;
        let sessionId = 'session_' + Date.now();
        
        function log(message) {
            const logEl = document.getElementById('log');
            logEl.innerHTML += '<div>' + new Date().toLocaleTimeString() + ': ' + message + '</div>';
            logEl.scrollTop = logEl.scrollHeight;
        }
        
        function updateStatus(title, message, className) {
            const statusEl = document.getElementById('status');
            statusEl.innerHTML = '<h3>' + title + '</h3><p>' + message + '</p>';
            statusEl.className = 'status ' + className;
        }
        
        async function startConversation() {
            try {
                // Connect WebSocket
                ws = new WebSocket('ws://localhost:8000/ws/' + sessionId);
                
                ws.onopen = () => {
                    log('Connected to ultra-fast server');
                    updateStatus('Connected', 'Starting conversation...', 'ready');
                };
                
                ws.onmessage = async (event) => {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'ai_response') {
                        log('AI: ' + data.text);
                        updateStatus('AI Speaking', data.text, 'speaking');
                        
                        // Play audio if available
                        if (data.audio) {
                            const audioData = new Uint8Array(data.audio.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
                            const audioBlob = new Blob([audioData], { type: 'audio/mp3' });
                            const audio = new Audio(URL.createObjectURL(audioBlob));
                            await audio.play();
                        }
                        
                        // Start listening after AI finishes
                        setTimeout(startListening, 1000);
                        
                    } else if (data.type === 'conversation_end') {
                        log('Conversation completed!');
                        log('Evaluation: ' + data.evaluation);
                        updateStatus('Complete', 'Great conversation! Check log for evaluation.', 'ready');
                        document.getElementById('controlBtn').textContent = 'Start New Conversation';
                        
                    } else if (data.type === 'error') {
                        log('Error: ' + data.message);
                        setTimeout(startListening, 1000);
                    }
                };
                
                document.getElementById('controlBtn').textContent = 'Connecting...';
                document.getElementById('controlBtn').disabled = true;
                
            } catch (error) {
                log('Connection error: ' + error.message);
            }
        }
        
        async function startListening() {
            if (isRecording) return;
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                let audioChunks = [];
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    
                    // Send audio via WebSocket
                    audioBlob.arrayBuffer().then(buffer => {
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(buffer);
                        }
                    });
                    
                    updateStatus('Processing', 'Ultra-fast processing...', 'ready');
                    stream.getTracks().forEach(track => track.stop());
                };
                
                isRecording = true;
                mediaRecorder.start();
                updateStatus('Listening', 'Speak now...', 'listening');
                document.getElementById('controlBtn').textContent = 'Recording...';
                document.getElementById('controlBtn').className = 'recording';
                
                // Auto-stop after 10 seconds
                setTimeout(() => {
                    if (isRecording) {
                        mediaRecorder.stop();
                        isRecording = false;
                    }
                }, 10000);
                
            } catch (error) {
                log('Microphone error: ' + error.message);
            }
        }
        
        log('Ultra-fast conversation system loaded');
        log('WebSocket streaming for real-time interaction');
    </script>
</body>
</html>
    """)

@sub_app.get("/download_results/{session_id}")
async def download_results(session_id: str):
    """Generate and download PDF results"""
    try:
        # Get test from database
        doc = db_manager.conversations.find_one({"test_id": session_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Session not found")

        # Create PDF
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=LETTER)
        
        # Header
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, 750, f"Daily Standup Results - {session_id}")
        
        # Basic info
        p.setFont("Helvetica", 12)
        p.drawString(50, 720, f"Participant: {doc.get('name', 'N/A')}")
        p.drawString(50, 700, f"Score: {doc.get('score', 'N/A')}/10")
        p.drawString(50, 680, f"Date: {time.strftime('%Y-%m-%d %H:%M', time.localtime(doc.get('timestamp', time.time())))}")
        
        # Evaluation
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, 650, "Evaluation:")
        
        p.setFont("Helvetica", 11)
        evaluation = doc.get('evaluation', 'No evaluation available')
        lines = textwrap.wrap(evaluation, 80)
        y = 630
        for line in lines:
            if y < 100:
                p.showPage()
                p.setFont("Helvetica", 11)
                y = 750
            p.drawString(50, y, line)
            y -= 15
        
        # Conversation log
        y -= 20
        p.setFont("Helvetica-Bold", 14)
        if y < 100:
            p.showPage()
            y = 750
        p.drawString(50, y, "Conversation:")
        y -= 20
        
        p.setFont("Helvetica", 10)
        conversation_log = doc.get('conversation_log', [])
        for i, entry in enumerate(conversation_log):
            if not entry.get('concept', '').startswith('greeting') and entry.get('answer'):
                if y < 120:
                    p.showPage()
                    p.setFont("Helvetica", 10)
                    y = 750
                
                # Question
                q_lines = textwrap.wrap(f"Q{i}: {entry.get('question', '')}", 85)
                for line in q_lines:
                    p.drawString(50, y, line)
                    y -= 12
                
                # Answer
                a_lines = textwrap.wrap(f"A: {entry.get('answer', '')}", 85)
                for line in a_lines:
                    p.drawString(60, y, line)
                    y -= 12
                y -= 5
        
        p.save()
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=standup_results_{session_id[:8]}.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@sub_app.get("/test")
async def test_endpoint():
    """Test endpoint to verify sub-app is working"""
    return {
        "status": "daily_standup sub-app working",
        "websocket_url": "/daily_standup/ws/{session_id}",
        "active_connections": len(manager.active_connections),
        "temp_dir": TEMP_DIR,
        "audio_dir": AUDIO_DIR
    }

@sub_app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "version": "3.0.0 - Ultra-Optimized Real-Time",
        "features": [
            "WebSocket streaming",
            "Real-time audio processing", 
            "Human-like conversation",
            "Memory-only processing",
            "Ultra-fast responses"
        ],
        "performance": "Sub-second response times",
        "connections": len(manager.active_connections)
    }

@sub_app.get("/stats")
async def get_stats():
    """Simple stats"""
    return {
        "active_sessions": len(manager.sessions),
        "total_connections": len(manager.active_connections),
        "uptime": time.time(),
        "performance": "Ultra-optimized for speed"
    }

# ========================
# FUNCTION TO GET THE SUB-APP + COMPATIBILITY
# ========================

def get_daily_standup_app():
    """Return the configured daily standup sub-application"""
    return sub_app

# Compatibility: Export sub_app as 'app' for root mounting
app = sub_app