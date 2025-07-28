# weekly_interview/main.py
"""
Enhanced Mock Interview System - Main Application
Production-ready FastAPI application with WebSocket support and modular architecture
"""

import os
import time
import uuid
import logging
import asyncio
import json
import base64
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager
import io
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import textwrap

# Local imports
from .core.config import config
from .core.database import DatabaseManager, get_db_manager
from .core.ai_services import (
    shared_clients, InterviewSessionManager, OptimizedAudioProcessor,
    UltraFastTTSProcessor, OptimizedConversationManager, InterviewStage, InterviewState
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class StartInterviewRequest(BaseModel):
    websocket_endpoint: Optional[str] = None

class StartInterviewResponse(BaseModel):
    test_id: str
    session_id: str
    websocket_url: str
    student_name: str
    estimated_duration_minutes: int
    message: str

class EvaluationResponse(BaseModel):
    evaluation: str
    scores: Dict[str, float]
    analytics: Dict[str, Any]
    pdf_url: str

# =============================================================================
# ENHANCED INTERVIEW MANAGER
# =============================================================================

class EnhancedInterviewManager:
    """Complete interview management with WebSocket and audio processing"""
    
    def __init__(self):
        self.session_manager = None
        self.audio_processor = None
        self.tts_processor = None
        self.conversation_manager = None
        self.db_manager = None
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize database manager
            self.db_manager = get_db_manager(shared_clients)
            await self.db_manager.initialize()
            
            # Initialize managers
            self.session_manager = InterviewSessionManager(self.db_manager)
            self.audio_processor = OptimizedAudioProcessor(shared_clients)
            self.tts_processor = UltraFastTTSProcessor()
            self.conversation_manager = OptimizedConversationManager(shared_clients)
            
            logger.info("✅ Enhanced Interview Manager initialized")
            
        except Exception as e:
            logger.error(f"❌ Interview Manager initialization failed: {e}")
            raise Exception(f"Interview Manager initialization failed: {e}")
    
    async def start_interview_session(self, websocket: Optional[WebSocket] = None) -> Dict[str, Any]:
        """Start new interview session with WebSocket support"""
        try:
            # Create session
            session = await self.session_manager.create_session_fast(websocket)
            
            # Generate initial greeting
            greeting = await self.conversation_manager.generate_interview_response(session)
            
            # Add greeting to session
            session.add_exchange(greeting, "", 0.0, False, 0.0)
            session.current_state = InterviewState.IN_PROGRESS
            
            # Start round timing
            session.round_start_times["greeting"] = time.time()
            
            return {
                "test_id": session.test_id,
                "session_id": session.session_id,
                "websocket_url": f"{config.API_PREFIX}{config.WEBSOCKET_ENDPOINT}/{session.session_id}",
                "student_name": session.student_name,
                "estimated_duration_minutes": config.INTERVIEW_DURATION_MINUTES,
                "greeting": greeting,
                "message": "Interview session created successfully"
            }
            
        except Exception as e:
            logger.error(f"❌ Start interview failed: {e}")
            raise Exception(f"Failed to start interview: {e}")
    
    async def process_audio_message(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
        """Process audio message and generate response"""
        try:
            session = self.session_manager.validate_session(session_id)
            start_time = time.time()
            
            # Transcribe audio
            transcript, quality = await self.audio_processor.transcribe_audio_fast(audio_data)
            
            if not transcript or len(transcript.strip()) < 2:
                return {
                    "type": "clarification",
                    "message": "I didn't hear your response clearly. Could you please repeat that?",
                    "status": session.current_stage.value
                }
            
            logger.info(f"🎙️ User response: '{transcript}' (quality: {quality:.2f})")
            
            # Update last exchange with user response
            if session.exchanges:
                session.update_last_response(transcript, quality)
            
            # Check for stage transitions
            await self._check_stage_transitions(session)
            
            # Generate AI response
            ai_response = await self.conversation_manager.generate_interview_response(session, transcript)
            
            # Add new exchange
            processing_time = time.time() - start_time
            is_followup = self._determine_if_followup(session, ai_response)
            session.add_exchange(ai_response, "", 0.0, is_followup, processing_time)
            
            return {
                "type": "ai_response",
                "message": ai_response,
                "stage": session.current_stage.value,
                "question_number": session.questions_per_round[session.current_stage.value],
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Audio processing failed: {e}")
            return {
                "type": "error",
                "message": f"Processing failed: {str(e)}",
                "status": "error"
            }
    
    async def _check_stage_transitions(self, session):
        """Check and handle stage transitions"""
        current_stage = session.current_stage
        questions_in_stage = session.questions_per_round[current_stage.value]
        
        # Transition logic based on question counts and stage
        if current_stage == InterviewStage.GREETING and questions_in_stage >= 2:
            session.current_stage = InterviewStage.TECHNICAL
            session.round_start_times["technical"] = time.time()
            logger.info(f"🔄 Transitioned to TECHNICAL stage")
            
        elif current_stage == InterviewStage.TECHNICAL and questions_in_stage >= config.MAX_QUESTIONS_PER_ROUND:
            session.current_stage = InterviewStage.COMMUNICATION
            session.round_start_times["communication"] = time.time()
            logger.info(f"🔄 Transitioned to COMMUNICATION stage")
            
        elif current_stage == InterviewStage.COMMUNICATION and questions_in_stage >= config.MAX_QUESTIONS_PER_ROUND:
            session.current_stage = InterviewStage.HR
            session.round_start_times["hr"] = time.time()
            logger.info(f"🔄 Transitioned to HR stage")
            
        elif current_stage == InterviewStage.HR and questions_in_stage >= config.MAX_QUESTIONS_PER_ROUND:
            session.current_stage = InterviewStage.COMPLETE
            session.current_state = InterviewState.COMPLETED
            logger.info(f"🏁 Interview completed")
    
    def _determine_if_followup(self, session, ai_response: str) -> bool:
        """Determine if the current question is a follow-up"""
        followup_indicators = [
            "follow up", "elaborate", "can you explain", "tell me more",
            "what about", "how did you", "could you describe"
        ]
        
        return any(indicator in ai_response.lower() for indicator in followup_indicators)
    
    async def generate_tts_stream(self, session_id: str, text: str):
        """Generate TTS audio stream for WebSocket"""
        try:
            session = self.session_manager.validate_session(session_id)
            
            async for audio_chunk in self.tts_processor.generate_audio_stream(text):
                if audio_chunk and session.is_active:
                    yield audio_chunk
                    
        except Exception as e:
            logger.error(f"❌ TTS streaming failed: {e}")
    
    async def complete_interview(self, session_id: str) -> Dict[str, Any]:
        """Complete interview and generate evaluation"""
        try:
            session = self.session_manager.validate_session(session_id)
            
            # Generate comprehensive evaluation
            evaluation, scores = await self.conversation_manager.generate_comprehensive_evaluation(session)
            
            # Prepare session analytics
            analytics = {
                "total_duration_minutes": round((time.time() - session.created_at) / 60, 1),
                "questions_per_round": dict(session.questions_per_round),
                "total_questions": session.total_questions_asked,
                "followup_questions": session.followup_questions,
                "audio_quality_avg": round(sum(session.audio_quality_scores) / len(session.audio_quality_scores), 2) if session.audio_quality_scores else 0,
                "stage_completion_times": self._calculate_stage_times(session),
                "response_quality_metrics": self._calculate_response_metrics(session)
            }
            
            # Save to database
            interview_data = {
                "test_id": session.test_id,
                "session_id": session.session_id,
                "student_id": session.student_id,
                "student_name": session.student_name,
                "conversation_log": [
                    {
                        "timestamp": ex.timestamp,
                        "stage": ex.stage.value,
                        "ai_message": ex.ai_message,
                        "user_response": ex.user_response,
                        "transcript_quality": ex.transcript_quality,
                        "is_followup": ex.is_followup
                    }
                    for ex in session.exchanges
                ],
                "evaluation": evaluation,
                "scores": scores,
                "duration_minutes": analytics["total_duration_minutes"],
                "questions_per_round": analytics["questions_per_round"],
                "followup_questions": analytics["followup_questions"],
                "avg_response_time": session.avg_response_time,
                "audio_metrics": {"avg_quality": analytics["audio_quality_avg"]},
                "flow_analytics": analytics["stage_completion_times"],
                "websocket_used": True,
                "tts_voice": session.tts_voice
            }
            
            await self.db_manager.save_interview_result_fast(interview_data)
            
            # Cleanup session
            self.session_manager.cleanup_session(session_id)
            
            logger.info(f"✅ Interview completed and saved: {session.test_id}")
            
            return {
                "evaluation": evaluation,
                "scores": scores,
                "analytics": analytics,
                "pdf_url": f"{config.API_PREFIX}/download_results/{session.test_id}"
            }
            
        except Exception as e:
            logger.error(f"❌ Interview completion failed: {e}")
            raise Exception(f"Interview completion failed: {e}")
    
    def _calculate_stage_times(self, session) -> Dict[str, float]:
        """Calculate time spent in each stage"""
        stage_times = {}
        
        for stage_name, start_time in session.round_start_times.items():
            if start_time:
                # Find end time (start of next stage or current time)
                stage_names = ["greeting", "technical", "communication", "hr"]
                try:
                    current_idx = stage_names.index(stage_name)
                    if current_idx < len(stage_names) - 1:
                        next_stage = stage_names[current_idx + 1]
                        end_time = session.round_start_times.get(next_stage, time.time())
                    else:
                        end_time = time.time()
                    
                    stage_times[stage_name] = round((end_time - start_time) / 60, 1)  # Minutes
                except ValueError:
                    stage_times[stage_name] = 0.0
        
        return stage_times
    
    def _calculate_response_metrics(self, session) -> Dict[str, Any]:
        """Calculate response quality metrics"""
        responses = [ex.user_response for ex in session.exchanges if ex.user_response]
        
        if not responses:
            return {"avg_length": 0, "total_responses": 0}
        
        avg_length = sum(len(response.split()) for response in responses) / len(responses)
        
        return {
            "avg_length": round(avg_length, 1),
            "total_responses": len(responses),
            "short_responses": len([r for r in responses if len(r.split()) < 10]),
            "detailed_responses": len([r for r in responses if len(r.split()) > 30])
        }
    
    async def get_interview_result(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get interview result by test ID"""
        try:
            return await self.db_manager.get_interview_result_fast(test_id)
        except Exception as e:
            logger.error(f"❌ Get interview result failed: {e}")
            return None
    
    def cleanup_expired_sessions(self):
        """Cleanup expired sessions"""
        if self.session_manager:
            return self.session_manager.cleanup_expired_sessions()
        return 0

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Enhanced Mock Interview System starting...")
    
    try:
        # Initialize interview manager
        await interview_manager.initialize()
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
        logger.info("✅ Enhanced Mock Interview System operational")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise Exception(f"Application startup failed: {e}")
    
    yield
    
    # Cleanup on shutdown
    logger.info("👋 Shutting down Enhanced Mock Interview System...")
    try:
        cleanup_task.cancel()
        await shared_clients.close_connections()
        if interview_manager.db_manager:
            await interview_manager.db_manager.close_connections()
        logger.info("✅ Graceful shutdown completed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

async def periodic_cleanup():
    """Periodic cleanup task"""
    while True:
        try:
            await asyncio.sleep(config.CLEANUP_INTERVAL)
            cleaned = interview_manager.cleanup_expired_sessions()
            if cleaned > 0:
                logger.info(f"🧹 Cleaned up {cleaned} expired sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"❌ Cleanup task error: {e}")

# Create FastAPI application
app = FastAPI(
    title=config.APP_TITLE,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS,
    allow_headers=config.CORS_ALLOW_HEADERS,
)

# Mount static files
if config.AUDIO_DIR.exists():
    app.mount("/audio", StaticFiles(directory=str(config.AUDIO_DIR)), name="audio")

# Initialize interview manager
interview_manager = EnhancedInterviewManager()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        health_status = {
            "status": "healthy",
            "service": config.APP_TITLE,
            "version": config.APP_VERSION,
            "timestamp": time.time()
        }
        
        # Database health
        if interview_manager.db_manager:
            db_health = await interview_manager.db_manager.health_check()
            health_status["database"] = db_health
            health_status["overall_healthy"] = db_health["overall"]
        else:
            health_status["database"] = {"status": "not_initialized"}
            health_status["overall_healthy"] = False
        
        # Active sessions
        if interview_manager.session_manager:
            health_status["active_sessions"] = len(interview_manager.session_manager.active_sessions)
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@app.post("/start_interview", response_model=StartInterviewResponse)
async def start_interview():
    """Start new interview session"""
    try:
        result = await interview_manager.start_interview_session()
        
        return StartInterviewResponse(
            test_id=result["test_id"],
            session_id=result["session_id"],
            websocket_url=result["websocket_url"],
            student_name=result["student_name"],
            estimated_duration_minutes=result["estimated_duration_minutes"],
            message=result["message"]
        )
        
    except Exception as e:
        logger.error(f"❌ Start interview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Enhanced WebSocket endpoint for real-time interview communication"""
    await websocket.accept()
    
    try:
        logger.info(f"🔌 WebSocket connected for session: {session_id}")
        
        # Validate session
        session = interview_manager.session_manager.get_session(session_id)
        if not session:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Interview session not found. Please start a new interview.",
                "status": "error"
            }))
            return
        
        # Update session with websocket
        session.websocket = websocket
        
        # Send initial greeting with audio
        if session.exchanges:
            initial_greeting = session.exchanges[0].ai_message
            
            # Send text response
            await websocket.send_text(json.dumps({
                "type": "ai_response",
                "message": initial_greeting,
                "stage": session.current_stage.value,
                "status": "greeting"
            }))
            
            # Generate and stream audio
            try:
                audio_chunks = []
                async for audio_chunk in interview_manager.generate_tts_stream(session_id, initial_greeting):
                    if audio_chunk:
                        audio_chunks.append(audio_chunk)
                
                # Combine audio chunks and send
                if audio_chunks:
                    combined_audio = b''.join(audio_chunks)
                    audio_b64 = base64.b64encode(combined_audio).decode('utf-8')
                    
                    await websocket.send_text(json.dumps({
                        "type": "audio_data",
                        "audio": audio_b64,
                        "status": "greeting"
                    }))
                
                await websocket.send_text(json.dumps({
                    "type": "audio_end",
                    "status": "greeting"
                }))
                
            except Exception as e:
                logger.error(f"❌ Initial audio generation failed: {e}")
        
        # Main WebSocket communication loop
        while session.is_active and session.current_stage != InterviewStage.COMPLETE:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=config.WEBSOCKET_TIMEOUT
                )
                
                data = json.loads(message)
                
                if data.get("type") == "audio_data":
                    # Process audio message
                    audio_b64 = data.get("audio", "")
                    if audio_b64:
                        try:
                            audio_data = base64.b64decode(audio_b64)
                            
                            # Process audio and get response
                            response = await interview_manager.process_audio_message(session_id, audio_data)
                            
                            # Send text response
                            await websocket.send_text(json.dumps(response))
                            
                            # Generate and send audio if it's an AI response
                            if response.get("type") == "ai_response":
                                message_text = response.get("message", "")
                                
                                try:
                                    # Generate TTS audio
                                    audio_chunks = []
                                    async for audio_chunk in interview_manager.generate_tts_stream(session_id, message_text):
                                        if audio_chunk:
                                            audio_chunks.append(audio_chunk)
                                    
                                    # Send audio data
                                    if audio_chunks:
                                        combined_audio = b''.join(audio_chunks)
                                        audio_b64 = base64.b64encode(combined_audio).decode('utf-8')
                                        
                                        await websocket.send_text(json.dumps({
                                            "type": "audio_data",
                                            "audio": audio_b64,
                                            "status": response.get("stage", "unknown")
                                        }))
                                    
                                    # Signal audio end
                                    await websocket.send_text(json.dumps({
                                        "type": "audio_end",
                                        "status": response.get("stage", "unknown")
                                    }))
                                    
                                except Exception as e:
                                    logger.error(f"❌ TTS generation failed: {e}")
                                    await websocket.send_text(json.dumps({
                                        "type": "error",
                                        "message": "Audio generation failed",
                                        "status": "error"
                                    }))
                            
                        except Exception as e:
                            logger.error(f"❌ Audio processing failed: {e}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": "Audio processing failed",
                                "status": "error"
                            }))
                
                elif data.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
                elif data.get("type") == "complete_interview":
                    # Handle interview completion
                    try:
                        result = await interview_manager.complete_interview(session_id)
                        await websocket.send_text(json.dumps({
                            "type": "interview_complete",
                            "result": result
                        }))
                        break
                    except Exception as e:
                        logger.error(f"❌ Interview completion failed: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Interview completion failed",
                            "status": "error"
                        }))
                
            except asyncio.TimeoutError:
                logger.info(f"⏰ WebSocket timeout for session: {session_id}")
                await websocket.send_text(json.dumps({
                    "type": "timeout",
                    "message": "Session timeout due to inactivity",
                    "status": "timeout"
                }))
                break
                
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected: {session_id}")
                break
                
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Communication error: {str(e)}",
                    "status": "error"
                }))
                break
        
        # Check for interview completion
        if session.current_stage == InterviewStage.COMPLETE:
            try:
                result = await interview_manager.complete_interview(session_id)
                await websocket.send_text(json.dumps({
                    "type": "interview_complete",
                    "result": result
                }))
            except Exception as e:
                logger.error(f"❌ Final completion failed: {e}")
    
    except Exception as e:
        logger.error(f"❌ WebSocket endpoint error: {e}")
    
    finally:
        # Cleanup session
        if session_id in interview_manager.session_manager.active_sessions:
            interview_manager.session_manager.cleanup_session(session_id)

@app.get("/evaluate")
async def get_evaluation(test_id: str):
    """Get interview evaluation (for compatibility)"""
    try:
        result = await interview_manager.get_interview_result(test_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Interview results not found")
        
        return EvaluationResponse(
            evaluation=result.get("evaluation", "Evaluation not available"),
            scores=result.get("scores", {}),
            analytics=result.get("interview_analytics", {}),
            pdf_url=f"{config.API_PREFIX}/download_results/{test_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_results/{test_id}")
async def download_results(test_id: str):
    """Download interview results as PDF"""
    try:
        result = await interview_manager.get_interview_result(test_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Interview results not found")
        
        # Generate PDF
        pdf_buffer = await generate_pdf_report(result, test_id)
        
        return StreamingResponse(
            io.BytesIO(pdf_buffer),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=interview_report_{test_id}.pdf"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/interviews")
async def get_all_interviews():
    """Get all interview results"""
    try:
        results = await interview_manager.db_manager.get_all_interview_results_fast()
        
        return {
            "interviews": results,
            "count": len(results),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Get all interviews error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/interview-students")
async def get_interview_students():
    """Get unique students from interview records"""
    try:
        # This would require aggregation - implementing basic version
        results = await interview_manager.db_manager.get_all_interview_results_fast(100)
        
        # Extract unique students
        students = {}
        for result in results:
            student_id = result.get("student_id")
            if student_id and student_id not in students:
                students[student_id] = {
                    "Student_ID": student_id,
                    "name": result.get("student_name", "Unknown")
                }
        
        return {
            "count": len(students),
            "students": list(students.values()),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Get students error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/interview-students/{student_id}/interviews")
async def get_student_interviews(student_id: str):
    """Get interviews for specific student"""
    try:
        # Get all interviews and filter by student
        all_results = await interview_manager.db_manager.get_all_interview_results_fast(200)
        
        student_interviews = [
            result for result in all_results 
            if str(result.get("student_id", "")) == student_id
        ]
        
        if not student_interviews:
            raise HTTPException(status_code=404, detail="No interviews found for this student")
        
        return {
            "count": len(student_interviews),
            "interviews": student_interviews,
            "student_id": int(student_id),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get student interviews error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cleanup")
async def cleanup_resources():
    """Manual cleanup endpoint"""
    try:
        expired_count = interview_manager.cleanup_expired_sessions()
        
        return {
            "message": f"Cleanup completed - removed {expired_count} expired sessions",
            "expired_sessions": expired_count,
            "active_sessions": len(interview_manager.session_manager.active_sessions) if interview_manager.session_manager else 0,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def api_info():
    """API information and capabilities"""
    return {
        "name": config.APP_TITLE,
        "version": config.APP_VERSION,
        "description": config.APP_DESCRIPTION,
        "features": {
            "websocket_real_time": "Real-time WebSocket communication",
            "ai_powered_questions": "Dynamic question generation based on recent summaries",
            "multi_round_assessment": "Greeting → Technical → Communication → HR rounds",
            "strict_evaluation": "Professional-grade evaluation with detailed scoring",
            "audio_streaming": "Real-time TTS and STT processing",
            "session_management": "Robust session handling with timeout protection",
            "pdf_reports": "Comprehensive PDF evaluation reports"
        },
        "configuration": {
            "interview_duration_minutes": config.INTERVIEW_DURATION_MINUTES,
            "questions_per_round": config.QUESTIONS_PER_ROUND,
            "recent_summaries_days": config.RECENT_SUMMARIES_DAYS,
            "websocket_timeout": config.WEBSOCKET_TIMEOUT,
            "session_timeout": config.SESSION_TIMEOUT
        },
        "endpoints": {
            "start_interview": "POST /start_interview",
            "websocket": "WS /ws/{session_id}",
            "evaluation": "GET /evaluate?test_id={test_id}",
            "download_pdf": "GET /download_results/{test_id}",
            "health": "GET /health"
        }
    }

# =============================================================================
# PDF GENERATION UTILITY
# =============================================================================

async def generate_pdf_report(result: Dict[str, Any], test_id: str) -> bytes:
    """Generate comprehensive PDF report"""
    
    def _generate_pdf_sync():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=LETTER)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = f"Mock Interview Evaluation Report"
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 12))
        
        # Basic Information
        info_text = f"""
        <b>Test ID:</b> {test_id}<br/>
        <b>Student:</b> {result.get('student_name', 'Unknown')}<br/>
        <b>Date:</b> {datetime.fromtimestamp(result.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Duration:</b> {result.get('interview_analytics', {}).get('total_duration_minutes', 0)} minutes<br/>
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Scores
        scores = result.get('scores', {})
        if scores:
            story.append(Paragraph("<b>Performance Scores</b>", styles['Heading2']))
            
            score_mapping = {
                'technical_score': 'Technical Assessment',
                'communication_score': 'Communication Skills',
                'behavioral_score': 'Behavioral/Cultural Fit',
                'overall_score': 'Overall Presentation'
            }
            
            for key, label in score_mapping.items():
                score = scores.get(key, 0)
                story.append(Paragraph(f"<b>{label}:</b> {score}/10", styles['Normal']))
            
            if 'weighted_overall' in scores:
                story.append(Paragraph(f"<b>Weighted Overall Score:</b> {scores['weighted_overall']}/10", styles['Normal']))
            
            story.append(Spacer(1, 15))
        
        # Analytics
        analytics = result.get('interview_analytics', {})
        if analytics:
            story.append(Paragraph("<b>Interview Analytics</b>", styles['Heading2']))
            
            analytics_text = f"""
            <b>Total Questions:</b> {analytics.get('total_questions', 0)}<br/>
            <b>Technical Questions:</b> {analytics.get('technical_questions', 0)}<br/>
            <b>Communication Questions:</b> {analytics.get('communication_questions', 0)}<br/>
            <b>HR Questions:</b> {analytics.get('hr_questions', 0)}<br/>
            <b>Follow-up Questions:</b> {analytics.get('followup_questions', 0)}<br/>
            <b>Average Audio Quality:</b> {analytics.get('avg_audio_quality', 0)}<br/>
            """
            story.append(Paragraph(analytics_text, styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Evaluation
        evaluation = result.get('evaluation', '')
        if evaluation:
            story.append(Paragraph("<b>Detailed Evaluation</b>", styles['Heading2']))
            
            # Split evaluation into paragraphs and wrap long lines
            eval_paragraphs = evaluation.split('\n\n')
            for para in eval_paragraphs:
                if para.strip():
                    wrapped_lines = textwrap.wrap(para.strip(), width=80)
                    para_text = '<br/>'.join(wrapped_lines)
                    story.append(Paragraph(para_text, styles['Normal']))
                    story.append(Spacer(1, 8))
        
        # Conversation Summary
        conversation_log = result.get('conversation_log', [])
        if conversation_log:
            story.append(Paragraph("<b>Interview Conversation Summary</b>", styles['Heading2']))
            
            # Group by stages
            stages = {'technical': [], 'communication': [], 'hr': []}
            for entry in conversation_log:
                stage = entry.get('stage', 'unknown')
                if stage in stages:
                    stages[stage].append(entry)
            
            for stage_name, entries in stages.items():
                if entries:
                    story.append(Paragraph(f"<b>{stage_name.title()} Round ({len(entries)} exchanges):</b>", styles['Heading3']))
                    
                    for i, entry in enumerate(entries[:3], 1):  # Show first 3 exchanges
                        question = entry.get('ai_message', '')[:150] + ('...' if len(entry.get('ai_message', '')) > 150 else '')
                        answer = entry.get('user_response', '')[:150] + ('...' if len(entry.get('user_response', '')) > 150 else '')
                        
                        story.append(Paragraph(f"<b>Q{i}:</b> {question}", styles['Normal']))
                        story.append(Paragraph(f"<b>A{i}:</b> {answer}", styles['Normal']))
                        story.append(Spacer(1, 5))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    
    # Run PDF generation in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        shared_clients.executor,
        _generate_pdf_sync
    )

# =============================================================================
# ERROR HANDLERS
# =============================================================================

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
    port = 8062
    
    print(f"🚀 Starting Enhanced Mock Interview System")
    print(f"📡 Server: https://{local_ip}:{port}")
    print(f"📋 API Docs: https://{local_ip}:{port}/docs")
    print(f"🎙️ WebSocket: wss://{local_ip}:{port}/ws/{{session_id}}")
    print(f"🔄 Real-time Communication: Enabled")
    print(f"🧠 AI-Powered Assessment: Enabled")
    print(f"📊 Strict Evaluation: Enabled")
    print(f"🌐 CORS Origins: {config.CORS_ALLOW_ORIGINS}")
    
    # SSL configuration
    ssl_config = {}
    if config.USE_SSL and os.path.exists(config.SSL_CERT_PATH) and os.path.exists(config.SSL_KEY_PATH):
        ssl_config = {
            "ssl_certfile": config.SSL_CERT_PATH,
            "ssl_keyfile": config.SSL_KEY_PATH
        }
        print(f"🔒 SSL/HTTPS: Enabled")
    else:
        print(f"🔓 SSL/HTTPS: Disabled (certificates not found)")
    
    uvicorn.run(
        "weekly_interview.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level=config.LOG_LEVEL.lower(),
        **ssl_config
    )