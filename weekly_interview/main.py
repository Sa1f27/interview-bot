# weekly_interview/main.py
"""
Enhanced Mock Interview System - Main Application with Enhanced Error Handling
Fast WebSocket-based real-time interview system
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
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import textwrap

from .core.config import config
from .core.database import DatabaseManager, get_db_manager
from .core.ai_services import (
    shared_clients, InterviewSessionManager, OptimizedAudioProcessor,
    UltraFastTTSProcessor, OptimizedConversationManager, InterviewStage, InterviewState
)

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

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

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: float
    components: Dict[str, Any]
    error_details: Optional[str] = None

# =============================================================================
# ENHANCED INTERVIEW MANAGER WITH PROPER ERROR HANDLING
# =============================================================================

class EnhancedInterviewManager:
    def __init__(self):
        self.session_manager = None
        self.audio_processor = None
        self.tts_processor = None
        self.conversation_manager = None
        self.db_manager = None
        self.is_initialized = False
        self.initialization_error = None
        
    async def initialize(self):
        """Initialize all components with comprehensive error handling"""
        try:
            logger.info("?? Starting Enhanced Interview Manager initialization...")
            
            # Step 1: Initialize shared clients first
            try:
                logger.info("?? Initializing shared AI clients...")
                await shared_clients.initialize()
                logger.info("? Shared AI clients initialized")
            except Exception as e:
                error_msg = f"Shared clients initialization failed: {str(e)}"
                logger.error(f"? {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Step 2: Initialize database manager
            try:
                logger.info("?? Initializing database manager...")
                self.db_manager = get_db_manager(shared_clients)
                if not self.db_manager:
                    raise RuntimeError("Failed to create database manager instance")
                
                await self.db_manager.initialize()
                logger.info("? Database manager initialized")
                
                # Test database connections
                health_status = await self.db_manager.health_check()
                if not health_status.get("overall", False):
                    raise RuntimeError(f"Database health check failed: {health_status}")
                logger.info("? Database health check passed")
                
            except Exception as e:
                error_msg = f"Database initialization failed: {str(e)}"
                logger.error(f"? {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Step 3: Initialize session manager
            try:
                logger.info("?? Initializing session manager...")
                self.session_manager = InterviewSessionManager(self.db_manager)
                await self.session_manager.initialize()
                logger.info("? Session manager initialized")
            except Exception as e:
                error_msg = f"Session manager initialization failed: {str(e)}"
                logger.error(f"? {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Step 4: Initialize audio processor
            try:
                logger.info("?? Initializing audio processor...")
                self.audio_processor = OptimizedAudioProcessor(shared_clients)
                await self.audio_processor.initialize()
                logger.info("? Audio processor initialized")
            except Exception as e:
                error_msg = f"Audio processor initialization failed: {str(e)}"
                logger.error(f"? {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Step 5: Initialize TTS processor
            try:
                logger.info("?? Initializing TTS processor...")
                self.tts_processor = UltraFastTTSProcessor()
                await self.tts_processor.initialize()
                logger.info("? TTS processor initialized")
            except Exception as e:
                error_msg = f"TTS processor initialization failed: {str(e)}"
                logger.error(f"? {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Step 6: Initialize conversation manager
            try:
                logger.info("?? Initializing conversation manager...")
                self.conversation_manager = OptimizedConversationManager(shared_clients)
                await self.conversation_manager.initialize()
                logger.info("? Conversation manager initialized")
            except Exception as e:
                error_msg = f"Conversation manager initialization failed: {str(e)}"
                logger.error(f"? {error_msg}")
                raise RuntimeError(error_msg) from e
            
            self.is_initialized = True
            self.initialization_error = None
            logger.info("? Enhanced Interview Manager initialized successfully")
            
        except Exception as e:
            self.is_initialized = False
            self.initialization_error = str(e)
            logger.error(f"? Interview Manager initialization failed: {e}")
            logger.error(f"? Traceback: {traceback.format_exc()}")
            # Re-raise with more context
            raise RuntimeError(f"Interview Manager initialization failed: {str(e)}") from e
    
    def check_initialization(self) -> Dict[str, Any]:
        """Check initialization status and return detailed info"""
        return {
            "is_initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "components": {
                "db_manager": self.db_manager is not None,
                "session_manager": self.session_manager is not None,
                "audio_processor": self.audio_processor is not None,
                "tts_processor": self.tts_processor is not None,
                "conversation_manager": self.conversation_manager is not None
            }
        }
    
    async def start_interview_session(self, websocket: Optional[WebSocket] = None) -> Dict[str, Any]:
        """Start interview session with enhanced error handling"""
        try:
            # Check initialization first
            if not self.is_initialized:
                raise RuntimeError(f"Interview Manager not initialized: {self.initialization_error}")
            
            # Validate components
            if not all([self.session_manager, self.db_manager, self.conversation_manager]):
                raise RuntimeError("Interview manager components not properly initialized")
            
            logger.info("?? Starting new interview session...")
            
            # Create session
            session = await self.session_manager.create_session_fast(websocket)
            logger.info(f"? Session created: {session.session_id}")
            
            # Generate greeting
            greeting = await self.conversation_manager.generate_interview_response(session)
            logger.info(f"? Greeting generated: {greeting[:100]}...")
            
            # Update session
            session.add_exchange(greeting, "", 0.0, False, 0.0)
            session.current_state = InterviewState.IN_PROGRESS
            session.round_start_times["greeting"] = time.time()
            
            result = {
                "test_id": session.test_id,
                "session_id": session.session_id,
                "websocket_url": f"/weekly_interview/ws/{session.session_id}",
                "student_name": session.student_name,
                "estimated_duration_minutes": config.INTERVIEW_DURATION_MINUTES,
                "greeting": greeting,
                "message": "Interview session created successfully"
            }
            
            logger.info(f"? Interview session started successfully: {session.test_id}")
            return result
            
        except Exception as e:
            logger.error(f"? Start interview failed: {e}")
            logger.error(f"? Traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to start interview: {e}")
    
    async def process_audio_message(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
        """Process audio message with enhanced error handling"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Interview Manager not initialized")
            
            session = self.session_manager.validate_session(session_id)
            start_time = time.time()
            
            transcript, quality = await self.audio_processor.transcribe_audio_fast(audio_data)
            
            if not transcript or len(transcript.strip()) < 2:
                return {
                    "type": "clarification",
                    "message": "I didn't hear your response clearly. Could you please repeat that?",
                    "status": session.current_stage.value
                }
            
            logger.info(f"??? User response: '{transcript}' (quality: {quality:.2f})")
            
            if session.exchanges:
                session.update_last_response(transcript, quality)
            
            await self._check_stage_transitions(session)
            
            ai_response = await self.conversation_manager.generate_interview_response(session, transcript)
            
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
            logger.error(f"? Audio processing failed: {e}")
            logger.error(f"? Traceback: {traceback.format_exc()}")
            return {
                "type": "error",
                "message": f"Processing failed: {str(e)}",
                "status": "error"
            }
    
    async def _check_stage_transitions(self, session):
        """Check and handle stage transitions"""
        current_stage = session.current_stage
        questions_in_stage = session.questions_per_round[current_stage.value]
        
        if current_stage == InterviewStage.GREETING and questions_in_stage >= 2:
            session.current_stage = InterviewStage.TECHNICAL
            session.round_start_times["technical"] = time.time()
            logger.info(f"?? Transitioned to TECHNICAL stage")
            
        elif current_stage == InterviewStage.TECHNICAL and questions_in_stage >= config.MAX_QUESTIONS_PER_ROUND:
            session.current_stage = InterviewStage.COMMUNICATION
            session.round_start_times["communication"] = time.time()
            logger.info(f"?? Transitioned to COMMUNICATION stage")
            
        elif current_stage == InterviewStage.COMMUNICATION and questions_in_stage >= config.MAX_QUESTIONS_PER_ROUND:
            session.current_stage = InterviewStage.HR
            session.round_start_times["hr"] = time.time()
            logger.info(f"?? Transitioned to HR stage")
            
        elif current_stage == InterviewStage.HR and questions_in_stage >= config.MAX_QUESTIONS_PER_ROUND:
            session.current_stage = InterviewStage.COMPLETE
            session.current_state = InterviewState.COMPLETED
            logger.info(f"?? Interview completed")
    
    def _determine_if_followup(self, session, ai_response: str) -> bool:
        """Determine if response is a follow-up question"""
        followup_indicators = [
            "follow up", "elaborate", "can you explain", "tell me more",
            "what about", "how did you", "could you describe"
        ]
        return any(indicator in ai_response.lower() for indicator in followup_indicators)
    
    async def generate_tts_stream(self, session_id: str, text: str):
        """Generate TTS stream with error handling"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Interview Manager not initialized")
            
            session = self.session_manager.validate_session(session_id)
            
            async for audio_chunk in self.tts_processor.generate_audio_stream(text):
                if audio_chunk and session.is_active:
                    yield audio_chunk
        except Exception as e:
            logger.error(f"? TTS streaming failed: {e}")
    
    async def complete_interview(self, session_id: str) -> Dict[str, Any]:
        """Complete interview with enhanced error handling"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Interview Manager not initialized")
            
            session = self.session_manager.validate_session(session_id)
            
            evaluation, scores = await self.conversation_manager.generate_comprehensive_evaluation(session)
            
            analytics = {
                "total_duration_minutes": round((time.time() - session.created_at) / 60, 1),
                "questions_per_round": dict(session.questions_per_round),
                "total_questions": session.total_questions_asked,
                "followup_questions": session.followup_questions,
                "audio_quality_avg": round(sum(session.audio_quality_scores) / len(session.audio_quality_scores), 2) if session.audio_quality_scores else 0,
                "stage_completion_times": self._calculate_stage_times(session),
                "response_quality_metrics": self._calculate_response_metrics(session)
            }
            
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
            self.session_manager.cleanup_session(session_id)
            
            logger.info(f"? Interview completed and saved: {session.test_id}")
            
            return {
                "evaluation": evaluation,
                "scores": scores,
                "analytics": analytics,
                "pdf_url": f"/weekly_interview/download_results/{session.test_id}"
            }
        except Exception as e:
            logger.error(f"? Interview completion failed: {e}")
            logger.error(f"? Traceback: {traceback.format_exc()}")
            raise Exception(f"Interview completion failed: {e}")
    
    def _calculate_stage_times(self, session) -> Dict[str, float]:
        """Calculate stage completion times"""
        stage_times = {}
        
        for stage_name, start_time in session.round_start_times.items():
            if start_time:
                stage_names = ["greeting", "technical", "communication", "hr"]
                try:
                    current_idx = stage_names.index(stage_name)
                    if current_idx < len(stage_names) - 1:
                        next_stage = stage_names[current_idx + 1]
                        end_time = session.round_start_times.get(next_stage, time.time())
                    else:
                        end_time = time.time()
                    
                    stage_times[stage_name] = round((end_time - start_time) / 60, 1)
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
        """Get interview result with error handling"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Interview Manager not initialized")
            
            return await self.db_manager.get_interview_result_fast(test_id)
        except Exception as e:
            logger.error(f"? Get interview result failed: {e}")
            return None
    
    def cleanup_expired_sessions(self):
        """Cleanup expired sessions"""
        if self.session_manager:
            return self.session_manager.cleanup_expired_sessions()
        return 0

# =============================================================================
# FASTAPI APPLICATION SETUP WITH ENHANCED ERROR HANDLING
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan management with detailed error handling"""
    logger.info("?? Enhanced Mock Interview System starting...")
    
    cleanup_task = None
    
    try:
        # Initialize interview manager with detailed error reporting
        logger.info("?? Initializing Interview Manager...")
        await interview_manager.initialize()
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
        logger.info("? Enhanced Mock Interview System operational")
        
    except Exception as e:
        logger.error(f"? Startup failed: {e}")
        logger.error(f"? Detailed traceback: {traceback.format_exc()}")
        
        # Store initialization error for health check
        interview_manager.initialization_error = str(e)
        
        # Don't raise - allow the app to start in degraded mode
        logger.warning("?? Starting in degraded mode - interview functionality disabled")
    
    yield
    
    logger.info("?? Shutting down Enhanced Mock Interview System...")
    try:
        if cleanup_task:
            cleanup_task.cancel()
        await shared_clients.close_connections()
        if interview_manager.db_manager:
            await interview_manager.db_manager.close_connections()
        logger.info("? Graceful shutdown completed")
    except Exception as e:
        logger.error(f"? Shutdown error: {e}")

async def periodic_cleanup():
    """Periodic cleanup with error handling"""
    while True:
        try:
            await asyncio.sleep(config.CLEANUP_INTERVAL)
            if interview_manager.is_initialized:
                cleaned = interview_manager.cleanup_expired_sessions()
                if cleaned > 0:
                    logger.info(f"?? Cleaned up {cleaned} expired sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"? Cleanup task error: {e}")

app = FastAPI(
    title=config.APP_TITLE,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS,
    allow_headers=config.CORS_ALLOW_HEADERS,
)

interview_manager = EnhancedInterviewManager()

# =============================================================================
# ENHANCED API ENDPOINTS WITH COMPREHENSIVE ERROR HANDLING
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with detailed component status"""
    try:
        logger.info("?? Health check requested")
        
        # Basic service info
        health_status = {
            "status": "healthy",
            "service": config.APP_TITLE,
            "version": config.APP_VERSION,
            "timestamp": time.time(),
            "components": {}
        }
        
        # Check interview manager initialization
        init_status = interview_manager.check_initialization()
        health_status["components"]["interview_manager"] = init_status
        
        # Check database if initialized
        if interview_manager.db_manager:
            try:
                db_health = await interview_manager.db_manager.health_check()
                health_status["components"]["database"] = db_health
            except Exception as e:
                health_status["components"]["database"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            health_status["components"]["database"] = {
                "status": "not_initialized",
                "error": "Database manager not initialized"
            }
        
        # Check session manager
        if interview_manager.session_manager:
            health_status["components"]["sessions"] = {
                "status": "healthy",
                "active_sessions": len(interview_manager.session_manager.active_sessions)
            }
        else:
            health_status["components"]["sessions"] = {
                "status": "not_initialized"
            }
        
        # Determine overall health
        overall_healthy = (
            init_status.get("is_initialized", False) and
            health_status["components"].get("database", {}).get("overall", False)
        )
        
        if not overall_healthy:
            health_status["status"] = "degraded"
            health_status["error_details"] = init_status.get("initialization_error")
        
        # Return appropriate status code
        status_code = 200 if overall_healthy else 503
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
        
    except Exception as e:
        logger.error(f"? Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": config.APP_TITLE,
                "error": str(e),
                "timestamp": time.time(),
                "components": {}
            }
        )

@app.get("/start_interview", response_model=StartInterviewResponse)
async def start_interview():
    """Start interview with enhanced error handling"""
    try:
        logger.info("?? Start interview endpoint called")
        
        # Check if system is initialized
        if not interview_manager.is_initialized:
            error_msg = f"System not ready: {interview_manager.initialization_error}"
            logger.error(f"? {error_msg}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service unavailable",
                    "message": "Interview system is not ready. Please try again later.",
                    "details": error_msg,
                    "retry_after": 30
                }
            )
        
        # Start interview session
        result = await interview_manager.start_interview_session()
        
        logger.info(f"? Interview started successfully: {result['test_id']}")
        
        return StartInterviewResponse(
            test_id=result["test_id"],
            session_id=result["session_id"],
            websocket_url=result["websocket_url"],
            student_name=result["student_name"],
            estimated_duration_minutes=result["estimated_duration_minutes"],
            message=result["message"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? Start interview error: {e}")
        logger.error(f"? Traceback: {traceback.format_exc()}")
        
        # Provide detailed error information
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to start interview session",
                "details": str(e),
                "timestamp": time.time()
            }
        )

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Enhanced WebSocket endpoint with comprehensive error handling"""
    await websocket.accept()
    
    try:
        logger.info(f"?? WebSocket connected for session: {session_id}")
        
        # Check if system is ready
        if not interview_manager.is_initialized:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Interview system is not ready. Please try again later.",
                "status": "system_not_ready"
            }))
            return
        
        session = interview_manager.session_manager.get_session(session_id)
        if not session:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Interview session not found. Please start a new interview.",
                "status": "session_not_found"
            }))
            return
        
        session.websocket = websocket
        
        # Send initial greeting with audio
        if session.exchanges:
            initial_greeting = session.exchanges[0].ai_message
            
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
                logger.error(f"? Initial audio generation failed: {e}")
        
        # Main WebSocket communication loop
        while session.is_active and session.current_stage != InterviewStage.COMPLETE:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=config.WEBSOCKET_TIMEOUT
                )
                
                data = json.loads(message)
                
                if data.get("type") == "audio_data":
                    audio_b64 = data.get("audio", "")
                    if audio_b64:
                        try:
                            audio_data = base64.b64decode(audio_b64)
                            
                            response = await interview_manager.process_audio_message(session_id, audio_data)
                            
                            await websocket.send_text(json.dumps(response))
                            
                            if response.get("type") == "ai_response":
                                message_text = response.get("message", "")
                                
                                try:
                                    audio_chunks = []
                                    async for audio_chunk in interview_manager.generate_tts_stream(session_id, message_text):
                                        if audio_chunk:
                                            audio_chunks.append(audio_chunk)
                                    
                                    if audio_chunks:
                                        combined_audio = b''.join(audio_chunks)
                                        audio_b64 = base64.b64encode(combined_audio).decode('utf-8')
                                        
                                        await websocket.send_text(json.dumps({
                                            "type": "audio_data",
                                            "audio": audio_b64,
                                            "status": response.get("stage", "unknown")
                                        }))
                                    
                                    await websocket.send_text(json.dumps({
                                        "type": "audio_end",
                                        "status": response.get("stage", "unknown")
                                    }))
                                    
                                except Exception as e:
                                    logger.error(f"? TTS generation failed: {e}")
                                    await websocket.send_text(json.dumps({
                                        "type": "error",
                                        "message": "Audio generation failed",
                                        "status": "error"
                                    }))
                            
                        except Exception as e:
                            logger.error(f"? Audio processing failed: {e}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": "Audio processing failed",
                                "status": "error"
                            }))
                
                elif data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
                elif data.get("type") == "complete_interview":
                    try:
                        result = await interview_manager.complete_interview(session_id)
                        await websocket.send_text(json.dumps({
                            "type": "interview_complete",
                            "result": result
                        }))
                        break
                    except Exception as e:
                        logger.error(f"? Interview completion failed: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Interview completion failed",
                            "status": "error"
                        }))
                
            except asyncio.TimeoutError:
                logger.info(f"?? WebSocket timeout for session: {session_id}")
                await websocket.send_text(json.dumps({
                    "type": "timeout",
                    "message": "Session timeout due to inactivity",
                    "status": "timeout"
                }))
                break
                
            except WebSocketDisconnect:
                logger.info(f"?? WebSocket disconnected: {session_id}")
                break
                
            except Exception as e:
                logger.error(f"? WebSocket error: {e}")
                logger.error(f"? Traceback: {traceback.format_exc()}")
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
                logger.error(f"? Final completion failed: {e}")
    
    except Exception as e:
        logger.error(f"? WebSocket endpoint error: {e}")
        logger.error(f"? Traceback: {traceback.format_exc()}")
    
    finally:
        if session_id in interview_manager.session_manager.active_sessions:
            interview_manager.session_manager.cleanup_session(session_id)

@app.get("/evaluate")
async def get_evaluation(test_id: str):
    """Get evaluation with enhanced error handling"""
    try:
        if not interview_manager.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Interview system not ready"
            )
        
        result = await interview_manager.get_interview_result(test_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Interview results not found")
        
        return EvaluationResponse(
            evaluation=result.get("evaluation", "Evaluation not available"),
            scores=result.get("scores", {}),
            analytics=result.get("interview_analytics", {}),
            pdf_url=f"/weekly_interview/download_results/{test_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? Get evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_results/{test_id}")
async def download_results(test_id: str):
    """Download results with enhanced error handling"""
    try:
        if not interview_manager.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Interview system not ready"
            )
        
        result = await interview_manager.get_interview_result(test_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Interview results not found")
        
        pdf_buffer = await generate_pdf_report(result, test_id)
        
        return StreamingResponse(
            io.BytesIO(pdf_buffer),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=interview_report_{test_id}.pdf"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/interview-students")
async def get_interview_students():
    """Get interview students with enhanced error handling"""
    try:
        if not interview_manager.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Interview system not ready"
            )
        
        results = await interview_manager.db_manager.get_all_interview_results_fast(100)
        
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
            "data": list(students.values()),
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? Get students error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/interview-students/{student_id}/interviews")
async def get_student_interviews(student_id: str):
    """Get student interviews with enhanced error handling"""
    try:
        if not interview_manager.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Interview system not ready"
            )
        
        all_results = await interview_manager.db_manager.get_all_interview_results_fast(200)
        
        student_interviews = [
            result for result in all_results 
            if str(result.get("student_id", "")) == student_id
        ]
        
        if not student_interviews:
            raise HTTPException(status_code=404, detail="No interviews found for this student")
        
        return {
            "count": len(student_interviews),
            "data": student_interviews,
            "student_id": int(student_id),
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? Get student interviews error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# PDF GENERATION UTILITY
# =============================================================================

async def generate_pdf_report(result: Dict[str, Any], test_id: str) -> bytes:
    """Generate PDF report with error handling"""
    def _generate_pdf_sync():
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=LETTER)
            styles = getSampleStyleSheet()
            story = []
            
            title = f"Mock Interview Evaluation Report"
            story.append(Paragraph(title, styles['Title']))
            story.append(Spacer(1, 12))
            
            info_text = f"""
            <b>Test ID:</b> {test_id}<br/>
            <b>Student:</b> {result.get('student_name', 'Unknown')}<br/>
            <b>Date:</b> {datetime.fromtimestamp(result.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>Duration:</b> {result.get('interview_analytics', {}).get('total_duration_minutes', 0)} minutes<br/>
            """
            story.append(Paragraph(info_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
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
            
            evaluation = result.get('evaluation', '')
            if evaluation:
                story.append(Paragraph("<b>Detailed Evaluation</b>", styles['Heading2']))
                
                eval_paragraphs = evaluation.split('\n\n')
                for para in eval_paragraphs:
                    if para.strip():
                        wrapped_lines = textwrap.wrap(para.strip(), width=80)
                        para_text = '<br/>'.join(wrapped_lines)
                        story.append(Paragraph(para_text, styles['Normal']))
                        story.append(Spacer(1, 8))
            
            doc.build(story)
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            logger.error(f"? PDF generation failed: {e}")
            raise Exception(f"PDF generation failed: {e}")
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(shared_clients.executor, _generate_pdf_sync)

# =============================================================================
# ENHANCED ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler"""
    logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail if isinstance(exc.detail, str) else exc.detail.get("error", "Unknown error"),
            "message": exc.detail.get("message") if isinstance(exc.detail, dict) else str(exc.detail),
            "details": exc.detail.get("details") if isinstance(exc.detail, dict) else None,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Enhanced general exception handler"""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred during interview processing",
            "details": str(exc) if config.LOG_LEVEL == "DEBUG" else None,
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

# =============================================================================
# STARTUP VALIDATION
# =============================================================================

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

    # Use your actual server configuration
    server_ip = "192.168.48.201"  # Your Linux server IP
    port = 8070  # Your existing port
    
    print(f"?? Starting Enhanced Mock Interview System")
    print(f"?? Server: https://{server_ip}:{port}")
    print(f"?? API Docs: https://{server_ip}:{port}/docs")
    print(f"?? WebSocket: wss://{server_ip}:{port}/weekly_interview/ws/{{session_id}}")
    print(f"?? Real-time Communication: Enabled")
    print(f"?? AI-Powered Assessment: Enabled")
    print(f"?? Strict Evaluation: Enabled")
    print(f"?? CORS Origins: {config.CORS_ALLOW_ORIGINS}")
    print(f"?? Remote Access: Windows laptop ? Linux server")
    
    # SSL configuration for production
    ssl_config = {}
    if config.USE_SSL and os.path.exists(config.SSL_CERT_PATH) and os.path.exists(config.SSL_KEY_PATH):
        ssl_config = {
            "ssl_certfile": config.SSL_CERT_PATH,
            "ssl_keyfile": config.SSL_KEY_PATH
        }
        print(f"?? SSL/HTTPS: Enabled with certificates")
    else:
        print(f"?? SSL/HTTPS: Disabled (certificates not found)")
        print(f"?? For production, consider enabling SSL certificates")
    
    uvicorn.run(
        "weekly_interview.main:app",
        host="0.0.0.0",  # Listen on all interfaces for remote access
        port=port,
        reload=True,
        log_level=config.LOG_LEVEL.lower(),
        ws_ping_interval=20,
        ws_ping_timeout=20,
        timeout_keep_alive=30,
        **ssl_config
    )