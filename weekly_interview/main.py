# weekly_interview/main.py
"""
Enhanced Mock Interview System - Daily Standup Style Ultra-Fast Streaming
Real-time WebSocket interview with 7-day fragment processing and streaming TTS
TTS now imported from separate tts_processor.py module
"""

import os
import time
import uuid
import logging
import asyncio
import json
import base64
from typing import Dict, Optional, Any
import io
from datetime import datetime
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Form, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import textwrap

from .core.config import config
from .core.database import DatabaseManager
from .core.ai_services import (
    shared_clients, InterviewSession, InterviewStage,
    EnhancedInterviewFragmentManager, OptimizedAudioProcessor,
    OptimizedConversationManager
)

# Import TTS processor from separate file
from .core.tts_processor import UltraFastTTSProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ULTRA-FAST INTERVIEW SESSION MANAGER (Daily Standup Style)
# =============================================================================

class UltraFastInterviewManager:
    def __init__(self):
        self.active_sessions: Dict[str, InterviewSession] = {}
        self.db_manager = DatabaseManager(shared_clients)
        self.audio_processor = OptimizedAudioProcessor(shared_clients)
        self.tts_processor = UltraFastTTSProcessor()  # Now imported from separate file
        self.conversation_manager = OptimizedConversationManager(shared_clients)
    
    async def create_session_fast(self, websocket: Optional[Any] = None) -> InterviewSession:
        """Ultra-fast session creation with 7-day summary processing"""
        session_id = str(uuid.uuid4())
        test_id = f"interview_{int(time.time())}"
        
        try:
            logger.info(f"?? Creating ultra-fast interview session: {session_id}")
            
            # Get student info and 7-day summaries in parallel
            student_info_task = asyncio.create_task(self.db_manager.get_student_info_fast())
            summaries_task = asyncio.create_task(self.db_manager.get_recent_summaries_fast(
                days=config.RECENT_SUMMARIES_DAYS,
                limit=config.SUMMARIES_LIMIT
            ))
            
            student_id, first_name, last_name, session_key = await student_info_task
            summaries = await summaries_task
            
            # Validate data
            if not summaries or len(summaries) == 0:
                raise Exception("No summaries available for interview content generation")
            
            if not first_name or not last_name:
                raise Exception("Invalid student data retrieved from database")
            
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
                websocket=websocket
            )
            
            # Initialize enhanced fragment manager with 7-day summaries
            fragment_manager = EnhancedInterviewFragmentManager(shared_clients, session_data)
            if not fragment_manager.initialize_fragments(summaries):
                raise Exception("Failed to initialize fragments from 7-day summaries")
            
            session_data.fragment_manager = fragment_manager
            self.active_sessions[session_id] = session_data
            
            logger.info(f"? Ultra-fast interview session created: {session_id} for {session_data.student_name} "
                       f"with {len(session_data.fragment_keys)} fragments from {len(summaries)} summaries")
            
            return session_data
            
        except Exception as e:
            logger.error(f"? Failed to create interview session: {e}")
            raise Exception(f"Session creation failed: {e}")
    
    async def remove_session(self, session_id: str):
        """Fast session removal"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"??? Removed session {session_id}")
    
    async def process_audio_ultra_fast(self, session_id: str, audio_data: bytes):
        """Ultra-fast audio processing pipeline (identical to daily_standup style)"""
        session_data = self.active_sessions.get(session_id)
        if not session_data or not session_data.is_active:
            logger.warning(f"?? Inactive session: {session_id}")
            return
        
        start_time = time.time()
        
        try:
            audio_size = len(audio_data)
            logger.info(f"?? Session {session_id}: Received {audio_size} bytes of audio data")
            
            # Lenient audio size check with better error handling
            if audio_size < 100:
                logger.warning(f"?? Very small audio chunk ({audio_size} bytes)")
                await self._send_quick_message(session_data, {
                    "type": "clarification",
                    "text": "I didn't hear anything clear. Could you please speak a bit louder?",
                    "status": session_data.current_stage.value
                })
                return
            
            # Ultra-fast transcription
            transcript, quality = await self.audio_processor.transcribe_audio_fast(audio_data)
            
            if not transcript or len(transcript.strip()) < 2:
                # Dynamic clarification request
                clarification_message = "The audio wasn't very clear. Could you please repeat that?"
                await self._send_quick_message(session_data, {
                    "type": "clarification",
                    "text": clarification_message,
                    "status": session_data.current_stage.value
                })
                return
            
            logger.info(f"? Session {session_id}: User said: '{transcript}' (quality: {quality:.2f})")
            
            # Update last exchange with user response
            if session_data.exchanges:
                session_data.update_last_response(transcript, quality)
            
            # Generate AI response immediately
            ai_response = await self.conversation_manager.generate_fast_response(session_data, transcript)
            
            # Add exchange to session
            concept = session_data.current_concept if session_data.current_concept else "unknown"
            is_followup = self._determine_if_followup(ai_response)
            
            session_data.add_exchange(ai_response, "", quality, concept, is_followup)
            
            # Update session state (check stage transitions)
            await self._update_session_state_fast(session_data)
            
            # Send response with ultra-fast audio streaming
            await self._send_response_with_ultra_fast_audio(session_data, ai_response)
            
            processing_time = time.time() - start_time
            logger.info(f"? Total processing time: {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"? Audio processing error: {e}")
            
            # Send helpful error message
            if "too small" in str(e).lower():
                error_message = "The audio recording was too short. Please try speaking for a few seconds."
            elif "transcription" in str(e).lower():
                error_message = "I had trouble understanding the audio. Please try speaking clearly into your microphone."
            else:
                error_message = "Sorry, there was a technical issue. Please try again."
            
            await self._send_quick_message(session_data, {
                "type": "error",
                "text": error_message,
                "status": "error"
            })
    
    def _determine_if_followup(self, ai_response: str) -> bool:
        """Determine if response is a follow-up question"""
        followup_indicators = [
            "elaborate", "can you explain", "tell me more", "what about", 
            "how did you", "could you describe", "follow up"
        ]
        return any(indicator in ai_response.lower() for indicator in followup_indicators)
    
    async def _update_session_state_fast(self, session_data: InterviewSession):
        """Ultra-fast session state updates with interview round logic"""
        current_stage = session_data.current_stage
        fragment_manager = session_data.fragment_manager
        
        if current_stage == InterviewStage.GREETING:
            if session_data.questions_per_round["greeting"] >= 2:
                session_data.current_stage = InterviewStage.TECHNICAL
                logger.info(f"?? Session {session_data.session_id} moved to TECHNICAL stage")
        
        elif current_stage in [InterviewStage.TECHNICAL, InterviewStage.COMMUNICATION, InterviewStage.HR]:
            # Check if current round should continue
            if not fragment_manager.should_continue_round(current_stage):
                # Move to next stage
                next_stage = self._get_next_stage(current_stage)
                session_data.current_stage = next_stage
                logger.info(f"?? Session {session_data.session_id} moved to {next_stage.value} stage")
                
                # Check if interview is complete
                if next_stage == InterviewStage.COMPLETE:
                    logger.info(f"?? Session {session_data.session_id} interview completed")
                    # Generate evaluation and save session in background
                    asyncio.create_task(self._finalize_session_fast(session_data))
    
    def _get_next_stage(self, current_stage: InterviewStage) -> InterviewStage:
        """Get next interview stage"""
        stage_progression = {
            InterviewStage.TECHNICAL: InterviewStage.COMMUNICATION,
            InterviewStage.COMMUNICATION: InterviewStage.HR,
            InterviewStage.HR: InterviewStage.COMPLETE
        }
        return stage_progression.get(current_stage, InterviewStage.COMPLETE)
    
    async def _finalize_session_fast(self, session_data: InterviewSession):
        """Fast session finalization with real database save"""
        try:
            # Generate evaluation
            evaluation, scores = await self.conversation_manager.generate_fast_evaluation(session_data)
            
            # Prepare interview data for database
            interview_data = {
                "test_id": session_data.test_id,
                "session_id": session_data.session_id,
                "student_id": session_data.student_id,
                "student_name": session_data.student_name,
                "conversation_log": [
                    {
                        "timestamp": ex.timestamp,
                        "stage": ex.stage.value,
                        "ai_message": ex.ai_message,
                        "user_response": ex.user_response,
                        "transcript_quality": ex.transcript_quality,
                        "concept": ex.concept,
                        "is_followup": ex.is_followup
                    }
                    for ex in session_data.exchanges
                ],
                "evaluation": evaluation,
                "scores": scores,
                "duration_minutes": round((time.time() - session_data.created_at) / 60, 1),
                "questions_per_round": dict(session_data.questions_per_round),
                "followup_questions": session_data.followup_questions,
                "fragments_covered": len([c for c, count in session_data.concept_question_counts.items() if count > 0]),
                "total_fragments": len(session_data.fragment_keys),
                "websocket_used": True,
                "tts_voice": config.TTS_VOICE
            }
            
            # Save to database
            save_success = await self.db_manager.save_interview_result_fast(interview_data)
            
            if not save_success:
                logger.error(f"? Failed to save session {session_data.session_id}")
            
            # Calculate overall score for display
            overall_score = scores.get("weighted_overall", scores.get("overall_score", 8.0))
            
            completion_message = f"Excellent work! Your interview is complete. You scored {overall_score}/10 across all rounds. Thank you!"
            
            await self._send_quick_message(session_data, {
                "type": "interview_complete",
                "text": completion_message,
                "evaluation": evaluation,
                "scores": scores,
                "pdf_url": f"/weekly_interview/download_results/{session_data.test_id}",
                "status": "complete"
            })
            
            # Generate and send final audio using separate TTS processor
            async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(completion_message):
                if audio_chunk:
                    await self._send_quick_message(session_data, {
                        "type": "audio_chunk",
                        "audio": audio_chunk.hex(),
                        "status": "complete"
                    })
            
            await self._send_quick_message(session_data, {"type": "audio_end", "status": "complete"})
            
            session_data.is_active = False
            logger.info(f"? Session {session_data.session_id} finalized and saved")
            
        except Exception as e:
            logger.error(f"? Fast session finalization error: {e}")
            session_data.is_active = False
    
    async def _send_response_with_ultra_fast_audio(self, session_data: InterviewSession, text: str):
        """Send response with ultra-fast audio streaming using separate TTS processor"""
        try:
            await self._send_quick_message(session_data, {
                "type": "ai_response",
                "text": text,
                "stage": session_data.current_stage.value,
                "question_number": session_data.questions_per_round[session_data.current_stage.value]
            })
            
            chunk_count = 0
            # Use separate TTS processor with enhanced error handling
            async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(text):
                if audio_chunk and session_data.is_active:
                    await self._send_quick_message(session_data, {
                        "type": "audio_chunk",
                        "audio": audio_chunk.hex(),
                        "status": session_data.current_stage.value
                    })
                    chunk_count += 1
            
            await self._send_quick_message(session_data, {
                "type": "audio_end",
                "status": session_data.current_stage.value
            })
            
            logger.info(f"?? Streamed {chunk_count} audio chunks")
            
        except Exception as e:
            logger.error(f"? Ultra-fast audio streaming error: {e}")
            # Send text-only response as fallback
            await self._send_quick_message(session_data, {
                "type": "audio_end",
                "status": session_data.current_stage.value,
                "fallback": "text_only"
            })
    
    async def _send_quick_message(self, session_data: InterviewSession, message: dict):
        """Ultra-fast WebSocket message sending (identical to daily_standup)"""
        try:
            if session_data.websocket:
                await session_data.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"? WebSocket send error: {e}")
    
    async def get_session_result_fast(self, test_id: str) -> dict:
        """Fast session result retrieval from real database"""
        try:
            result = await self.db_manager.get_interview_result_fast(test_id)
            if not result:
                raise Exception(f"Interview {test_id} not found in database")
            return result
        except Exception as e:
            logger.error(f"? Error fetching interview result: {e}")
            raise Exception(f"Interview result retrieval failed: {e}")

# =============================================================================
# FASTAPI APPLICATION - DAILY STANDUP STYLE
# =============================================================================

app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS,
    allow_headers=config.CORS_ALLOW_HEADERS,
)

# Mount static files
app.mount("/audio", StaticFiles(directory=str(config.AUDIO_DIR)), name="audio")

# Initialize ultra-fast interview manager
interview_manager = UltraFastInterviewManager()

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup - test real connections"""
    logger.info("?? Ultra-Fast Interview application starting...")
    
    try:
        # Test database connections on startup
        db_manager = DatabaseManager(shared_clients)
        
        # Test MySQL connection
        try:
            conn = db_manager.get_mysql_connection()
            conn.close()
            logger.info("? MySQL connection test successful")
        except Exception as e:
            logger.error(f"? MySQL connection test failed: {e}")
            raise Exception(f"MySQL connection failed: {e}")
        
        # Test MongoDB connection
        try:
            await db_manager.get_mongo_client()
            logger.info("? MongoDB connection test successful")
        except Exception as e:
            logger.error(f"? MongoDB connection test failed: {e}")
            raise Exception(f"MongoDB connection failed: {e}")
        
        # Initialize TTS processor
        try:
            await interview_manager.tts_processor.initialize()
            logger.info("? TTS processor initialized successfully")
        except Exception as e:
            logger.warning(f"?? TTS processor initialization warning: {e}")
            # Continue startup even if TTS has issues - fallback will handle it
        
        logger.info("? All systems verified and ready")
        
    except Exception as e:
        logger.error(f"? Startup failed: {e}")
        raise Exception(f"Application startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await shared_clients.close_connections()
    await interview_manager.db_manager.close_connections()
    logger.info("?? Interview application shutting down")

# =============================================================================
# API ENDPOINTS - REAL DATA ONLY
# =============================================================================

@app.get("/start_interview")
async def start_interview_session_fast():
    """Start a new interview session with 7-day summary processing"""
    try:
        logger.info("?? Starting real interview session with 7-day summaries...")
        
        session_data = await interview_manager.create_session_fast()
        
        greeting = f"Hello {session_data.student_name}! Welcome to your mock interview. I'm excited to learn about your technical skills and experience. How are you feeling today?"
        
        # Add initial greeting to session
        session_data.add_exchange(greeting, "", 0.0, "greeting", False)
        session_data.fragment_manager.add_question(greeting, "greeting", False)
        
        logger.info(f"? Real interview session created: {session_data.test_id}")
        
        return {
            "status": "success",
            "message": "Interview session started successfully",
            "test_id": session_data.test_id,
            "session_id": session_data.session_id,
            "websocket_url": f"/weekly_interview/ws/{session_data.session_id}",
            "greeting": greeting,
            "student_name": session_data.student_name,
            "fragments_count": len(session_data.fragment_keys),
            "summaries_processed": len(session_data.fragment_keys),
            "estimated_duration": config.INTERVIEW_DURATION_MINUTES
        }
        
    except Exception as e:
        logger.error(f"? Error starting interview session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint_ultra_fast(websocket: WebSocket, session_id: str):
    """Ultra-fast WebSocket endpoint with real-time streaming (daily_standup style)"""
    await websocket.accept()
    
    try:
        logger.info(f"?? WebSocket connected for interview session: {session_id}")
        
        session_data = interview_manager.active_sessions.get(session_id)
        if not session_data:
            logger.error(f"? Session {session_id} not found in active sessions")
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": f"Session {session_id} not found. Please start a new interview.",
                "status": "error"
            }))
            return
        
        session_data.websocket = websocket
        
        # Send initial greeting with ultra-fast audio using separate TTS processor
        if session_data.exchanges:
            greeting = session_data.exchanges[0].ai_message
            await websocket.send_text(json.dumps({
                "type": "ai_response",
                "text": greeting,
                "stage": "greeting",
                "status": "greeting"
            }))
            
            # Generate and stream greeting audio with minimal delay
            try:
                async for audio_chunk in interview_manager.tts_processor.generate_ultra_fast_stream(greeting):
                    if audio_chunk:
                        await websocket.send_text(json.dumps({
                            "type": "audio_chunk",
                            "audio": audio_chunk.hex(),
                            "status": "greeting"
                        }))
                
                await websocket.send_text(json.dumps({
                    "type": "audio_end",
                    "status": "greeting"
                }))
            except Exception as tts_error:
                logger.warning(f"?? TTS error for greeting: {tts_error}")
                # Continue without audio - text response already sent
                await websocket.send_text(json.dumps({
                    "type": "audio_end",
                    "status": "greeting",
                    "fallback": "text_only"
                }))
        
        # Main communication loop
        while session_data.is_active and session_data.current_stage != InterviewStage.COMPLETE:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=config.WEBSOCKET_TIMEOUT)
                message = json.loads(data)
                
                if message.get("type") == "audio_data":
                    audio_data = base64.b64decode(message.get("audio", ""))
                    asyncio.create_task(
                        interview_manager.process_audio_ultra_fast(session_id, audio_data)
                    )
                
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
            except asyncio.TimeoutError:
                logger.info(f"? WebSocket timeout: {session_id}")
                break
            except WebSocketDisconnect:
                logger.info(f"?? WebSocket disconnected: {session_id}")
                break
            except Exception as e:
                logger.error(f"? WebSocket error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "text": f"Error: {str(e)}",
                    "status": "error"
                }))
                break
    
    except Exception as e:
        logger.error(f"? WebSocket endpoint error: {e}")
    finally:
        await interview_manager.remove_session(session_id)

@app.get("/evaluate")
async def get_evaluation_fast(test_id: str):
    """Get evaluation with enhanced error handling"""
    try:
        logger.info(f"?? Getting evaluation for test_id: {test_id}")
        
        result = await interview_manager.get_session_result_fast(test_id)
        
        return {
            "test_id": test_id,
            "evaluation": result.get("evaluation", "Evaluation not available"),
            "scores": result.get("scores", {}),
            "analytics": result.get("interview_analytics", {}),
            "pdf_url": f"/weekly_interview/download_results/{test_id}",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"? Error getting evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation: {str(e)}")

@app.get("/download_results/{test_id}")
async def download_results_fast(test_id: str):
    """Fast PDF generation and download from real data"""
    try:
        result = await interview_manager.get_session_result_fast(test_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Interview results not found")
        
        loop = asyncio.get_event_loop()
        pdf_buffer = await loop.run_in_executor(
            shared_clients.executor,
            generate_pdf_report,
            result, test_id
        )
        
        return StreamingResponse(
            io.BytesIO(pdf_buffer),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=interview_report_{test_id}.pdf"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.get("/health")
async def health_check_fast():
    """Ultra-fast health check with real database status and TTS status"""
    try:
        db_status = {"mysql": False, "mongodb": False}
        tts_status = {"status": "unknown"}
        
        # Quick database health check
        try:
            db_manager = DatabaseManager(shared_clients)
            
            # Test MySQL
            conn = db_manager.get_mysql_connection()
            conn.close()
            db_status["mysql"] = True
            
            # Test MongoDB
            await db_manager.get_mongo_client()
            db_status["mongodb"] = True
            
        except Exception as e:
            logger.warning(f"?? Database health check failed: {e}")
        
        # Quick TTS health check
        try:
            tts_status = await interview_manager.tts_processor.health_check()
        except Exception as e:
            logger.warning(f"?? TTS health check failed: {e}")
            tts_status = {"status": "error", "error": str(e)}
        
        overall_status = "healthy" if (all(db_status.values()) and tts_status.get("status") != "error") else "degraded"
        
        return {
            "status": overall_status,
            "service": "ultra_fast_interview_system",
            "timestamp": time.time(),
            "active_sessions": len(interview_manager.active_sessions),
            "version": config.APP_VERSION,
            "database_status": db_status,
            "tts_status": tts_status,
            "features": {
                "7_day_summaries": True,
                "fragment_based_questions": True,
                "real_time_streaming": True,
                "ultra_fast_tts": True,
                "round_based_interview": True,
                "modular_tts": True
            }
        }
    except Exception as e:
        logger.error(f"? Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/api/interview-students")
async def get_interview_students():
    """Get interview students for frontend compatibility"""
    try:
        results = await interview_manager.db_manager.get_all_interview_results_fast(100)
        
        students = {}
        for result in results:
            student_id = result.get("student_id")
            if student_id and student_id not in students:
                students[student_id] = {
                    "Student_ID": student_id,
                    "name": result.get("student_name", "Unknown")
                }
        
        return list(students.values())
        
    except Exception as e:
        logger.error(f"? Get students error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/interview-students/{student_id}/interviews")
async def get_student_interviews(student_id: str):
    """Get student interviews for frontend compatibility"""
    try:
        all_results = await interview_manager.db_manager.get_all_interview_results_fast(200)
        
        student_interviews = [
            {
                "interview_id": result.get("test_id"),
                "test_id": result.get("test_id"),
                "session_id": result.get("session_id"),
                "timestamp": result.get("timestamp"),
                "scores": result.get("scores", {}),
                "Student_ID": result.get("student_id"),
                "name": result.get("student_name")
            }
            for result in all_results 
            if str(result.get("student_id", "")) == student_id
        ]
        
        return student_interviews
        
    except Exception as e:
        logger.error(f"? Get student interviews error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# PDF GENERATION UTILITY
# =============================================================================

def generate_pdf_report(result: Dict[str, Any], test_id: str) -> bytes:
    """Generate PDF report from real interview data"""
    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=LETTER)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = f"Mock Interview Report - {result.get('student_name', 'Student')}"
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 12))
        
        # Session info
        info_text = f"""
        Test ID: {test_id}
        Student: {result.get('student_name', 'Unknown')}
        Date: {datetime.fromtimestamp(result.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')}
        Duration: {result.get('duration_minutes', 0)} minutes
        Rounds Completed: {len(result.get('questions_per_round', {}))}
        TTS System: {result.get('system_info', {}).get('tts_voice', 'EdgeTTS')}
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Scores section
        scores = result.get('scores', {})
        if scores:
            story.append(Paragraph("Performance Scores", styles['Heading2']))
            score_text = f"""
            Technical Assessment: {scores.get('technical_score', 0)}/10
            Communication Skills: {scores.get('communication_score', 0)}/10
            Behavioral/Cultural Fit: {scores.get('behavioral_score', 0)}/10
            Overall Presentation: {scores.get('overall_score', 0)}/10
            Weighted Overall: {scores.get('weighted_overall', 0)}/10
            """
            story.append(Paragraph(score_text, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Evaluation
        if result.get('evaluation'):
            story.append(Paragraph("Detailed Evaluation", styles['Heading2']))
            # Split evaluation into paragraphs for better formatting
            eval_paragraphs = result['evaluation'].split('\n\n')
            for para in eval_paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), styles['Normal']))
                    story.append(Spacer(1, 6))
        
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer.read()
        
    except Exception as e:
        logger.error(f"? PDF generation error: {e}")
        raise Exception(f"PDF generation failed: {e}")

# Additional WebSocket endpoint for compatibility with frontend routing
@app.websocket("/weekly_interview/ws/{session_id}")
async def websocket_endpoint_weekly_interview(websocket: WebSocket, session_id: str):
    """Reuse the same logic as the /ws/{session_id} endpoint for routing compatibility"""
    await websocket_endpoint_ultra_fast(websocket, session_id)