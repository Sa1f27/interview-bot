from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

from database import fetch_latest_applicant
from interview import test_manager, llm_manager, audio_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Interview API", version="3.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=FileResponse)
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

@app.post("/start_test")
async def start_test():
    try:
        logger.info("Starting new interview...")
        applicant = fetch_latest_applicant()
        
        if not applicant or not applicant.get('ai_data'):
            raise HTTPException(status_code=404, detail="No applicant found with AI data")

        voice = audio_manager.get_random_voice()
        test_id = test_manager.create_test(applicant['ai_data'], voice)
        
        initial_question = llm_manager.generate_initial_greeting(applicant['ai_data'])
        test_manager.add_entry(test_id, "assistant", initial_question)
        
        audio_path = await audio_manager.text_to_speech(initial_question, voice)
        
        return {
            "test_id": test_id,
            "question": initial_question,
            "audio_path": audio_path,
            "applicant_name": applicant.get('name', 'Candidate')
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting test: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interview/{test_id}")
async def interview_step(test_id: str, request: Request):
    try:
        logger.info(f"Processing interview step for {test_id}")
        test = test_manager.validate_test(test_id)
        
        if test.is_complete:
            return {
                "text": "Interview already completed. Thank you!",
                "audio_path": None,
                "ended": True
            }
        
        audio_data = await request.body()
        
        if not audio_data or len(audio_data) < 100:
            return JSONResponse(
                status_code=400,
                content={"error": "No audio received. Please try again."}
            )
        
        logger.info(f"Received {len(audio_data)} bytes")
        
        # Transcribe with Whisper
        user_response = audio_manager.transcribe_audio(audio_data)
        
        if not user_response:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not transcribe. Please speak clearly."}
            )
        
        logger.info(f"User: {user_response}")
        test_manager.add_entry(test_id, "user", user_response)
        
        # Generate next question with GPT
        next_question = llm_manager.generate_next_question(test)
        test_manager.add_entry(test_id, "assistant", next_question)
        
        interview_ended = (
            "conclude" in next_question.lower() or 
            "thank you for your time" in next_question.lower() or
            test.question_index >= 6
        )
        
        if interview_ended:
            test_manager.mark_complete(test_id)
        
        # Convert to speech
        audio_path = await audio_manager.text_to_speech(next_question, test.voice)
        
        return {
            "text": next_question,
            "audio_path": audio_path,
            "ended": interview_ended,
            "question_number": test.question_index,
            "user_transcript": user_response
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in interview step: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/{test_id}")
async def get_summary(test_id: str):
    try:
        test = test_manager.validate_test(test_id)
        evaluation = llm_manager.generate_evaluation(test)
        
        return {
            "summary": evaluation,
            "conversation_log": test.conversation_log,
            "question_count": test.question_index
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_tests": len(test_manager.tests)}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Interview Server")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)