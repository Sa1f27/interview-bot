from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Import the refactored modules
from database import fetch_latest_applicant
from interview import test_manager, llm_manager, audio_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

# =======================
# Application setup
# =======================

app = FastAPI(
    title="AI Interview API",
    description="API for AI-driven interviews",
    version="2.0.0",
)

# Get base directory and setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static File Serving
# Mount the 'audio' directory to serve generated audio files.
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
# Mount the 'static' directory to serve JS, CSS, etc.
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=FileResponse)
async def read_index():
    """Serves the main index.html file."""
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

@app.post("/interview/{test_id}")
async def interview_step(test_id: str, request: Request):
    """Handles a single step of the interview."""
    try:
        test = test_manager.validate_test(test_id)
        
        # The client sends raw audio blob.
        audio_data = await request.body()

        # Here you would typically save the audio and transcribe it.
        # For this refactor, we'll simulate transcription.
        user_response = "This is a simulated user response."
        logger.info(f"Test {test_id}: Simulated user response: {user_response}")
        test_manager.add_entry(test_id, "user", user_response)

        # Generate the next question
        next_question = await llm_manager.generate_next_question(test)
        test_manager.add_entry(test_id, "assistant", next_question)
        
        # Generate audio for the next question
        audio_path = await audio_manager.text_to_speech(next_question, test.voice)
        
        # Send the question and audio path back to the client
        response = {
            "text": next_question,
            "audio_path": audio_path,
            "ended": "interview is now complete" in next_question.lower()
        }
        
        return response

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Interview step error for test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start_test")
async def start_test():
    """Starts a new interview session."""
    try:
        applicant = fetch_latest_applicant()
        
        if not applicant or not applicant.get('ai_data'):
            raise HTTPException(status_code=404, detail="No suitable applicant found with AI data.")

        voice = audio_manager.get_random_voice()
        test_id = test_manager.create_test(applicant['ai_data'], voice)
        
        initial_question = await llm_manager.generate_initial_greeting()
        test_manager.add_entry(test_id, "assistant", initial_question)
        
        audio_path = await audio_manager.text_to_speech(initial_question, voice)
        
        return {
            "test_id": test_id,
            "question": initial_question,
            "audio_path": audio_path,
            "applicant_name": applicant.get('name', 'Candidate')
        }
    except Exception as e:
        logger.error(f"Error starting test: {e}")
        raise HTTPException(status_code=500, detail="Failed to start test")

@app.get("/summary/{test_id}")
async def get_summary(test_id: str):
    """Get a summary of the interview."""
    try:
        test = test_manager.validate_test(test_id)
        evaluation = await llm_manager.generate_evaluation(test)
        return {
            "summary": evaluation,
            "conversation_log": test.conversation_log
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating summary for test {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary")


if __name__ == "__main__":
    import uvicorn
    
    print("Starting AI Interview API Server")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
