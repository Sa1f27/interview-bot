from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os, time, asyncio, numpy as np, sounddevice as sd, scipy.io.wavfile as wavfile
import edge_tts, pygame
from openai import OpenAI
import logging
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

try:
    pygame.mixer.init()
    client = OpenAI()
except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    raise

SAMPLE_RATE = 16000
BLOCK_SIZE = 4096
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 2.0
MAX_RECORDING_DURATION = 10.0
TEST_DURATION_SEC = 120
INACTIVITY_TIMEOUT = 120  # Match test duration

sessions = {}

with open("frontend/data/transcript.txt", "r", encoding="utf-8") as file:
    TRANSCRIPT = file.read()

with open("frontend/data/summary.txt", "r", encoding="utf-8") as file:
    Summary = file.read()

SUMMARY_PROMPT = """You are an assistant that summarizes lecture transcripts for an interviewer.

Given the following transcript, produce a clean and concise summary of the key concepts.
Break down the summary into 4 to 6 clearly defined topics.
For each topic, start a new line in the format:
Topic X: [Summary of that topic]

Transcript:
{TRANSCRIPT}
"""

QUESTION_PROMPT = """
You are conducting a voice-based test. Use the lecture summary below to ask the next most important question that checks the student's understanding of a key concept.

Lecture Summary:
{summary}

Conversation so far:
{history}

Return only the question. Do not explain, paraphrase, or provide any feedback. Ask only one clear and concise question that is strictly relevant to the transcript.

"""

FOLLOWUP_PROMPT = """
You are conducting a voice-based test.

Here is the lecture summary:
{summary}

Conversation so far:
{history}

Previous question:
{previous_question}

User's response:
{user_response}

Was the user's response relevant and appropriate?

- If YES: Return the next must-ask question from the lecture summary.
- If NO: Ask a short follow-up question focused on the original concept.

Return ONLY the question. Do NOT explain or give feedback.
"""

EVALUATION_PROMPT = """
You are evaluating a student's spoken test on the topic below:

Transcript summary:
{summary}

Full Q&A log:
{conversation}

Now generate a short evaluation with:
- Key strengths
- Gaps in understanding (if any)
- Number of questions attempted
- How well they covered the core concepts

Do NOT score the user or give numeric ratings.
"""


def chat(prompt):
    logger.info(f"Chat prompt: {prompt[:50]}...")
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300)
        response = r.choices[0].message.content.strip()
        logger.info("Chat response generated")
        return response
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise

# SUMMARY = chat(SUMMARY_PROMPT.format(TRANSCRIPT=TRANSCRIPT))
SUMMARY = Summary
print(f"Lecture Summary: {SUMMARY}")
    


def record_audio():
    logger.info("Starting audio recording...")
    chunks, silence_start, recording, start_time = [], None, True, time.time()
    
    def callback(indata, frames, time_info, status):
        nonlocal silence_start, recording
        if status:
            logger.error(f"Recording status error: {status}")
            recording = False
            return
        rms = np.sqrt(np.mean(indata**2))
        chunks.append(indata.copy())
        logger.debug(f"RMS: {rms:.6f}")
        
        if rms < SILENCE_THRESHOLD:
            silence_start = silence_start or time.time()
            if silence_start and (time.time() - silence_start) > SILENCE_DURATION:
                logger.info("Silence detected")
                recording = False
        else:
            silence_start = None
        
        if (time.time() - start_time) > MAX_RECORDING_DURATION:
            logger.info("Max duration reached")
            recording = False

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=callback):
            while recording and (time.time() - start_time) <= MAX_RECORDING_DURATION:
                sd.sleep(100)
        logger.info("Recording completed")
    except Exception as e:
        logger.error(f"Recording error: {str(e)}")
        return None
    
    if not chunks:
        logger.warning("No audio chunks")
        return None
    
    audio = np.concatenate(chunks)
    if len(audio) / SAMPLE_RATE < 0.5:
        logger.warning("Audio too short")
        return None
    
    try:
        wavfile.write("temp_in.wav", SAMPLE_RATE, (audio * 32767).astype(np.int16))
        logger.info("Audio saved")
        return "temp_in.wav"
    except Exception as e:
        logger.error(f"Error saving audio: {str(e)}")
        return None

def transcribe(path):
    logger.info(f"Transcribing: {path}")
    try:
        with open(path, "rb") as f:
            txt = client.audio.transcriptions.create(
                file=f, model="whisper-1", response_format="text").strip()
        logger.info(f"Transcription: {txt}")
        return txt
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise
    
    
import subprocess

def clean_audio_folder(folder="audio"):
    for filename in os.listdir(folder):
        if filename.endswith(".mp3"):
            try:
                os.remove(os.path.join(folder, filename))
            except Exception as e:
                logger.warning(f"Could not delete {filename}: {str(e)}")

async def tts(text, voice, speed=1.2):
    logger.info(f"TTS for: {text[:50]}...")
    try:
        clean_audio_folder()  # ✅ Cleanup old files

        timestamp = int(time.time() * 1000)
        original_path = f"audio/ai_raw_{timestamp}.mp3"
        final_path = f"audio/ai_{timestamp}.mp3"

        # 1. Generate TTS
        await edge_tts.Communicate(text, voice).save(original_path)

        # 2. Speed up using ffmpeg
        if os.path.exists(original_path):
            command = [
                "ffmpeg",
                "-y",
                "-i", original_path,
                "-filter:a", f"atempo={speed}",
                "-vn",
                final_path
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(original_path)

            if os.path.exists(final_path):
                logger.info(f"TTS saved: {final_path}")
                return "/" + final_path
            else:
                logger.error("Speed-up output not created")
        else:
            logger.error("TTS file not created")
        return None
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        return None



def get_conversation_history(session_id):
    session = sessions.get(session_id, {})
    conversation_log = session.get("conversation_log", [])
    return "\n".join(f"Q: {x['q']}\nA: {x.get('a','')}" for x in conversation_log)

def validate_session(session_id):
    session = sessions.get(session_id)
    if not session:
        logger.warning(f"Session {session_id} not found")
        raise HTTPException(status_code=400, detail="Session not found")
    if time.time() > session.get("last_activity", 0) + INACTIVITY_TIMEOUT:
        logger.warning(f"Session {session_id} inactive")
        raise HTTPException(status_code=400, detail="Session timed out")
    session["last_activity"] = time.time()
    return session

@app.get("/", response_class=HTMLResponse)
async def index():
    logger.info("Serving index.html")
    return FileResponse("frontend/index.html")

import random
def get_voice():

    voices = [
        # "en-US-JennyNeural",
        # "en-US-ChristopherNeural",
        # "en-US-GuyNeural",
        # "en-US-AriaNeural"
        # "en-IN-NeerjaNeural",
        "en-IN-PrabhatNeural"
    ]
    # Randomly select a voice
    selected_voice = random.choice(voices)
    return selected_voice

@app.get("/start_test")
async def start_test():
    session_id = str(uuid.uuid4())
    logger.info(f"New session: {session_id}")
    
    voice = get_voice()  # ✅ select a random voice per session

    sessions[session_id] = {
        "conversation_log": [],
        "deadline": time.time() + TEST_DURATION_SEC,
        "last_activity": time.time(),
        "start_time": time.time(),
        "voice": voice  # ✅ store voice in session
    }
    logger.info(f"Selected voice for session {session_id}: {voice}")

    try:
        q1 = chat(QUESTION_PROMPT.format(summary=SUMMARY, history=""))

        sessions[session_id]["conversation_log"].append({"q": q1})
        audio_path = await tts(q1, voice)
        return {"session_id": session_id, "question": q1, "audio_path": audio_path}
    except Exception as e:
        logger.error(f"Start test error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/record_and_respond")
async def record_and_respond(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    logger.info(f"Record and respond: {session_id}")
    
    try:
        session = validate_session(session_id)
        logger.info(f"Session deadline: {session['deadline']}, Current time: {time.time()}")
        if time.time() > session["deadline"]:
            logger.info("Test duration exceeded")
            closing = "The 5-minute test is complete. Thank you."
            audio_path = await tts(closing, session["voice"])
            return {"ended": True, "response": closing, "audio_path": audio_path}
    except HTTPException as e:
        closing = "Session expired. Please start a new test."
        audio_path = await tts(closing, session["voice"])

        return {"ended": True, "response": closing, "audio_path": audio_path}
    
    wav = record_audio()
    if not wav:
        logger.warning("No audio")
        return JSONResponse({"error": "No speech detected"}, status_code=400)
    
    try:
        user = transcribe(wav)
        logger.info(f"User response: {user}")
        conversation_log = session["conversation_log"]
        last_q = conversation_log[-1]["q"]
        conversation_log[-1]["a"] = user
        history = get_conversation_history(session_id)
        follow = chat(FOLLOWUP_PROMPT.format(
            summary=SUMMARY,
            history=history,
            previous_question=last_q,
            user_response=user
        ))

        conversation_log.append({"q": follow})
        audio_path = await tts(follow, session["voice"])
        logger.info(f"Follow-up: {follow}")
        return {"ended": False, "response": follow, "audio_path": audio_path}
    except Exception as e:
        logger.error(f"Record error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(wav):
            os.remove(wav)
            logger.info("Cleaned up audio")

@app.get("/summary")
async def summary(session_id: str):
    logger.info(f"Summary for: {session_id}")
    try:
        session = validate_session(session_id)
        conversation_log = session["conversation_log"]
        history = get_conversation_history(session_id)
        evaluation = chat(EVALUATION_PROMPT.format(history))
        
        num_questions = len(conversation_log)
        responses = [x.get("a", "") for x in conversation_log if x.get("a")]
        avg_response_length = sum(len(r.split()) for r in responses) / len(responses) if responses else 0
        
        logger.info("Summary generated")
        return {
            "summary": evaluation,
            "analytics": {
                "num_questions": num_questions,
                "avg_response_length": round(avg_response_length, 1)
            }
        }
    except Exception as e:
        logger.error(f"Summary error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7001, reload=True)
