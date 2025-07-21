"""
Configuration module for Daily Standup application
Handles all non-sensitive configuration values
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent / '.env')

class Config:
    """Central configuration class"""
    
    # =============================================================================
    # PATHS AND DIRECTORIES
    # =============================================================================
    CURRENT_DIR = Path(__file__).resolve().parent.parent  # daily_standup directory
    AUDIO_DIR = CURRENT_DIR / "audio"
    TEMP_DIR = CURRENT_DIR / "temp"
    REPORTS_DIR = CURRENT_DIR / "reports"
    
    # =============================================================================
    # TTS CONFIGURATION
    # =============================================================================
    TTS_VOICE = "en-IN-PrabhatNeural"
    TTS_RATE = "+25%"
    TTS_CHUNK_SIZE = 30
    TTS_OVERLAP = 3
    
    # =============================================================================
    # DYNAMIC QUESTIONING CONFIGURATION (From Old System)
    # =============================================================================
    TOTAL_QUESTIONS = 20  # Baseline hint for ratio calculation
    MIN_QUESTIONS_PER_CONCEPT = 1  # Minimum questions per concept
    MAX_QUESTIONS_PER_CONCEPT = 4  # Maximum questions per concept for balance
    ESTIMATED_SECONDS_PER_QUESTION = 180  # 3 minutes, for UI timer estimation
    
    # =============================================================================
    # CONVERSATION SETTINGS
    # =============================================================================
    CONVERSATION_WINDOW_SIZE = 5  # Conversation history window per concept
    MAX_RECORDING_TIME = 25.0
    SILENCE_THRESHOLD = 800
    
    # =============================================================================
    # AI MODEL CONFIGURATION
    # =============================================================================
    OPENAI_MODEL = "gpt-4o-mini"
    OPENAI_TEMPERATURE = 0.7
    OPENAI_MAX_TOKENS = 300
    GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
    
    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    APP_TITLE = "Ultra-Fast Daily Standup System"
    APP_VERSION = "2.0.0"
    WEBSOCKET_TIMEOUT = 300.0
    
    # =============================================================================
    # DEVELOPMENT FLAGS
    # =============================================================================
    USE_DUMMY_DATA = os.getenv("USE_DUMMY_DATA", "true").lower() == "true"
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # =============================================================================
    # DATABASE COLLECTION NAMES
    # =============================================================================
    TRANSCRIPTS_COLLECTION = os.getenv("MONGODB_TRANSCRIPTS_COLLECTION", "original-1")
    RESULTS_COLLECTION = os.getenv("MONGODB_RESULTS_COLLECTION", "daily_standup_results-1")
    
    # =============================================================================
    # PERFORMANCE SETTINGS
    # =============================================================================
    THREAD_POOL_MAX_WORKERS = 4
    MONGO_MAX_POOL_SIZE = 50
    MONGO_SERVER_SELECTION_TIMEOUT = 5000
    
    # =============================================================================
    # CORS SETTINGS
    # =============================================================================
    CORS_ALLOW_ORIGINS = ["*"]
    CORS_ALLOW_CREDENTIALS = True
    CORS_ALLOW_METHODS = ["*"]
    CORS_ALLOW_HEADERS = ["*"]

# Global config instance
config = Config()

# Ensure directories exist
for directory in [config.AUDIO_DIR, config.TEMP_DIR, config.REPORTS_DIR]:
    directory.mkdir(exist_ok=True)