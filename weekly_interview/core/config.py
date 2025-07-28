# weekly_interview/core/config.py
"""
Configuration module for Enhanced Mock Interview System
Environment-driven configuration with no hardcoded values
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

class Config:
    """Central configuration class for Mock Interview System"""
    
    # =============================================================================
    # PATHS AND DIRECTORIES
    # =============================================================================
    CURRENT_DIR = Path(__file__).resolve().parent.parent  # weekly_interview directory
    AUDIO_DIR = CURRENT_DIR / "audio"
    TEMP_DIR = CURRENT_DIR / "temp"
    REPORTS_DIR = CURRENT_DIR / "reports"
    
    # =============================================================================
    # DATABASE CONFIGURATION - MYSQL (Student Data)
    # =============================================================================
    MYSQL_HOST = os.getenv("MYSQL_HOST", "192.168.48.201")
    MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "SuperDB")
    MYSQL_USER = os.getenv("MYSQL_USER", "sa")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "Welcome@123")
    
    # =============================================================================
    # DATABASE CONFIGURATION - MONGODB (Interview Content)
    # =============================================================================
    MONGODB_HOST = os.getenv("MONGODB_HOST", "192.168.48.201")
    MONGODB_PORT = int(os.getenv("MONGODB_PORT", "27017"))
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "ml_notes")
    MONGODB_USERNAME = os.getenv("MONGODB_USERNAME", "connectly")
    MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD", "LT@connect25")
    MONGODB_AUTH_SOURCE = os.getenv("MONGODB_AUTH_SOURCE", "admin")
    
    # =============================================================================
    # COLLECTION NAMES
    # =============================================================================
    SUMMARIES_COLLECTION = os.getenv("SUMMARIES_COLLECTION", "summaries")
    INTERVIEW_RESULTS_COLLECTION = os.getenv("INTERVIEW_RESULTS_COLLECTION", "interview_results")
    
    # =============================================================================
    # INTERVIEW CONFIGURATION
    # =============================================================================
    INTERVIEW_DURATION_MINUTES = int(os.getenv("INTERVIEW_DURATION_MINUTES", "60"))  # 45-60 minutes
    QUESTIONS_PER_ROUND = int(os.getenv("QUESTIONS_PER_ROUND", "6"))
    MIN_QUESTIONS_PER_ROUND = int(os.getenv("MIN_QUESTIONS_PER_ROUND", "4"))
    MAX_QUESTIONS_PER_ROUND = int(os.getenv("MAX_QUESTIONS_PER_ROUND", "8"))
    
    # Interview rounds configuration
    TOTAL_ROUNDS = 4  # Greeting, Technical, Communication, HR
    ROUND_NAMES = ["greeting", "technical", "communication", "hr"]
    
    # =============================================================================
    # CONTENT PROCESSING CONFIGURATION
    # =============================================================================
    RECENT_SUMMARIES_DAYS = int(os.getenv("RECENT_SUMMARIES_DAYS", "7"))  # Last 7 days
    SUMMARIES_LIMIT = int(os.getenv("SUMMARIES_LIMIT", "10"))  # Max summaries to fetch
    CONTENT_SLICE_FRACTION = float(os.getenv("CONTENT_SLICE_FRACTION", "0.4"))  # 40% of content
    MIN_CONTENT_LENGTH = int(os.getenv("MIN_CONTENT_LENGTH", "200"))  # Minimum content length
    
    # =============================================================================
    # AUDIO CONFIGURATION
    # =============================================================================
    TTS_VOICE = os.getenv("TTS_VOICE", "en-IN-PrabhatNeural")
    TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))  # Normal speed
    TTS_CHUNK_SIZE = int(os.getenv("TTS_CHUNK_SIZE", "100"))  # Characters per chunk
    
    # Audio processing
    MAX_RECORDING_DURATION = int(os.getenv("MAX_RECORDING_DURATION", "30"))  # 30 seconds
    SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.01"))
    SILENCE_DURATION = int(os.getenv("SILENCE_DURATION", "2"))  # 2 seconds
    
    # =============================================================================
    # AI MODEL CONFIGURATION
    # =============================================================================
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "800"))
    
    GROQ_MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3-turbo")
    GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "30"))
    
    # =============================================================================
    # WEBSOCKET CONFIGURATION
    # =============================================================================
    WEBSOCKET_TIMEOUT = float(os.getenv("WEBSOCKET_TIMEOUT", "300.0"))  # 5 minutes
    PING_INTERVAL = int(os.getenv("PING_INTERVAL", "30"))  # 30 seconds
    MAX_MESSAGE_SIZE = int(os.getenv("MAX_MESSAGE_SIZE", "16777216"))  # 16MB
    
    # =============================================================================
    # SESSION MANAGEMENT
    # =============================================================================
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour
    MAX_ACTIVE_SESSIONS = int(os.getenv("MAX_ACTIVE_SESSIONS", "100"))
    CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "300"))  # 5 minutes
    
    # =============================================================================
    # PERFORMANCE SETTINGS
    # =============================================================================
    THREAD_POOL_MAX_WORKERS = int(os.getenv("THREAD_POOL_MAX_WORKERS", "8"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))
    
    # Database connection pooling
    MYSQL_POOL_SIZE = int(os.getenv("MYSQL_POOL_SIZE", "20"))
    MONGODB_POOL_SIZE = int(os.getenv("MONGODB_POOL_SIZE", "50"))
    
    # =============================================================================
    # CACHING CONFIGURATION
    # =============================================================================
    CACHE_DURATION_HOURS = int(os.getenv("CACHE_DURATION_HOURS", "6"))
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    # =============================================================================
    # CORS SETTINGS
    # =============================================================================
    CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    CORS_ALLOW_METHODS = ["*"]
    CORS_ALLOW_HEADERS = ["*"]
    
    # =============================================================================
    # SSL/HTTPS CONFIGURATION
    # =============================================================================
    USE_SSL = os.getenv("USE_SSL", "true").lower() == "true"
    SSL_CERT_PATH = os.getenv("SSL_CERT_PATH", "./certs/cert.pem")
    SSL_KEY_PATH = os.getenv("SSL_KEY_PATH", "./certs/key.pem")
    
    # =============================================================================
    # EVALUATION CONFIGURATION
    # =============================================================================
    STRICT_EVALUATION = os.getenv("STRICT_EVALUATION", "true").lower() == "true"
    EVALUATION_CRITERIA = {
        "technical_weight": 0.35,      # 35% Technical Assessment
        "communication_weight": 0.30,  # 30% Communication Skills  
        "behavioral_weight": 0.25,     # 25% Behavioral/HR
        "overall_presentation": 0.10   # 10% Overall Presentation
    }
    
    # Scoring thresholds for strict evaluation
    EXCELLENT_THRESHOLD = 8.5  # 85%+
    GOOD_THRESHOLD = 7.0       # 70%+
    ACCEPTABLE_THRESHOLD = 6.0  # 60%+
    
    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    APP_TITLE = "Enhanced Mock Interview System"
    APP_VERSION = "3.0.0"
    APP_DESCRIPTION = "AI-powered mock interview system with real-time WebSocket communication"
    
    # API endpoints configuration
    API_PREFIX = "/weekly_interview"
    WEBSOCKET_ENDPOINT = "/ws"
    
    # =============================================================================
    # LOGGING CONFIGURATION
    # =============================================================================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # =============================================================================
    # ENVIRONMENT VALIDATION
    # =============================================================================
    @staticmethod
    def validate_required_env_vars():
        """Validate that all required environment variables are set"""
        required_vars = [
            "OPENAI_API_KEY",
            "GROQ_API_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    @property
    def mysql_connection_config(self) -> dict:
        """Get MySQL connection configuration"""
        return {
            'host': self.MYSQL_HOST,
            'port': self.MYSQL_PORT,
            'database': self.MYSQL_DATABASE,
            'user': self.MYSQL_USER,
            'password': self.MYSQL_PASSWORD
        }
    
    @property
    def mongodb_connection_string(self) -> str:
        """Get MongoDB connection string"""
        from urllib.parse import quote_plus
        username = quote_plus(self.MONGODB_USERNAME)
        password = quote_plus(self.MONGODB_PASSWORD)
        return f"mongodb://{username}:{password}@{self.MONGODB_HOST}:{self.MONGODB_PORT}/{self.MONGODB_AUTH_SOURCE}"
    
    def get_round_config(self, round_name: str) -> dict:
        """Get configuration for specific interview round"""
        round_configs = {
            "greeting": {
                "duration_minutes": 2,
                "max_questions": 2,
                "focus": "introduction_and_rapport"
            },
            "technical": {
                "duration_minutes": 25,
                "max_questions": self.MAX_QUESTIONS_PER_ROUND,
                "focus": "technical_skills_assessment"
            },
            "communication": {
                "duration_minutes": 20,
                "max_questions": self.MAX_QUESTIONS_PER_ROUND,
                "focus": "communication_and_presentation"
            },
            "hr": {
                "duration_minutes": 13,
                "max_questions": self.MAX_QUESTIONS_PER_ROUND,
                "focus": "behavioral_and_cultural_fit"
            }
        }
        return round_configs.get(round_name, {})

# Global config instance
config = Config()

# Validate required environment variables on import
config.validate_required_env_vars()

# Ensure directories exist
for directory in [config.AUDIO_DIR, config.TEMP_DIR, config.REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)