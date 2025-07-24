# weekend_mocktest/core/config.py
import os
from pathlib import Path
from typing import Dict, Any
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    """Centralized configuration management"""
    
    # ==================== API Configuration ====================
    API_TITLE = "Mock Test API"
    API_DESCRIPTION = "Lightweight mock testing system"
    API_VERSION = "5.0.0-modular"
    
    # ==================== Database Configuration ====================
    # SQL Server
    DB_CONFIG = {
        "DRIVER": os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server"),
        "SERVER": os.getenv("SQL_SERVER", "192.168.48.200"),
        "DATABASE": os.getenv("SQL_DATABASE", "SuperDB"),
        "UID": os.getenv("SQL_UID", "sa"),
        "PWD": os.getenv("SQL_PWD", "Welcome@123"),
    }
    
    @property
    def SQL_CONNECTION_STRING(self) -> str:
        return (
            f"DRIVER={{{self.DB_CONFIG['DRIVER']}}};"
            f"SERVER={self.DB_CONFIG['SERVER']};"
            f"DATABASE={self.DB_CONFIG['DATABASE']};"
            f"UID={self.DB_CONFIG['UID']};"
            f"PWD={self.DB_CONFIG['PWD']}"
        )
    
    # MongoDB
    MONGO_USER = os.getenv("MONGO_USER", "LanTech")
    MONGO_PASS = os.getenv("MONGO_PASS", "L@nc^ere@0012")
    MONGO_HOST = os.getenv("MONGO_HOST", "192.168.48.201:27017")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "Api-1")
    MONGO_AUTH_SOURCE = os.getenv("MONGO_AUTH_SOURCE", "admin")
    
    @property
    def MONGO_CONNECTION_STRING(self) -> str:
        return (
            f"mongodb://{quote_plus(self.MONGO_USER)}:"
            f"{quote_plus(self.MONGO_PASS)}@{self.MONGO_HOST}/"
            f"{self.MONGO_DB_NAME}?authSource={self.MONGO_AUTH_SOURCE}"
        )
    
    # Collections
    SUMMARIES_COLLECTION = "original-1"
    TEST_RESULTS_COLLECTION = "mock_test_results"
    
    # ==================== Development Settings ====================
    USE_DUMMY_DATA = os.getenv("USE_DUMMY_DATA", "true").lower() == "true"
    
    # ==================== Content Generation Configuration ====================
    # Summary processing
    RECENT_SUMMARIES_COUNT = int(os.getenv("RECENT_SUMMARIES_COUNT", "7"))
    SUMMARY_SLICE_FRACTION = float(os.getenv("SUMMARY_SLICE_FRACTION", "0.33"))
    
    # Question generation
    QUESTIONS_PER_TEST = int(os.getenv("QUESTIONS_PER_TEST", "10"))
    QUESTION_CACHE_DURATION_HOURS = int(os.getenv("QUESTION_CACHE_DURATION_HOURS", "24"))
    
    # Question types rotation
    QUESTION_TYPES = ["practical", "conceptual", "analytical"]
    
    # ==================== Test Configuration ====================
    # Time limits (seconds)
    DEV_TIME_LIMIT = int(os.getenv("DEV_TIME_LIMIT", "300"))  # 5 minutes per question
    NON_DEV_TIME_LIMIT = int(os.getenv("NON_DEV_TIME_LIMIT", "120"))  # 2 minutes per question
    
    # Test expiration
    TEST_EXPIRATION_SECONDS = int(os.getenv("TEST_EXPIRATION_SECONDS", "3600"))  # 1 hour
    MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "1800"))  # 30 minutes
    
    # ==================== AI Service Configuration ====================
    # Groq settings
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "30"))
    GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
    GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "2000"))
    GROQ_TOP_P = float(os.getenv("GROQ_TOP_P", "0.9"))
    
    # Batch generation settings
    BATCH_GENERATION_RETRIES = int(os.getenv("BATCH_GENERATION_RETRIES", "3"))
    BATCH_GENERATION_TIMEOUT = int(os.getenv("BATCH_GENERATION_TIMEOUT", "60"))
    
    # ==================== Evaluation Configuration ====================
    EVALUATION_TEMPERATURE = float(os.getenv("EVALUATION_TEMPERATURE", "0.3"))
    EVALUATION_MAX_TOKENS = int(os.getenv("EVALUATION_MAX_TOKENS", "1500"))
    
    # ==================== PDF Configuration ====================
    PDF_FONT_SIZE = int(os.getenv("PDF_FONT_SIZE", "12"))
    PDF_TITLE_FONT_SIZE = int(os.getenv("PDF_TITLE_FONT_SIZE", "18"))
    PDF_PAGE_SIZE = os.getenv("PDF_PAGE_SIZE", "LETTER")
    
    # ==================== Environment Overrides ====================
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config with environment variable overrides"""
        config = cls()
        
        # All values are already loaded from environment in the class definition
        # This method can be extended for additional custom logic if needed
        
        return config
    
    # ==================== Validation ====================
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        if self.QUESTIONS_PER_TEST < 1:
            issues.append("QUESTIONS_PER_TEST must be at least 1")
        
        if self.RECENT_SUMMARIES_COUNT < 1:
            issues.append("RECENT_SUMMARIES_COUNT must be at least 1")
        
        if not (0 < self.SUMMARY_SLICE_FRACTION <= 1):
            issues.append("SUMMARY_SLICE_FRACTION must be between 0 and 1")
        
        if not self.USE_DUMMY_DATA and not self.GROQ_API_KEY:
            issues.append("GROQ_API_KEY is required when not using dummy data")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config_loaded": True,
            "using_dummy_data": self.USE_DUMMY_DATA
        }

# Global configuration instance
config = Config.from_env()

# Validate on import
validation_result = config.validate()
if not validation_result["valid"]:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Configuration issues: {validation_result['issues']}")