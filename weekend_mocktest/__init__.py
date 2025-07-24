# weekend_mocktest/__init__.py
"""
Mock Test Module - Modular Architecture
RAG-free, lightweight mock testing system with batch question generation
"""

__version__ = "5.0.0-modular"
__author__ = "Mock Test Team"
__description__ = "Modular mock testing system with AI-powered question generation"

# Core module exports
from .core.config import config
from .main import app

__all__ = ["app", "config"]

# ==========================================
# weekend_mocktest/core/__init__.py
"""
Core module containing configuration, database, AI services, and utilities
"""

from .config import config
from .database import get_db_manager
from .ai_services import get_ai_service
from .content_service import get_content_service

__all__ = [
    "config",
    "get_db_manager", 
    "get_ai_service",
    "get_content_service"
]

# ==========================================
# weekend_mocktest/models/__init__.py
"""
Pydantic models and schemas for request/response validation
"""

from .schemas import (
    StartTestRequest,
    SubmitAnswerRequest,
    TestResponse,
    SubmitAnswerResponse,
    TestResultsResponse,
    HealthResponse,
    NextQuestionResponse,
    QuestionData,
    AnswerData,
    TestData
)

__all__ = [
    "StartTestRequest",
    "SubmitAnswerRequest", 
    "TestResponse",
    "SubmitAnswerResponse",
    "TestResultsResponse",
    "HealthResponse",
    "NextQuestionResponse",
    "QuestionData",
    "AnswerData",
    "TestData"
]

# ==========================================
# weekend_mocktest/services/__init__.py
"""
Business logic services for test management, evaluation, and PDF generation
"""

from .test_service import get_test_service
from .pdf_service import get_pdf_service

__all__ = [
    "get_test_service",
    "get_pdf_service"
]

# ==========================================
# weekend_mocktest/api/__init__.py
"""
FastAPI routes and API layer
"""

from .routes import router

__all__ = ["router"]