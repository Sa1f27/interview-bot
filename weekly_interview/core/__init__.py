# weekly_interview/core/__init__.py
"""
Core module for Enhanced Mock Interview System
Exports all essential components for clean imports
"""

from .config import config
from .database import DatabaseManager, get_db_manager
from .content_service import ContentService
from .ai_services import (
    shared_clients,
    SharedClientManager,
    InterviewSession,
    InterviewStage,
    InterviewState,
    ConversationExchange,
    InterviewSessionManager,
    OptimizedAudioProcessor,
    UltraFastTTSProcessor,
    OptimizedConversationManager
)

__all__ = [
    'config',
    'DatabaseManager',
    'get_db_manager', 
    'ContentService',
    'shared_clients',
    'SharedClientManager',
    'InterviewSession',
    'InterviewStage',
    'InterviewState',
    'ConversationExchange',
    'InterviewSessionManager',
    'OptimizedAudioProcessor',
    'UltraFastTTSProcessor',
    'OptimizedConversationManager'
]