# weekly_interview/core/__init__.py
"""
Core module for Enhanced Mock Interview System
Exports all essential components for clean imports
FIXED: Updated imports after TTS modularization
"""

from .config import config
from .database import DatabaseManager
from .content_service import ContentService
from .ai_services import (
    shared_clients,
    SharedClientManager,
    InterviewSession,
    InterviewStage,
    ConversationExchange,
    OptimizedAudioProcessor,
    OptimizedConversationManager
)
# FIXED: Import TTS processor from separate module
from .tts_processor import UltraFastTTSProcessor

__all__ = [
    'config',
    'DatabaseManager',
    'ContentService',
    'shared_clients',
    'SharedClientManager',
    'InterviewSession',
    'InterviewStage',
    'ConversationExchange',
    'OptimizedAudioProcessor',
    'UltraFastTTSProcessor',  # Now imported from tts_processor
    'OptimizedConversationManager'
]