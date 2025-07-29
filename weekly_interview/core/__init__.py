# weekly_interview/core/__init__.py
"""
Core module for Enhanced Mock Interview System
Exports all essential components for clean imports
FIXED: Updated imports after modularization and Unicode fixes
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
    OptimizedConversationManager,
    EnhancedInterviewFragmentManager
)
from .tts_processor import UltraFastTTSProcessor
from .prompts import (
    build_stage_prompt,
    build_conversation_prompt,
    build_evaluation_prompt,
    validate_prompts
)

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
    'UltraFastTTSProcessor',
    'OptimizedConversationManager',
    'EnhancedInterviewFragmentManager',
    'build_stage_prompt',
    'build_conversation_prompt',
    'build_evaluation_prompt',
    'validate_prompts'
]