# weekend_mocktest/core/content_service.py
import logging
import random
import re
from typing import List, Dict, Any
from .config import config
from .database import get_db_manager
from .dummy_data import get_dummy_data_service

logger = logging.getLogger(__name__)

class ContentService:
    """Service for processing summaries and creating context for questions"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.use_dummy = config.USE_DUMMY_DATA
        
        if self.use_dummy:
            self.dummy_service = get_dummy_data_service()
            logger.info("🔧 Content service using dummy data")
    
    def extract_bullet_points(self, summary_text: str) -> List[str]:
        """Extract numbered bullet points from summary text"""
        try:
            # Pattern to match numbered bullet points like "1. text..." or "1) text..."
            pattern = r'^\d+[\.\)]\s*(.+?)(?=^\d+[\.\)]|\Z)'
            
            # Split by lines and rejoin to handle multiline points
            lines = summary_text.strip().split('\n')
            bullet_points = []
            current_point = ""
            
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+[\.\)]\s', line):
                    # Save previous point if exists
                    if current_point:
                        bullet_points.append(current_point.strip())
                    # Start new point
                    current_point = re.sub(r'^\d+[\.\)]\s*', '', line)
                else:
                    # Continue current point
                    if current_point and line:
                        current_point += " " + line
            
            # Add the last point
            if current_point:
                bullet_points.append(current_point.strip())
            
            # Filter out empty or very short points
            valid_points = [point for point in bullet_points if len(point.strip()) > 20]
            
            logger.info(f"✅ Extracted {len(valid_points)} bullet points from summary")
            return valid_points
            
        except Exception as e:
            logger.error(f"❌ Failed to extract bullet points: {e}")
            return []
    
    def slice_content_randomly(self, content: str, fraction: float = None) -> str:
        """Randomly slice a fraction of content from a random position"""
        if fraction is None:
            fraction = config.SUMMARY_SLICE_FRACTION
        
        try:
            content = content.strip()
            if not content:
                return ""
            
            total_length = len(content)
            slice_length = int(total_length * fraction)
            
            # Ensure minimum slice length
            slice_length = max(slice_length, 50)
            slice_length = min(slice_length, total_length)
            
            # Random start position
            max_start = max(0, total_length - slice_length)
            start_pos = random.randint(0, max_start) if max_start > 0 else 0
            end_pos = start_pos + slice_length
            
            sliced_content = content[start_pos:end_pos]
            
            # Try to end at a sentence boundary for better readability
            last_period = sliced_content.rfind('.')
            last_space = sliced_content.rfind(' ')
            
            if last_period > len(sliced_content) * 0.7:  # If period is in last 30%
                sliced_content = sliced_content[:last_period + 1]
            elif last_space > len(sliced_content) * 0.8:  # If space is in last 20%
                sliced_content = sliced_content[:last_space]
            
            logger.debug(f"Sliced content: {len(sliced_content)}/{total_length} chars")
            return sliced_content.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to slice content: {e}")
            return content  # Return original on error
    
    def process_summary_for_context(self, summary_doc: Dict[str, Any]) -> str:
        """Process a single summary document into context"""
        try:
            summary_text = summary_doc.get("summary", "")
            if not summary_text:
                return ""
            
            # Extract bullet points
            bullet_points = self.extract_bullet_points(summary_text)
            
            if not bullet_points:
                # Fallback: use raw summary with slicing
                return self.slice_content_randomly(summary_text)
            
            # Randomly select and slice bullet points
            selected_points = []
            num_points_to_select = max(1, int(len(bullet_points) * config.SUMMARY_SLICE_FRACTION))
            
            if num_points_to_select >= len(bullet_points):
                selected_points = bullet_points
            else:
                selected_points = random.sample(bullet_points, num_points_to_select)
            
            # Combine selected points
            combined_content = ". ".join(selected_points)
            
            # Apply final slicing to ensure variety
            final_content = self.slice_content_randomly(combined_content, fraction=0.8)
            
            logger.debug(f"Processed summary: {len(bullet_points)} -> {len(selected_points)} points")
            return final_content
            
        except Exception as e:
            logger.error(f"❌ Failed to process summary: {e}")
            return ""
    
    def get_context_for_questions(self, user_type: str = "dev") -> str:
        """Get context from recent summaries for question generation"""
        try:
            logger.info(f"🔍 Getting context for {user_type} questions")
            
            if self.use_dummy:
                logger.info("🔧 Using dummy summaries for context")
                recent_summaries = self.dummy_service.get_recent_summaries(
                    limit=config.RECENT_SUMMARIES_COUNT
                )
            else:
                # Fetch recent summaries from database
                recent_summaries = self.db_manager.get_recent_summaries(
                    limit=config.RECENT_SUMMARIES_COUNT
                )
            
            if not recent_summaries:
                raise Exception("No recent summaries available")
            
            # Process each summary
            context_parts = []
            for i, summary_doc in enumerate(recent_summaries, 1):
                processed_content = self.process_summary_for_context(summary_doc)
                
                if processed_content and len(processed_content.strip()) > 30:
                    # Add source identifier for variety
                    context_parts.append(f"Source {i}: {processed_content}")
            
            if not context_parts:
                raise Exception("No valid content extracted from summaries")
            
            # Combine all context parts
            combined_context = "\n\n".join(context_parts)
            
            # Add user type specific context hint
            if user_type == "dev":
                context_prefix = "Programming and Development Context:\n\n"
            else:
                context_prefix = "Technical Concepts and Analysis Context:\n\n"
            
            final_context = context_prefix + combined_context
            
            # Validate final context quality
            if len(final_context) < 200:
                raise Exception(f"Context too short: {len(final_context)} chars, need at least 200")
            
            logger.info(f"✅ Context generated: {len(final_context)} chars from {len(context_parts)} sources")
            return final_context
            
        except Exception as e:
            logger.error(f"❌ Context generation failed: {e}")
            raise Exception(f"Context generation failed: {e}")
    
    def validate_context_quality(self, context: str) -> Dict[str, Any]:
        """Validate the quality of generated context"""
        try:
            stats = {
                "length": len(context),
                "word_count": len(context.split()),
                "source_count": context.count("Source "),
                "has_technical_terms": any(term in context.lower() for term in [
                    "development", "programming", "algorithm", "data", "system",
                    "process", "implementation", "analysis", "framework"
                ]),
                "quality_score": 0
            }
            
            # Calculate quality score
            quality_factors = []
            
            # Length factor
            if stats["length"] >= 500:
                quality_factors.append(0.3)
            elif stats["length"] >= 200:
                quality_factors.append(0.2)
            else:
                quality_factors.append(0.1)
            
            # Source diversity factor
            if stats["source_count"] >= 5:
                quality_factors.append(0.3)
            elif stats["source_count"] >= 3:
                quality_factors.append(0.2)
            else:
                quality_factors.append(0.1)
            
            # Technical relevance factor
            if stats["has_technical_terms"]:
                quality_factors.append(0.4)
            else:
                quality_factors.append(0.2)
            
            stats["quality_score"] = sum(quality_factors)
            stats["is_high_quality"] = stats["quality_score"] >= 0.7
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Context validation failed: {e}")
            return {"error": str(e), "is_high_quality": False}

# Singleton pattern for content service
_content_service = None

def get_content_service() -> ContentService:
    """Get content service instance (singleton)"""
    global _content_service
    if _content_service is None:
        _content_service = ContentService()
    return _content_service# weekend_mocktest/core/content_service.py
import logging
import random
import re
from typing import List, Dict, Any
from .config import config
from .database import get_db_manager

logger = logging.getLogger(__name__)

class ContentService:
    """Service for processing summaries and creating context for questions"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
    
    def extract_bullet_points(self, summary_text: str) -> List[str]:
        """Extract numbered bullet points from summary text"""
        try:
            # Pattern to match numbered bullet points like "1. text..." or "1) text..."
            pattern = r'^\d+[\.\)]\s*(.+?)(?=^\d+[\.\)]|\Z)'
            
            # Split by lines and rejoin to handle multiline points
            lines = summary_text.strip().split('\n')
            bullet_points = []
            current_point = ""
            
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+[\.\)]\s', line):
                    # Save previous point if exists
                    if current_point:
                        bullet_points.append(current_point.strip())
                    # Start new point
                    current_point = re.sub(r'^\d+[\.\)]\s*', '', line)
                else:
                    # Continue current point
                    if current_point and line:
                        current_point += " " + line
            
            # Add the last point
            if current_point:
                bullet_points.append(current_point.strip())
            
            # Filter out empty or very short points
            valid_points = [point for point in bullet_points if len(point.strip()) > 20]
            
            logger.info(f"✅ Extracted {len(valid_points)} bullet points from summary")
            return valid_points
            
        except Exception as e:
            logger.error(f"❌ Failed to extract bullet points: {e}")
            return []
    
    def slice_content_randomly(self, content: str, fraction: float = None) -> str:
        """Randomly slice a fraction of content from a random position"""
        if fraction is None:
            fraction = config.SUMMARY_SLICE_FRACTION
        
        try:
            content = content.strip()
            if not content:
                return ""
            
            total_length = len(content)
            slice_length = int(total_length * fraction)
            
            # Ensure minimum slice length
            slice_length = max(slice_length, 50)
            slice_length = min(slice_length, total_length)
            
            # Random start position
            max_start = max(0, total_length - slice_length)
            start_pos = random.randint(0, max_start) if max_start > 0 else 0
            end_pos = start_pos + slice_length
            
            sliced_content = content[start_pos:end_pos]
            
            # Try to end at a sentence boundary for better readability
            last_period = sliced_content.rfind('.')
            last_space = sliced_content.rfind(' ')
            
            if last_period > len(sliced_content) * 0.7:  # If period is in last 30%
                sliced_content = sliced_content[:last_period + 1]
            elif last_space > len(sliced_content) * 0.8:  # If space is in last 20%
                sliced_content = sliced_content[:last_space]
            
            logger.debug(f"Sliced content: {len(sliced_content)}/{total_length} chars")
            return sliced_content.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to slice content: {e}")
            return content  # Return original on error
    
    def process_summary_for_context(self, summary_doc: Dict[str, Any]) -> str:
        """Process a single summary document into context"""
        try:
            summary_text = summary_doc.get("summary", "")
            if not summary_text:
                return ""
            
            # Extract bullet points
            bullet_points = self.extract_bullet_points(summary_text)
            
            if not bullet_points:
                # Fallback: use raw summary with slicing
                return self.slice_content_randomly(summary_text)
            
            # Randomly select and slice bullet points
            selected_points = []
            num_points_to_select = max(1, int(len(bullet_points) * config.SUMMARY_SLICE_FRACTION))
            
            if num_points_to_select >= len(bullet_points):
                selected_points = bullet_points
            else:
                selected_points = random.sample(bullet_points, num_points_to_select)
            
            # Combine selected points
            combined_content = ". ".join(selected_points)
            
            # Apply final slicing to ensure variety
            final_content = self.slice_content_randomly(combined_content, fraction=0.8)
            
            logger.debug(f"Processed summary: {len(bullet_points)} -> {len(selected_points)} points")
            return final_content
            
        except Exception as e:
            logger.error(f"❌ Failed to process summary: {e}")
            return ""
    
    def get_context_for_questions(self, user_type: str = "dev") -> str:
        """Get context from recent summaries for question generation"""
        try:
            logger.info(f"🔍 Getting context for {user_type} questions")
            
            # Fetch recent summaries
            recent_summaries = self.db_manager.get_recent_summaries(
                limit=config.RECENT_SUMMARIES_COUNT
            )
            
            if not recent_summaries:
                raise Exception("No recent summaries available")
            
            # Process each summary
            context_parts = []
            for i, summary_doc in enumerate(recent_summaries, 1):
                processed_content = self.process_summary_for_context(summary_doc)
                
                if processed_content and len(processed_content.strip()) > 30:
                    # Add source identifier for variety
                    context_parts.append(f"Source {i}: {processed_content}")
            
            if not context_parts:
                raise Exception("No valid content extracted from summaries")
            
            # Combine all context parts
            combined_context = "\n\n".join(context_parts)
            
            # Add user type specific context hint
            if user_type == "dev":
                context_prefix = "Programming and Development Context:\n\n"
            else:
                context_prefix = "Technical Concepts and Analysis Context:\n\n"
            
            final_context = context_prefix + combined_context
            
            # Validate final context quality
            if len(final_context) < 200:
                raise Exception(f"Context too short: {len(final_context)} chars, need at least 200")
            
            logger.info(f"✅ Context generated: {len(final_context)} chars from {len(context_parts)} sources")
            return final_context
            
        except Exception as e:
            logger.error(f"❌ Context generation failed: {e}")
            raise Exception(f"Context generation failed: {e}")
    
    def validate_context_quality(self, context: str) -> Dict[str, Any]:
        """Validate the quality of generated context"""
        try:
            stats = {
                "length": len(context),
                "word_count": len(context.split()),
                "source_count": context.count("Source "),
                "has_technical_terms": any(term in context.lower() for term in [
                    "development", "programming", "algorithm", "data", "system",
                    "process", "implementation", "analysis", "framework"
                ]),
                "quality_score": 0
            }
            
            # Calculate quality score
            quality_factors = []
            
            # Length factor
            if stats["length"] >= 500:
                quality_factors.append(0.3)
            elif stats["length"] >= 200:
                quality_factors.append(0.2)
            else:
                quality_factors.append(0.1)
            
            # Source diversity factor
            if stats["source_count"] >= 5:
                quality_factors.append(0.3)
            elif stats["source_count"] >= 3:
                quality_factors.append(0.2)
            else:
                quality_factors.append(0.1)
            
            # Technical relevance factor
            if stats["has_technical_terms"]:
                quality_factors.append(0.4)
            else:
                quality_factors.append(0.2)
            
            stats["quality_score"] = sum(quality_factors)
            stats["is_high_quality"] = stats["quality_score"] >= 0.7
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Context validation failed: {e}")
            return {"error": str(e), "is_high_quality": False}

# Singleton pattern for content service
_content_service = None

def get_content_service() -> ContentService:
    """Get content service instance (singleton)"""
    global _content_service
    if _content_service is None:
        _content_service = ContentService()
    return _content_service