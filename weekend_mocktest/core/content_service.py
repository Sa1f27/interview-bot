# weekend_mocktest/core/content_service.py
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
        # MongoDB is always live now, only AI service uses dummy mode
        logger.info("?? Content service using LIVE MongoDB data")
    
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
            
            logger.debug(f"?? Extracted {len(valid_points)} bullet points from summary")
            return valid_points
            
        except Exception as e:
            logger.error(f"? Failed to extract bullet points: {e}")
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
            
            logger.debug(f"?? Sliced content: {len(sliced_content)}/{total_length} chars")
            return sliced_content.strip()
            
        except Exception as e:
            logger.error(f"? Failed to slice content: {e}")
            return content  # Return original on error
    
    def process_summary_for_context(self, summary_doc: Dict[str, Any]) -> str:
        """Process a single summary document into context"""
        try:
            summary_text = summary_doc.get("summary", "")
            if not summary_text:
                logger.warning("?? Empty summary found in document")
                return ""
            
            # Log summary info for debugging
            doc_id = summary_doc.get("_id", "unknown")
            logger.debug(f"?? Processing summary {doc_id}: {len(summary_text)} chars")
            
            # Extract bullet points
            bullet_points = self.extract_bullet_points(summary_text)
            
            if not bullet_points:
                # Fallback: use raw summary with slicing
                logger.debug("?? No bullet points found, using raw summary")
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
            
            logger.debug(f"? Processed summary: {len(bullet_points)} -> {len(selected_points)} points")
            return final_content
            
        except Exception as e:
            logger.error(f"? Failed to process summary: {e}")
            return ""
    
    def get_context_for_questions(self, user_type: str = "dev") -> str:
        """Get context from recent summaries for question generation - USING REAL MONGODB DATA"""
        try:
            logger.info(f"?? Getting context for {user_type} questions from LIVE MongoDB")
            
            # Fetch recent summaries from MongoDB (always live now)
            recent_summaries = self.db_manager.get_recent_summaries(
                limit=config.RECENT_SUMMARIES_COUNT
            )
            
            if not recent_summaries:
                raise Exception("No recent summaries available from MongoDB")
            
            logger.info(f"?? Retrieved {len(recent_summaries)} summaries from MongoDB")
            
            # Process each summary
            context_parts = []
            for i, summary_doc in enumerate(recent_summaries, 1):
                processed_content = self.process_summary_for_context(summary_doc)
                
                if processed_content and len(processed_content.strip()) > 30:
                    # Add source identifier for variety
                    doc_id = summary_doc.get("_id", f"doc_{i}")
                    context_parts.append(f"Source {i} (ID: {str(doc_id)[:8]}...): {processed_content}")
                    logger.debug(f"? Added context from source {i}: {len(processed_content)} chars")
            
            if not context_parts:
                raise Exception("No valid content extracted from MongoDB summaries")
            
            # Combine all context parts
            combined_context = "\n\n".join(context_parts)
            
            # Add user type specific context hint
            if user_type == "dev":
                context_prefix = "Programming and Development Context (from recent project summaries):\n\n"
            else:
                context_prefix = "Technical Concepts and Analysis Context (from recent project summaries):\n\n"
            
            final_context = context_prefix + combined_context
            
            # Validate final context quality
            if len(final_context) < 200:
                raise Exception(f"Context too short: {len(final_context)} chars, need at least 200")
            
            logger.info(f"? Context generated from LIVE data: {len(final_context)} chars from {len(context_parts)} MongoDB sources")
            
            # Log a sample of the context for verification
            sample_context = final_context[:300] + "..." if len(final_context) > 300 else final_context
            logger.debug(f"?? Context sample: {sample_context}")
            
            return final_context
            
        except Exception as e:
            logger.error(f"? Context generation failed: {e}")
            raise Exception(f"Context generation failed: {e}")
    
    def validate_context_quality(self, context: str) -> Dict[str, Any]:
        """Validate the quality of generated context"""
        try:
            stats = {
                "length": len(context),
                "word_count": len(context.split()),
                "source_count": context.count("Source "),
                "mongodb_sources": context.count("ID:"),  # MongoDB document IDs
                "has_technical_terms": any(term in context.lower() for term in [
                    "development", "programming", "algorithm", "data", "system",
                    "process", "implementation", "analysis", "framework", "project"
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
            stats["data_source"] = "live_mongodb"
            
            logger.info(f"?? Context quality: {stats['quality_score']:.2f}/1.0 from {stats['source_count']} MongoDB sources")
            
            return stats
            
        except Exception as e:
            logger.error(f"? Context validation failed: {e}")
            return {"error": str(e), "is_high_quality": False, "data_source": "unknown"}
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about available context data"""
        try:
            # Get summary count from MongoDB
            total_summaries = self.db_manager.summaries_collection.count_documents(
                {"summary": {"$exists": True, "$ne": ""}}
            )
            
            # Get recent summaries info
            recent_summaries = self.db_manager.get_recent_summaries(limit=1)
            latest_summary_date = None
            
            if recent_summaries:
                latest_doc = recent_summaries[0]
                latest_summary_date = latest_doc.get("date") or latest_doc.get("timestamp")
            
            stats = {
                "total_summaries_available": total_summaries,
                "summaries_used_for_context": config.RECENT_SUMMARIES_COUNT,
                "latest_summary_date": latest_summary_date,
                "collection_name": config.SUMMARIES_COLLECTION,
                "database_name": config.MONGO_DB_NAME,
                "data_source": "live_mongodb",
                "context_slice_fraction": config.SUMMARY_SLICE_FRACTION
            }
            
            logger.info(f"?? Context stats: {total_summaries} total summaries, using latest {config.RECENT_SUMMARIES_COUNT}")
            
            return stats
            
        except Exception as e:
            logger.error(f"? Failed to get context stats: {e}")
            return {"error": str(e), "data_source": "unknown"}

# Singleton pattern for content service
_content_service = None

def get_content_service() -> ContentService:
    """Get content service instance (singleton)"""
    global _content_service
    if _content_service is None:
        _content_service = ContentService()
    return _content_service