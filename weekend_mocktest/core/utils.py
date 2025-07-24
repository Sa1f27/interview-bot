# utils.py
# weekend_mocktest/core/utils.py
import logging
import time
import gc
import threading
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .config import config

logger = logging.getLogger(__name__)

class MemoryManager:
    """Memory management for test data and caching"""
    
    def __init__(self):
        self.tests = {}  # Active test data
        self.answers = {}  # Test answers
        self.question_cache = {}  # Generated questions cache
        self._cleanup_thread = None
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        logger.info("✅ Memory cleanup thread started")
    
    def _periodic_cleanup(self):
        """Periodic cleanup of expired data"""
        while True:
            try:
                time.sleep(config.MEMORY_CLEANUP_INTERVAL)
                self.cleanup_expired_data()
            except Exception as e:
                logger.error(f"Cleanup thread error: {e}")
    
    def cleanup_expired_data(self):
        """Clean up expired tests and cache"""
        try:
            current_time = time.time()
            
            # Clean expired tests
            expired_tests = []
            for test_id, test_data in list(self.tests.items()):
                if current_time - test_data.get("created_at", 0) > config.TEST_EXPIRATION_SECONDS:
                    expired_tests.append(test_id)
            
            for test_id in expired_tests:
                self.tests.pop(test_id, None)
                self.answers.pop(test_id, None)
            
            # Clean expired question cache
            cache_expiry = config.QUESTION_CACHE_DURATION_HOURS * 3600
            expired_cache = []
            for cache_key, cache_data in list(self.question_cache.items()):
                if current_time - cache_data.get("created_at", 0) > cache_expiry:
                    expired_cache.append(cache_key)
            
            for cache_key in expired_cache:
                self.question_cache.pop(cache_key, None)
            
            # Force garbage collection
            gc.collect()
            
            if expired_tests or expired_cache:
                logger.info(f"🧹 Cleanup: removed {len(expired_tests)} tests, {len(expired_cache)} cache entries")
        
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def create_test(self, user_type: str, questions: List[Dict[str, Any]]) -> str:
        """Create new test with generated questions"""
        test_id = str(uuid.uuid4())
        
        self.tests[test_id] = {
            "user_type": user_type,
            "total_questions": len(questions),
            "current_question": 1,
            "questions": questions,
            "created_at": time.time(),
            "started_at": time.time()
        }
        
        self.answers[test_id] = []
        
        logger.info(f"✅ Test created: {test_id} with {len(questions)} questions")
        return test_id
    
    def get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get test data by ID"""
        return self.tests.get(test_id)
    
    def get_current_question(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get current question for test"""
        test = self.tests.get(test_id)
        if not test:
            return None
        
        current_q_num = test["current_question"]
        questions = test["questions"]
        
        if 1 <= current_q_num <= len(questions):
            question_data = questions[current_q_num - 1]
            return {
                "question_number": current_q_num,
                "total_questions": len(questions),
                "question_html": question_data["question"],
                "options": question_data.get("options"),
                "difficulty": question_data.get("difficulty", "Medium"),
                "type": question_data.get("type", "General")
            }
        
        return None
    
    def submit_answer(self, test_id: str, question_number: int, answer: str) -> bool:
        """Submit answer for test question"""
        test = self.tests.get(test_id)
        if not test:
            return False
        
        if question_number != test["current_question"]:
            return False
        
        # Get question data
        questions = test["questions"]
        if 1 <= question_number <= len(questions):
            question_data = questions[question_number - 1]
            
            # Store answer
            answer_data = {
                "question_number": question_number,
                "question": question_data["question"],
                "answer": answer,
                "options": question_data.get("options", []),
                "submitted_at": time.time()
            }
            
            self.answers[test_id].append(answer_data)
            
            # Move to next question
            test["current_question"] += 1
            
            logger.info(f"✅ Answer submitted: {test_id}, Q{question_number}")
            return True
        
        return False
    
    def is_test_complete(self, test_id: str) -> bool:
        """Check if test is completed"""
        test = self.tests.get(test_id)
        if not test:
            return False
        
        return test["current_question"] > test["total_questions"]
    
    def get_test_answers(self, test_id: str) -> List[Dict[str, Any]]:
        """Get all answers for test"""
        return self.answers.get(test_id, [])
    
    def cache_questions(self, cache_key: str, questions: List[Dict[str, Any]]):
        """Cache generated questions"""
        self.question_cache[cache_key] = {
            "questions": questions,
            "created_at": time.time()
        }
        logger.info(f"✅ Questions cached: {cache_key}")
    
    def get_cached_questions(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached questions if not expired"""
        cache_data = self.question_cache.get(cache_key)
        if not cache_data:
            return None
        
        # Check expiry
        cache_age = time.time() - cache_data["created_at"]
        max_age = config.QUESTION_CACHE_DURATION_HOURS * 3600
        
        if cache_age > max_age:
            self.question_cache.pop(cache_key, None)
            return None
        
        return cache_data["questions"]
    
    def cleanup_test(self, test_id: str):
        """Clean up specific test data"""
        self.tests.pop(test_id, None)
        self.answers.pop(test_id, None)
        logger.info(f"✅ Test cleaned up: {test_id}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "active_tests": len(self.tests),
            "cached_questions": len(self.question_cache),
            "total_answers": sum(len(answers) for answers in self.answers.values()),
            "cleanup_thread_alive": self._cleanup_thread.is_alive() if self._cleanup_thread else False
        }

class CacheManager:
    """Simple cache manager for frequently accessed data"""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str, default_ttl: int = 3600) -> Any:
        """Get cached value if not expired"""
        if key not in self._cache:
            return None
        
        # Check expiry
        if time.time() - self._timestamps.get(key, 0) > default_ttl:
            self.delete(key)
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cached value with TTL"""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def delete(self, key: str):
        """Delete cached value"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear_expired(self):
        """Clear all expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self._timestamps.items():
            if current_time - timestamp > 3600:  # Default TTL
                expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)

class ValidationUtils:
    """Utility functions for data validation"""
    
    @staticmethod
    def validate_user_type(user_type: str) -> bool:
        """Validate user type"""
        return user_type in ["dev", "non_dev"]
    
    @staticmethod
    def validate_test_id(test_id: str) -> bool:
        """Validate test ID format"""
        try:
            uuid.UUID(test_id)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_question_number(question_number: Any, total_questions: int) -> bool:
        """Validate question number"""
        try:
            q_num = int(question_number)
            return 1 <= q_num <= total_questions
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_answer(answer: str, user_type: str) -> bool:
        """Validate answer format"""
        if not answer or not answer.strip():
            return False
        
        if user_type == "non_dev":
            # For MCQ, answer should be option index (0-3) or option text
            return True  # We'll handle conversion in the service
        
        # For dev questions, any non-empty text is valid
        return len(answer.strip()) > 0
    
    @staticmethod
    def sanitize_input(input_str: str, max_length: int = 5000) -> str:
        """Sanitize user input"""
        if not input_str:
            return ""
        
        # Basic sanitization
        sanitized = input_str.strip()
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized

class DateTimeUtils:
    """Utility functions for date/time operations"""
    
    @staticmethod
    def get_current_timestamp() -> float:
        """Get current timestamp"""
        return time.time()
    
    @staticmethod
    def format_timestamp(timestamp: float, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format timestamp to string"""
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime(format_str)
        except (ValueError, OSError):
            return "Invalid timestamp"
    
    @staticmethod
    def get_cache_key(user_type: str, date: Optional[str] = None) -> str:
        """Generate cache key for questions"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        return f"questions_{user_type}_{date}"
    
    @staticmethod
    def is_same_day(timestamp1: float, timestamp2: float) -> bool:
        """Check if two timestamps are on the same day"""
        try:
            dt1 = datetime.fromtimestamp(timestamp1)
            dt2 = datetime.fromtimestamp(timestamp2)
            return dt1.date() == dt2.date()
        except (ValueError, OSError):
            return False

class ResponseFormatter:
    """Utility functions for formatting API responses"""
    
    @staticmethod
    def format_test_response(test_id: str, test_data: Dict[str, Any], 
                           current_question: Dict[str, Any]) -> Dict[str, Any]:
        """Format test start response"""
        return {
            "test_id": test_id,
            "user_type": test_data["user_type"],
            "question_number": current_question["question_number"],
            "total_questions": current_question["total_questions"],
            "question_html": current_question["question_html"],
            "options": current_question.get("options"),
            "time_limit": config.DEV_TIME_LIMIT if test_data["user_type"] == "dev" else config.NON_DEV_TIME_LIMIT
        }
    
    @staticmethod
    def format_next_question_response(current_question: Dict[str, Any], 
                                    user_type: str) -> Dict[str, Any]:
        """Format next question response"""
        return {
            "question_number": current_question["question_number"],
            "total_questions": current_question["total_questions"],
            "question_html": current_question["question_html"],
            "options": current_question.get("options"),
            "time_limit": config.DEV_TIME_LIMIT if user_type == "dev" else config.NON_DEV_TIME_LIMIT
        }
    
    @staticmethod
    def format_completion_response(evaluation_result: Dict[str, Any], 
                                 total_questions: int) -> Dict[str, Any]:
        """Format test completion response"""
        return {
            "test_completed": True,
            "score": evaluation_result["total_correct"],
            "total_questions": total_questions,
            "analytics": evaluation_result["evaluation_report"]
        }
    
    @staticmethod
    def format_error_response(error_message: str, component: str = None, 
                            action: str = None) -> Dict[str, Any]:
        """Format error response"""
        response = {
            "error": error_message,
            "timestamp": time.time()
        }
        
        if component:
            response["component"] = component
        if action:
            response["suggested_action"] = action
        
        return response

class PerformanceMonitor:
    """Simple performance monitoring utilities"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str) -> str:
        """Start timing an operation"""
        timer_id = f"{operation}_{int(time.time() * 1000)}"
        self.metrics[timer_id] = {"start": time.time(), "operation": operation}
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """End timing and return duration"""
        if timer_id not in self.metrics:
            return 0.0
        
        duration = time.time() - self.metrics[timer_id]["start"]
        self.metrics[timer_id]["duration"] = duration
        return duration
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        operations = {}
        for timer_data in self.metrics.values():
            if "duration" in timer_data:
                op = timer_data["operation"]
                if op not in operations:
                    operations[op] = {"count": 0, "total_time": 0, "avg_time": 0}
                
                operations[op]["count"] += 1
                operations[op]["total_time"] += timer_data["duration"]
                operations[op]["avg_time"] = operations[op]["total_time"] / operations[op]["count"]
        
        return operations
    
    def clear_old_metrics(self, max_age: int = 3600):
        """Clear metrics older than max_age seconds"""
        current_time = time.time()
        expired_keys = []
        
        for timer_id, timer_data in self.metrics.items():
            if current_time - timer_data["start"] > max_age:
                expired_keys.append(timer_id)
        
        for key in expired_keys:
            self.metrics.pop(key, None)

# Global instances
memory_manager = MemoryManager()
cache_manager = CacheManager()
performance_monitor = PerformanceMonitor()

# Cleanup function for graceful shutdown
def cleanup_all():
    """Clean up all resources"""
    try:
        memory_manager.cleanup_expired_data()
        cache_manager.clear_expired()
        performance_monitor.clear_old_metrics()
        logger.info("✅ All resources cleaned up")
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")

# Helper functions for easy access
def generate_test_id() -> str:
    """Generate unique test ID"""
    return str(uuid.uuid4())

def generate_cache_key(user_type: str) -> str:
    """Generate cache key for today's questions"""
    return DateTimeUtils.get_cache_key(user_type)

def validate_request_data(test_id: str, question_number: int, answer: str, 
                         user_type: str, total_questions: int) -> List[str]:
    """Validate all request data and return list of errors"""
    errors = []
    
    if not ValidationUtils.validate_test_id(test_id):
        errors.append("Invalid test ID format")
    
    if not ValidationUtils.validate_question_number(question_number, total_questions):
        errors.append("Invalid question number")
    
    if not ValidationUtils.validate_answer(answer, user_type):
        errors.append("Invalid answer format")
    
    if not ValidationUtils.validate_user_type(user_type):
        errors.append("Invalid user type")
    
    return errors