# weekend_mocktest/services/test_service.py
import logging
import markdown
from typing import Dict, Any, List, Optional
from ..core.config import config
from ..core.database import get_db_manager
from ..core.ai_services import get_ai_service
from ..core.content_service import get_content_service
from ..core.utils import (
    memory_manager, generate_test_id, generate_cache_key,
    ValidationUtils, DateTimeUtils
)

logger = logging.getLogger(__name__)

class TestService:
    """Service for managing test lifecycle"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.ai_service = get_ai_service()
        self.content_service = get_content_service()
    
    async def start_test(self, user_type: str):
        """Start a new test with batch-generated questions"""
        logger.info(f"?? Starting {user_type} test")
        
        if not ValidationUtils.validate_user_type(user_type):
            raise ValueError("Invalid user type")
        
        try:
            # Check for cached questions first
            cache_key = generate_cache_key(user_type)
            cached_questions = memory_manager.get_cached_questions(cache_key)
            
            if cached_questions:
                logger.info(f"? Using cached questions: {len(cached_questions)} questions")
                questions = cached_questions
            else:
                # Generate new questions
                context = self.content_service.get_context_for_questions(user_type)
                questions_data = self.ai_service.generate_questions_batch(
                    user_type, context, config.QUESTIONS_PER_TEST
                )
                
                # Convert to simple dict format
                questions = []
                for i, q_data in enumerate(questions_data, 1):
                    question = {
                        "question_number": i,
                        "title": q_data.get("title", f"Question {i}"),
                        "difficulty": q_data.get("difficulty", "Medium"),
                        "type": q_data.get("type", "General"),
                        "question": q_data["question"],
                        "options": q_data.get("options")
                    }
                    questions.append(question)
                
                # Cache the questions
                memory_manager.cache_questions(cache_key, questions)
                logger.info(f"? Generated and cached {len(questions)} questions")
            
            # Create test
            test_id = memory_manager.create_test(user_type, questions)
            
            # Get first question
            current_question = memory_manager.get_current_question(test_id)
            if not current_question:
                raise Exception("Failed to get first question")
            
            # Convert markdown to HTML
            current_question["question_html"] = markdown.markdown(current_question["question_html"])
            
            # Get test data for response
            test_data = memory_manager.get_test(test_id)
            
            # Create mock response object
            class MockResponse:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            
            response = MockResponse(
                test_id=test_id,
                user_type=test_data["user_type"],
                question_number=current_question["question_number"],
                total_questions=current_question["total_questions"],
                question_html=current_question["question_html"],
                options=current_question.get("options"),
                time_limit=config.DEV_TIME_LIMIT if user_type == "dev" else config.NON_DEV_TIME_LIMIT
            )
            
            logger.info(f"? Test started successfully: {test_id}")
            return response
            
        except Exception as e:
            logger.error(f"? Test start failed: {e}")
            raise Exception(f"Test start failed: {e}")
    
    async def submit_answer(self, test_id: str, question_number: int, answer: str):
        """Submit answer and get next question or complete test"""
        logger.info(f"?? Submit answer: {test_id}, Q{question_number}")
        
        try:
            # Get test data
            test_data = memory_manager.get_test(test_id)
            if not test_data:
                raise ValueError("Test not found or expired")
            
            # Validate question number
            if not ValidationUtils.validate_question_number(question_number, test_data["total_questions"]):
                raise ValueError("Invalid question number")
            
            # Validate and sanitize answer
            if not ValidationUtils.validate_answer(answer, test_data["user_type"]):
                raise ValueError("Invalid answer")
            
            answer = ValidationUtils.sanitize_input(answer)
            
            # Convert MCQ answer if needed
            processed_answer = self._process_answer(answer, test_data["user_type"], test_id, question_number)
            
            # Submit answer
            success = memory_manager.submit_answer(test_id, question_number, processed_answer)
            if not success:
                raise Exception("Failed to submit answer")
            
            # Check if test is complete
            if memory_manager.is_test_complete(test_id):
                logger.info(f"?? Test completed: {test_id}")
                completion_response = await self._complete_test(test_id, test_data)
                return completion_response
            
            # Get next question
            next_question = memory_manager.get_current_question(test_id)
            if not next_question:
                raise Exception("Failed to get next question")
            
            # Convert markdown to HTML
            next_question["question_html"] = markdown.markdown(next_question["question_html"])
            
            # Create mock response objects
            class MockNextQuestion:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            
            class MockResponse:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            
            next_q_response = MockNextQuestion(
                question_number=next_question["question_number"],
                total_questions=next_question["total_questions"],
                question_html=next_question["question_html"],
                options=next_question.get("options"),
                time_limit=config.DEV_TIME_LIMIT if test_data["user_type"] == "dev" else config.NON_DEV_TIME_LIMIT
            )
            
            response = MockResponse(
                test_completed=False,
                next_question=next_q_response
            )
            
            logger.info(f"? Answer submitted, next question ready: {test_id}")
            return response
            
        except Exception as e:
            logger.error(f"? Answer submission failed: {e}")
            raise Exception(f"Answer submission failed: {e}")
    
    def _process_answer(self, answer: str, user_type: str, test_id: str, question_number: int) -> str:
        """Process answer based on user type"""
        if user_type == "non_dev":
            if answer.isdigit():
                try:
                    option_index = int(answer)
                    test_data = memory_manager.get_test(test_id)
                    questions = test_data["questions"]
                    
                    if 1 <= question_number <= len(questions):
                        question = questions[question_number - 1]
                        options = question.get("options")
                        
                        if options and 0 <= option_index < len(options):
                            return options[option_index]
                except (ValueError, IndexError):
                    pass
        
        return answer
    
    async def _complete_test(self, test_id: str, test_data: Dict[str, Any]):
        """Complete test and generate evaluation"""
        logger.info(f"?? Evaluating test: {test_id}")
        
        try:
            # Get all answers
            answers = memory_manager.get_test_answers(test_id)
            
            if not answers:
                raise Exception("No answers found for evaluation")
            
            # Prepare QA pairs for evaluation
            qa_pairs = []
            for answer_data in answers:
                qa_pairs.append({
                    "question": answer_data["question"],
                    "answer": answer_data["answer"],
                    "options": answer_data.get("options", [])
                })
            
            # Evaluate using AI service
            evaluation_result = self.ai_service.evaluate_test_batch(test_data["user_type"], qa_pairs)
            
            # Save results to database
            await self._save_test_results(test_id, test_data, evaluation_result, answers)
            
            # Clean up memory
            memory_manager.cleanup_test(test_id)
            
            # Create mock response
            class MockResponse:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            
            response = MockResponse(
                test_completed=True,
                score=evaluation_result["total_correct"],
                total_questions=test_data["total_questions"],
                analytics=evaluation_result["evaluation_report"]
            )
            
            logger.info(f"? Test completed and saved: {test_id}")
            return response
            
        except Exception as e:
            logger.error(f"? Test completion failed: {e}")
            raise Exception(f"Test completion failed: {e}")
    
    async def _save_test_results(self, test_id: str, test_data: Dict[str, Any], 
                               evaluation_result: Dict[str, Any], answers: List[Dict[str, Any]]):
        """Save test results to database"""
        try:
            # Update answers with evaluation results
            for i, answer in enumerate(answers):
                if i < len(evaluation_result.get("scores", [])):
                    answer["correct"] = bool(evaluation_result["scores"][i])
                if i < len(evaluation_result.get("feedbacks", [])):
                    answer["feedback"] = evaluation_result["feedbacks"][i]
            
            # Prepare test data for saving
            test_data_for_save = {
                "user_type": test_data["user_type"],
                "total_questions": test_data["total_questions"],
                "answers": answers
            }
            
            # Save to database
            self.db_manager.save_test_results(test_id, test_data_for_save, evaluation_result)
            
        except Exception as e:
            logger.error(f"? Failed to save test results: {e}")
    
    async def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get test results by ID"""
        try:
            if not ValidationUtils.validate_test_id(test_id):
                raise ValueError("Invalid test ID format")
            
            results = self.db_manager.get_test_results(test_id)
            return results
            
        except Exception as e:
            logger.error(f"? Failed to get test results: {e}")
            raise Exception(f"Test results retrieval failed: {e}")
    
    async def get_all_tests(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all test results"""
        try:
            results = self.db_manager.get_all_test_results(limit)
            return results
            
        except Exception as e:
            logger.error(f"? Failed to get all tests: {e}")
            raise Exception(f"Tests retrieval failed: {e}")
    
    async def get_students(self) -> List[Dict[str, Any]]:
        """Get list of students"""
        try:
            students = self.db_manager.get_student_list()
            return students
            
        except Exception as e:
            logger.error(f"? Failed to get students: {e}")
            raise Exception(f"Students retrieval failed: {e}")
    
    async def get_student_tests(self, student_id: str) -> List[Dict[str, Any]]:
        """Get tests for specific student"""
        try:
            tests = self.db_manager.get_student_tests(student_id)
            return tests
            
        except Exception as e:
            logger.error(f"? Failed to get student tests: {e}")
            raise Exception(f"Student tests retrieval failed: {e}")
    
    def cleanup_expired_tests(self) -> Dict[str, Any]:
        """Clean up expired tests and return stats"""
        try:
            memory_manager.cleanup_expired_data()
            stats = memory_manager.get_memory_stats()
            
            return {
                "message": "Cleanup completed successfully",
                "tests_cleaned": "automatic",
                "active_tests": stats["active_tests"],
                "timestamp": DateTimeUtils.get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"? Cleanup failed: {e}")
            raise Exception(f"Cleanup failed: {e}")

# Singleton pattern for test service
_test_service = None

def get_test_service() -> TestService:
    """Get test service instance (singleton)"""
    global _test_service
    if _test_service is None:
        _test_service = TestService()
    return _test_service