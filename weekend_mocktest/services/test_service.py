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
    """Service for managing test lifecycle with complete dummy data support"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.ai_service = get_ai_service()
        self.content_service = get_content_service()
        self.use_dummy = config.USE_DUMMY_DATA
        
        if self.use_dummy:
            logger.info("🔧 Test service using dummy data mode")
    
    async def start_test(self, user_type: str):
        """Start a new test with batch-generated questions"""
        logger.info(f"🚀 Starting {user_type} test (dummy: {self.use_dummy})")
        
        if not ValidationUtils.validate_user_type(user_type):
            raise ValueError("Invalid user type")
        
        try:
            # Check for cached questions first
            cache_key = generate_cache_key(user_type)
            cached_questions = memory_manager.get_cached_questions(cache_key)
            
            if cached_questions:
                logger.info(f"✅ Using cached questions: {len(cached_questions)} questions")
                questions = cached_questions
            else:
                # Generate new questions
                if self.use_dummy:
                    questions = await self._generate_dummy_questions(user_type)
                else:
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
                logger.info(f"✅ Generated and cached {len(questions)} questions")
            
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
            
            # Create response object with proper time limits
            time_limit = config.DEV_TIME_LIMIT if user_type == "dev" else config.NON_DEV_TIME_LIMIT
            
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
                time_limit=time_limit
            )
            
            logger.info(f"✅ Test started successfully: {test_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Test start failed: {e}")
            raise Exception(f"Test start failed: {e}")
    
    async def _generate_dummy_questions(self, user_type: str) -> List[Dict[str, Any]]:
        """Generate dummy questions when in dummy data mode"""
        logger.info(f"🔧 Generating dummy questions for {user_type}")
        
        if user_type == "dev":
            dummy_questions = [
                {
                    "question_number": 1,
                    "title": "Array Processing Algorithm",
                    "difficulty": "Easy",
                    "type": "Algorithm",
                    "question": """
<h3>Array Sum Challenge</h3>
<p>Write a function that takes an array of integers and returns the sum of all even numbers.</p>
<p><strong>Requirements:</strong></p>
<ul>
<li>Function should handle empty arrays</li>
<li>Include error handling for invalid inputs</li>
<li>Optimize for large arrays</li>
</ul>
<p><strong>Example:</strong></p>
<pre><code>Input: [1, 2, 3, 4, 5, 6]
Output: 12 (2 + 4 + 6)</code></pre>
<p>Explain your approach and time complexity.</p>
""",
                    "options": None
                },
                {
                    "question_number": 2,
                    "title": "Database Query Optimization",
                    "difficulty": "Hard",
                    "type": "System Design",
                    "question": """
<h3>User Management System</h3>
<p>Design a database schema for a user management system with roles and permissions.</p>
<p><strong>Requirements:</strong></p>
<ul>
<li>Users can have multiple roles</li>
<li>Roles can have multiple permissions</li>
<li>Support for hierarchical permissions</li>
</ul>
<p>Write optimized SQL queries to:</p>
<ol>
<li>Retrieve all users with 'admin' role</li>
<li>Find users with specific permissions</li>
<li>List all permissions for a user</li>
</ol>
<p>Explain your indexing strategy and performance considerations.</p>
""",
                    "options": None
                },
                {
                    "question_number": 3,
                    "title": "API Rate Limiting Implementation",
                    "difficulty": "Medium",
                    "type": "Practical",
                    "question": """
<h3>Rate Limiting Strategy</h3>
<p>Implement a rate limiting mechanism for an API endpoint that handles 1000+ requests per second.</p>
<p><strong>Constraints:</strong></p>
<ul>
<li>Maximum 100 requests per minute per user</li>
<li>Burst allowance of 20 requests</li>
<li>Graceful degradation under load</li>
</ul>
<p><strong>Consider:</strong></p>
<ul>
<li>Token bucket vs sliding window algorithms</li>
<li>Redis vs in-memory storage</li>
<li>Distributed system challenges</li>
</ul>
<p>Provide implementation and explain trade-offs.</p>
""",
                    "options": None
                },
                {
                    "question_number": 4,
                    "title": "Memory Leak Debugging",
                    "difficulty": "Hard",
                    "type": "Debugging",
                    "question": """
<h3>Node.js Memory Leak</h3>
<p>Debug this Node.js code that's causing memory leaks in production:</p>
<pre><code>class DataProcessor {
  constructor() {
    this.cache = new Map();
    this.listeners = [];
  }
  
  processData(data) {
    const id = data.id;
    this.cache.set(id, data);
    
    const listener = () => {
      console.log(`Processed: ${id}`);
    };
    
    this.listeners.push(listener);
    eventEmitter.on('complete', listener);
    
    return this.transformData(data);
  }
}</code></pre>
<p><strong>Tasks:</strong></p>
<ol>
<li>Identify all memory leak sources</li>
<li>Provide fixed implementation</li>
<li>Add proper resource cleanup</li>
<li>Suggest monitoring strategies</li>
</ol>
""",
                    "options": None
                },
                {
                    "question_number": 5,
                    "title": "Async Processing Pattern",
                    "difficulty": "Medium",
                    "type": "Practical",
                    "question": """
<h3>Concurrent API Processing</h3>
<p>Create an asynchronous function that processes multiple API calls concurrently with proper error handling.</p>
<p><strong>Requirements:</strong></p>
<ul>
<li>Process 50+ API endpoints simultaneously</li>
<li>Implement retry logic with exponential backoff</li>
<li>Handle partial failures gracefully</li>
<li>Set timeout for each request (5 seconds)</li>
<li>Return results as they complete</li>
</ul>
<p><strong>APIs to call:</strong></p>
<pre><code>const endpoints = [
  'https://api1.example.com/data',
  'https://api2.example.com/info',
  // ... 50 more endpoints
];</code></pre>
<p>Implement with modern JavaScript async/await patterns.</p>
""",
                    "options": None
                }
            ]
        else:
            dummy_questions = [
                {
                    "question_number": 1,
                    "title": "Software Development Lifecycle",
                    "difficulty": "Easy",
                    "type": "Conceptual",
                    "question": """
<h3>Agile Methodology</h3>
<p>Which software development methodology emphasizes iterative development, frequent customer collaboration, and adaptive planning?</p>
""",
                    "options": ["Waterfall Model", "Agile Methodology", "Spiral Model", "V-Model"]
                },
                {
                    "question_number": 2,
                    "title": "Database Design Principles",
                    "difficulty": "Medium",
                    "type": "Analytical",
                    "question": """
<h3>Database Normalization</h3>
<p>What is the primary advantage of using normalized database design over denormalized structures?</p>
""",
                    "options": [
                        "Faster query performance in all scenarios",
                        "Reduced data redundancy and improved data integrity",
                        "Simpler database structure for beginners",
                        "Better security through data encryption"
                    ]
                },
                {
                    "question_number": 3,
                    "title": "Cloud Computing Models",
                    "difficulty": "Easy",
                    "type": "Applied",
                    "question": """
<h3>Cloud Service Models</h3>
<p>Which cloud service model provides the most control over the underlying infrastructure while still offering cloud benefits?</p>
""",
                    "options": [
                        "Software as a Service (SaaS)",
                        "Platform as a Service (PaaS)",
                        "Infrastructure as a Service (IaaS)",
                        "Function as a Service (FaaS)"
                    ]
                },
                {
                    "question_number": 4,
                    "title": "AI Ethics and Bias",
                    "difficulty": "Medium",
                    "type": "Conceptual",
                    "question": """
<h3>Algorithmic Bias</h3>
<p>What is the primary concern with algorithmic bias in AI systems used for hiring and recruitment?</p>
""",
                    "options": [
                        "Slow processing speed of applications",
                        "Unfair discrimination against certain groups",
                        "High energy consumption during processing",
                        "Large storage requirements for candidate data"
                    ]
                },
                {
                    "question_number": 5,
                    "title": "Project Management Tools",
                    "difficulty": "Easy",
                    "type": "Applied",
                    "question": """
<h3>Project Visualization</h3>
<p>Which project management diagram is most effective for showing task dependencies and project timelines?</p>
""",
                    "options": [
                        "Pie Chart",
                        "Gantt Chart",
                        "Flow Chart",
                        "Organizational Chart"
                    ]
                },
                {
                    "question_number": 6,
                    "title": "Software Architecture Patterns",
                    "difficulty": "Medium",
                    "type": "Analytical",
                    "question": """
<h3>Microservices Architecture</h3>
<p>What is the main advantage of microservices architecture over monolithic architecture?</p>
""",
                    "options": [
                        "Simpler deployment process",
                        "Lower development costs",
                        "Independent scalability of services",
                        "Reduced network communication"
                    ]
                },
                {
                    "question_number": 7,
                    "title": "Data Security Principles",
                    "difficulty": "Medium",
                    "type": "Conceptual",
                    "question": """
<h3>Data Protection</h3>
<p>Which security principle ensures that data hasn't been altered during transmission or storage?</p>
""",
                    "options": [
                        "Confidentiality",
                        "Integrity",
                        "Availability",
                        "Authentication"
                    ]
                },
                {
                    "question_number": 8,
                    "title": "Machine Learning Applications",
                    "difficulty": "Easy",
                    "type": "Applied",
                    "question": """
<h3>ML Use Cases</h3>
<p>Which machine learning technique is most commonly used for email spam detection?</p>
""",
                    "options": [
                        "Linear Regression",
                        "Classification Algorithms",
                        "Clustering Analysis",
                        "Time Series Forecasting"
                    ]
                }
            ]
        
        return dummy_questions[:config.QUESTIONS_PER_TEST]
    
    async def submit_answer(self, test_id: str, question_number: int, answer: str):
        """Submit answer and get next question or complete test"""
        logger.info(f"📝 Submit answer: {test_id}, Q{question_number} (dummy: {self.use_dummy})")
        
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
                logger.info(f"🏁 Test completed: {test_id}")
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
            
            time_limit = config.DEV_TIME_LIMIT if test_data["user_type"] == "dev" else config.NON_DEV_TIME_LIMIT
            
            next_q_response = MockNextQuestion(
                question_number=next_question["question_number"],
                total_questions=next_question["total_questions"],
                question_html=next_question["question_html"],
                options=next_question.get("options"),
                time_limit=time_limit
            )
            
            response = MockResponse(
                test_completed=False,
                next_question=next_q_response
            )
            
            logger.info(f"✅ Answer submitted, next question ready: {test_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Answer submission failed: {e}")
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
        logger.info(f"🎯 Evaluating test: {test_id} (dummy: {self.use_dummy})")
        
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
            
            # Evaluate using AI service (handles dummy data internally)
            evaluation_result = self.ai_service.evaluate_test_batch(test_data["user_type"], qa_pairs)
            
            # Save results to database (handles dummy data internally)
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
            
            logger.info(f"✅ Test completed and saved: {test_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Test completion failed: {e}")
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
            
            # Save to database (handles dummy mode internally)
            self.db_manager.save_test_results(test_id, test_data_for_save, evaluation_result)
            
        except Exception as e:
            logger.error(f"❌ Failed to save test results: {e}")
            # Don't fail the entire test completion for save errors in dummy mode
            if self.use_dummy:
                logger.warning("Continuing despite save error in dummy mode")
            else:
                raise
    
    async def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get test results by ID"""
        try:
            if not ValidationUtils.validate_test_id(test_id):
                raise ValueError("Invalid test ID format")
            
            results = self.db_manager.get_test_results(test_id)
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get test results: {e}")
            raise Exception(f"Test results retrieval failed: {e}")
    
    async def get_all_tests(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all test results"""
        try:
            results = self.db_manager.get_all_test_results(limit)
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get all tests: {e}")
            raise Exception(f"Tests retrieval failed: {e}")
    
    async def get_students(self) -> List[Dict[str, Any]]:
        """Get list of students"""
        try:
            students = self.db_manager.get_student_list()
            return students
            
        except Exception as e:
            logger.error(f"❌ Failed to get students: {e}")
            raise Exception(f"Students retrieval failed: {e}")
    
    async def get_student_tests(self, student_id: str) -> List[Dict[str, Any]]:
        """Get tests for specific student"""
        try:
            tests = self.db_manager.get_student_tests(student_id)
            return tests
            
        except Exception as e:
            logger.error(f"❌ Failed to get student tests: {e}")
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
            logger.error(f"❌ Cleanup failed: {e}")
            raise Exception(f"Cleanup failed: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for test service"""
        try:
            stats = memory_manager.get_memory_stats()
            
            return {
                "status": "healthy",
                "mode": "dummy_data" if self.use_dummy else "live_data",
                "active_tests": stats["active_tests"],
                "cached_questions": stats["cached_questions"],
                "timestamp": DateTimeUtils.get_current_timestamp()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": DateTimeUtils.get_current_timestamp()
            }

# Singleton pattern for test service
_test_service = None

def get_test_service() -> TestService:
    """Get test service instance (singleton)"""
    global _test_service
    if _test_service is None:
        _test_service = TestService()
    return _test_service