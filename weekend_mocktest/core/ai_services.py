# weekend_mocktest/core/ai_services.py
import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq
from .config import config
from .prompts import PromptTemplates, PromptFormatter

logger = logging.getLogger(__name__)

class AIService:
    """Service for all AI operations including question generation and evaluation"""
    
    def __init__(self):
        """Initialize Groq client with validation"""
        self.client = None
        self.use_dummy = config.USE_DUMMY_DATA
        
        if not self.use_dummy:
            self._init_groq_client()
        else:
            logger.info("🔧 AI Service in dummy mode - using mock responses")
    
    def _init_groq_client(self):
        """Initialize and validate Groq client"""
        try:
            if not config.GROQ_API_KEY:
                raise Exception("GROQ_API_KEY not provided")
            
            self.client = Groq(api_key=config.GROQ_API_KEY, timeout=config.GROQ_TIMEOUT)
            
            # Test connection with a simple call
            test_completion = self.client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=10
            )
            
            if not test_completion.choices:
                raise Exception("Groq test call failed")
            
            logger.info("✅ Groq client initialized and tested")
            
        except Exception as e:
            logger.error(f"❌ Groq client initialization failed: {e}")
            raise Exception(f"AI service initialization failed: {e}")
    
    def generate_questions_batch(self, user_type: str, context: str, 
                               question_count: int = None) -> List[Dict[str, Any]]:
        """Generate all questions at once using batch processing"""
        if question_count is None:
            question_count = config.QUESTIONS_PER_TEST
        
        logger.info(f"🤖 Generating {question_count} {user_type} questions")
        
        if self.use_dummy:
            return self._generate_dummy_questions(user_type, question_count)
        
        if not self.client:
            raise Exception("AI service not available")
        
        try:
            # Format context for optimal processing
            formatted_context = PromptFormatter.format_context_for_llm(context)
            
            # Create batch generation prompt
            prompt = PromptTemplates.create_batch_questions_prompt(
                user_type, formatted_context, question_count
            )
            
            # Generate questions with retries
            response = self._call_llm_with_retries(
                prompt=prompt,
                max_tokens=config.GROQ_MAX_TOKENS,
                temperature=config.GROQ_TEMPERATURE
            )
            
            # Parse the batch response
            questions = self._parse_batch_questions(response, user_type)
            
            if len(questions) != question_count:
                logger.warning(f"Generated {len(questions)} questions, expected {question_count}")
            
            if not questions:
                raise Exception("No valid questions generated")
            
            logger.info(f"✅ Generated {len(questions)} questions successfully")
            return questions
            
        except Exception as e:
            logger.error(f"❌ Question generation failed: {e}")
            raise Exception(f"Question generation failed: {e}")
    
    def _generate_dummy_questions(self, user_type: str, question_count: int) -> List[Dict[str, Any]]:
        """Generate dummy questions for testing when server is down"""
        logger.info(f"🔧 Generating {question_count} dummy {user_type} questions")
        
        dummy_questions = []
        
        if user_type == "dev":
            dev_templates = [
                {
                    "title": "Array Processing Algorithm",
                    "difficulty": "Medium",
                    "type": "Algorithm",
                    "question": "Write a function that takes an array of integers and returns the sum of all even numbers. Include error handling for invalid inputs and optimize for large arrays. Explain your approach and time complexity."
                },
                {
                    "title": "Database Query Optimization",
                    "difficulty": "Hard", 
                    "type": "System Design",
                    "question": "Design a database schema for a user management system with roles and permissions. Write optimized SQL queries to retrieve users with specific permissions and explain your indexing strategy."
                },
                {
                    "title": "API Rate Limiting",
                    "difficulty": "Medium",
                    "type": "Practical",
                    "question": "Implement a rate limiting mechanism for an API endpoint. Consider different strategies (token bucket, sliding window) and explain when to use each approach."
                },
                {
                    "title": "Memory Management Debug",
                    "difficulty": "Hard",
                    "type": "Debugging", 
                    "question": "Debug this code that's causing memory leaks in a Node.js application. Identify the issues and provide optimized solution with proper resource cleanup."
                },
                {
                    "title": "Async Processing Pattern",
                    "difficulty": "Easy",
                    "type": "Practical",
                    "question": "Create an asynchronous function that processes multiple API calls concurrently. Handle errors gracefully and implement proper timeout mechanisms."
                }
            ]
        else:
            dev_templates = [
                {
                    "title": "Software Development Lifecycle",
                    "difficulty": "Easy",
                    "type": "Conceptual",
                    "question": "Which methodology emphasizes iterative development and frequent customer collaboration?",
                    "options": ["Waterfall", "Agile", "Spiral", "V-Model"]
                },
                {
                    "title": "Database Concepts", 
                    "difficulty": "Medium",
                    "type": "Analytical",
                    "question": "What is the primary advantage of using normalized database design?",
                    "options": ["Faster queries", "Reduced data redundancy", "Simpler structure", "Better security"]
                },
                {
                    "title": "Cloud Computing Benefits",
                    "difficulty": "Easy",
                    "type": "Applied",
                    "question": "Which cloud service model provides the most control over the underlying infrastructure?",
                    "options": ["SaaS", "PaaS", "IaaS", "FaaS"]
                },
                {
                    "title": "AI Ethics Considerations",
                    "difficulty": "Medium", 
                    "type": "Conceptual",
                    "question": "What is the primary concern with algorithmic bias in AI systems?",
                    "options": ["Processing speed", "Unfair discrimination", "Energy consumption", "Storage requirements"]
                },
                {
                    "title": "Project Management Tools",
                    "difficulty": "Easy",
                    "type": "Applied", 
                    "question": "Which diagram is most effective for showing project task dependencies?",
                    "options": ["Pie chart", "Gantt chart", "Flow chart", "Org chart"]
                }
            ]
        
        # Cycle through templates to generate required number of questions
        for i in range(question_count):
            template = dev_templates[i % len(dev_templates)]
            question = {
                "question_number": i + 1,
                "title": f"{template['title']} - Question {i + 1}",
                "difficulty": template["difficulty"],
                "type": template["type"],
                "question": template["question"],
                "options": template.get("options")
            }
            dummy_questions.append(question)
        
        logger.info(f"✅ Generated {len(dummy_questions)} dummy questions")
        return dummy_questions
    
    def _call_llm_with_retries(self, prompt: str, max_tokens: int = None, 
                              temperature: float = None, retries: int = None) -> str:
        """Call LLM with retry logic"""
        if max_tokens is None:
            max_tokens = config.GROQ_MAX_TOKENS
        if temperature is None:
            temperature = config.GROQ_TEMPERATURE
        if retries is None:
            retries = config.BATCH_GENERATION_RETRIES
        
        last_error = None
        
        for attempt in range(retries):
            try:
                logger.debug(f"LLM call attempt {attempt + 1}/{retries}")
                
                completion = self.client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    top_p=config.GROQ_TOP_P
                )
                
                if not completion.choices:
                    raise Exception("LLM returned no response")
                
                response = completion.choices[0].message.content.strip()
                
                if not response or len(response) < 100:
                    raise Exception("LLM returned insufficient content")
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"LLM call failed after {retries} attempts: {last_error}")
    
    def _parse_batch_questions(self, response: str, user_type: str) -> List[Dict[str, Any]]:
        """Parse batch-generated questions from LLM response"""
        try:
            questions = []
            
            # Split response by question markers
            question_blocks = re.split(r'=== QUESTION \d+ ===', response)[1:]  # Skip first empty part
            
            for i, block in enumerate(question_blocks, 1):
                try:
                    question_data = self._parse_single_question_block(block, user_type, i)
                    if question_data:
                        questions.append(question_data)
                except Exception as e:
                    logger.warning(f"Failed to parse question {i}: {e}")
            
            return questions
            
        except Exception as e:
            logger.error(f"❌ Batch question parsing failed: {e}")
            raise Exception(f"Question parsing failed: {e}")
    
    def _parse_single_question_block(self, block: str, user_type: str, question_number: int) -> Optional[Dict[str, Any]]:
        """Parse a single question block"""
        try:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            
            question_data = {
                "question_number": question_number,
                "title": "",
                "difficulty": "Medium",
                "type": "General",
                "question": "",
                "options": None
            }
            
            current_section = None
            question_lines = []
            options = []
            
            for line in lines:
                if line.startswith("## Title:"):
                    question_data["title"] = line.replace("## Title:", "").strip()
                elif line.startswith("## Difficulty:"):
                    question_data["difficulty"] = line.replace("## Difficulty:", "").strip()
                elif line.startswith("## Type:"):
                    question_data["type"] = line.replace("## Type:", "").strip()
                elif line.startswith("## Question:"):
                    current_section = "question"
                elif line.startswith("## Options:") and user_type == "non_dev":
                    current_section = "options"
                elif current_section == "question":
                    if not line.startswith("##"):
                        question_lines.append(line)
                elif current_section == "options" and user_type == "non_dev":
                    if re.match(r'^[A-D]\)', line):
                        option_text = line[3:].strip()
                        if option_text:
                            options.append(option_text)
            
            # Combine question lines
            question_data["question"] = "\n".join(question_lines).strip()
            
            # Set options for non-dev questions
            if user_type == "non_dev":
                question_data["options"] = options if len(options) == 4 else None
            
            # Validate question quality
            if not question_data["question"] or len(question_data["question"]) < 50:
                return None
            
            if user_type == "non_dev" and not question_data["options"]:
                return None
            
            return question_data
            
        except Exception as e:
            logger.error(f"Single question parsing failed: {e}")
            return None
    
    def evaluate_test_batch(self, user_type: str, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all test answers in batch"""
        logger.info(f"🧠 Evaluating {len(qa_pairs)} {user_type} answers")
        
        if not self.client:
            raise Exception("AI service not available")
        
        try:
            # Create evaluation prompt
            prompt = PromptTemplates.create_evaluation_prompt(user_type, qa_pairs)
            
            # Get evaluation response
            response = self._call_llm_with_retries(
                prompt=prompt,
                max_tokens=config.EVALUATION_MAX_TOKENS,
                temperature=config.EVALUATION_TEMPERATURE
            )
            
            # Parse evaluation result
            evaluation_result = self._parse_evaluation_response(response, qa_pairs)
            
            logger.info(f"✅ Evaluation completed: {evaluation_result['total_correct']}/{len(qa_pairs)}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise Exception(f"Evaluation failed: {e}")
    
    def _parse_evaluation_response(self, response: str, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse evaluation response from LLM"""
        try:
            parsed = {
                "scores": [],
                "feedbacks": [],
                "total_correct": 0,
                "evaluation_report": response
            }
            
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            for line in lines:
                if line.startswith('SCORES:'):
                    scores_str = line.replace('SCORES:', '').strip()
                    scores = []
                    for score_str in scores_str.split(','):
                        score_str = score_str.strip()
                        if score_str.isdigit():
                            scores.append(int(score_str))
                    
                    if len(scores) == len(qa_pairs):
                        parsed["scores"] = scores
                        parsed["total_correct"] = sum(scores)
                
                elif line.startswith('FEEDBACK:'):
                    feedback_str = line.replace('FEEDBACK:', '').strip()
                    feedbacks = [f.strip() for f in feedback_str.split('|')]
                    
                    if len(feedbacks) == len(qa_pairs):
                        parsed["feedbacks"] = feedbacks
            
            # Validate parsing success
            if not parsed["scores"] or len(parsed["scores"]) != len(qa_pairs):
                raise Exception("Failed to parse scores correctly")
            
            if not parsed["feedbacks"] or len(parsed["feedbacks"]) != len(qa_pairs):
                raise Exception("Failed to parse feedback correctly")
            
            # Update qa_pairs with evaluation results
            for i, qa in enumerate(qa_pairs):
                qa["correct"] = bool(parsed["scores"][i])
                qa["feedback"] = parsed["feedbacks"][i]
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Evaluation parsing failed: {e}")
            raise Exception(f"Evaluation parsing failed: {e}")
    
    def validate_questions(self, questions: List[Dict[str, Any]], user_type: str) -> Dict[str, Any]:
        """Validate generated questions for quality"""
        try:
            # Format questions for validation
            questions_text = "\n\n".join([
                f"Q{i+1}: {q['question']}" + 
                (f"\nOptions: {', '.join(q['options'])}" if q.get('options') else "")
                for i, q in enumerate(questions)
            ])
            
            prompt = PromptTemplates.validate_questions_prompt(questions_text, user_type)
            response = self._call_llm_with_retries(prompt, max_tokens=800, temperature=0.3)
            
            # Simple validation parsing
            validation_result = {
                "status": "PASS" if "VALIDATION_STATUS: PASS" in response else "FAIL",
                "quality_score": 7,  # Default
                "issues": [],
                "validated": True
            }
            
            # Extract quality score if present
            quality_match = re.search(r'QUALITY_SCORE:\s*(\d+)', response)
            if quality_match:
                validation_result["quality_score"] = int(quality_match.group(1))
            
            return validation_result
            
        except Exception as e:
            logger.warning(f"Question validation failed: {e}")
            return {"status": "UNKNOWN", "quality_score": 5, "issues": [str(e)], "validated": False}
    
    def health_check(self) -> Dict[str, Any]:
        """Check AI service health"""
        try:
            if not self.client:
                return {"status": "error", "message": "Client not initialized"}
            
            # Quick test call
            start_time = time.time()
            test_response = self.client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=5
            )
            response_time = time.time() - start_time
            
            if test_response.choices:
                return {
                    "status": "healthy",
                    "model": config.GROQ_MODEL,
                    "response_time_ms": round(response_time * 1000, 2),
                    "client_ready": True
                }
            else:
                return {"status": "error", "message": "No response from LLM"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Singleton pattern for AI service
_ai_service = None

def get_ai_service() -> AIService:
    """Get AI service instance (singleton)"""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService()
    return _ai_service

def close_ai_service():
    """Close AI service instance"""
    global _ai_service
    if _ai_service:
        _ai_service = None