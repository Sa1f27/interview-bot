# weekend_mocktest/core/ai_services.py
import logging
import time
import re
import random
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
        
        logger.info(f"🤖 Generating {question_count} {user_type} questions (dummy: {self.use_dummy})")
        
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
                    "question": """<h3>Array Sum Challenge</h3>
<p>Write a function that takes an array of integers and returns the sum of all even numbers. Include error handling for invalid inputs and optimize for large arrays.</p>
<p><strong>Requirements:</strong></p>
<ul>
<li>Handle empty arrays gracefully</li>
<li>Validate input parameters</li>
<li>Optimize for performance with large datasets</li>
</ul>
<p><strong>Example:</strong></p>
<pre><code>Input: [1, 2, 3, 4, 5, 6]
Output: 12 (2 + 4 + 6)</code></pre>
<p>Explain your approach and analyze time complexity.</p>"""
                },
                {
                    "title": "Database Query Optimization",
                    "difficulty": "Hard", 
                    "type": "System Design",
                    "question": """<h3>User Management System</h3>
<p>Design a database schema for a user management system with roles and permissions. Write optimized SQL queries to retrieve users with specific permissions and explain your indexing strategy.</p>
<p><strong>Schema Requirements:</strong></p>
<ul>
<li>Users can have multiple roles</li>
<li>Roles define sets of permissions</li>
<li>Support permission inheritance</li>
</ul>
<p><strong>Query Tasks:</strong></p>
<ol>
<li>Find all users with 'admin' role</li>
<li>Get users with specific permission</li>
<li>List permissions for a user</li>
</ol>"""
                },
                {
                    "title": "API Rate Limiting",
                    "difficulty": "Medium",
                    "type": "Practical",
                    "question": """<h3>Rate Limiting Implementation</h3>
<p>Implement a rate limiting mechanism for an API endpoint. Consider different strategies (token bucket, sliding window) and explain when to use each approach.</p>
<p><strong>Requirements:</strong></p>
<ul>
<li>100 requests per minute per user</li>
<li>Burst allowance of 20 requests</li>
<li>Distributed system support</li>
</ul>
<p><strong>Consider:</strong></p>
<ul>
<li>Memory vs Redis storage</li>
<li>Performance implications</li>
<li>Failover strategies</li>
</ul>"""
                },
                {
                    "title": "Memory Management Debug",
                    "difficulty": "Hard",
                    "type": "Debugging", 
                    "question": """<h3>Memory Leak Detection</h3>
<p>Debug this Node.js code that's causing memory leaks in a production application. Identify the issues and provide an optimized solution with proper resource cleanup.</p>
<pre><code>class EventProcessor {
  constructor() {
    this.cache = new Map();
    this.handlers = [];
  }
  
  process(data) {
    const handler = () => console.log(data);
    this.handlers.push(handler);
    emitter.on('event', handler);
    this.cache.set(data.id, data);
  }
}</code></pre>
<p><strong>Tasks:</strong></p>
<ol>
<li>Identify memory leak sources</li>
<li>Implement proper cleanup</li>
<li>Add monitoring capabilities</li>
</ol>"""
                },
                {
                    "title": "Async Processing Pattern",
                    "difficulty": "Easy",
                    "type": "Practical",
                    "question": """<h3>Concurrent API Processing</h3>
<p>Create an asynchronous function that processes multiple API calls concurrently. Handle errors gracefully and implement proper timeout mechanisms.</p>
<p><strong>Requirements:</strong></p>
<ul>
<li>Process 10+ endpoints simultaneously</li>
<li>5-second timeout per request</li>
<li>Retry failed requests (max 3 attempts)</li>
<li>Return partial results on failures</li>
</ul>
<p><strong>APIs:</strong></p>
<pre><code>const endpoints = [
  'https://api1.example.com/data',
  'https://api2.example.com/info'
  // ... more endpoints
];</code></pre>"""
                }
            ]
        else:
            dev_templates = [
                {
                    "title": "Software Development Lifecycle",
                    "difficulty": "Easy",
                    "type": "Conceptual",
                    "question": "<h3>Agile Methodology</h3><p>Which methodology emphasizes iterative development and frequent customer collaboration?</p>",
                    "options": ["Waterfall", "Agile", "Spiral", "V-Model"]
                },
                {
                    "title": "Database Concepts", 
                    "difficulty": "Medium",
                    "type": "Analytical",
                    "question": "<h3>Database Normalization</h3><p>What is the primary advantage of using normalized database design?</p>",
                    "options": ["Faster queries", "Reduced data redundancy", "Simpler structure", "Better security"]
                },
                {
                    "title": "Cloud Computing Benefits",
                    "difficulty": "Easy",
                    "type": "Applied",
                    "question": "<h3>Cloud Service Models</h3><p>Which cloud service model provides the most control over the underlying infrastructure?</p>",
                    "options": ["SaaS", "PaaS", "IaaS", "FaaS"]
                },
                {
                    "title": "AI Ethics Considerations",
                    "difficulty": "Medium", 
                    "type": "Conceptual",
                    "question": "<h3>Algorithmic Bias</h3><p>What is the primary concern with algorithmic bias in AI systems?</p>",
                    "options": ["Processing speed", "Unfair discrimination", "Energy consumption", "Storage requirements"]
                },
                {
                    "title": "Project Management Tools",
                    "difficulty": "Easy",
                    "type": "Applied", 
                    "question": "<h3>Project Visualization</h3><p>Which diagram is most effective for showing project task dependencies?</p>",
                    "options": ["Pie chart", "Gantt chart", "Flow chart", "Org chart"]
                },
                {
                    "title": "Software Architecture",
                    "difficulty": "Medium",
                    "type": "Analytical",
                    "question": "<h3>Microservices vs Monolith</h3><p>What is the main advantage of microservices architecture?</p>",
                    "options": ["Simpler deployment", "Lower costs", "Independent scaling", "Less complexity"]
                },
                {
                    "title": "Data Security",
                    "difficulty": "Medium",
                    "type": "Conceptual",
                    "question": "<h3>Security Principles</h3><p>Which principle ensures data hasn't been altered during transmission?</p>",
                    "options": ["Confidentiality", "Integrity", "Availability", "Authentication"]
                },
                {
                    "title": "Machine Learning",
                    "difficulty": "Easy",
                    "type": "Applied",
                    "question": "<h3>ML Applications</h3><p>Which technique is most commonly used for email spam detection?</p>",
                    "options": ["Linear Regression", "Classification", "Clustering", "Time Series"]
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
        
        return dummy_questions
    
    def evaluate_test_batch(self, user_type: str, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all test answers in batch"""
        logger.info(f"🎯 Evaluating {len(qa_pairs)} {user_type} answers (dummy: {self.use_dummy})")
        
        if self.use_dummy:
            return self._generate_dummy_evaluation(qa_pairs, user_type)
        
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
    
    def _generate_dummy_evaluation(self, qa_pairs: List[Dict[str, Any]], user_type: str) -> Dict[str, Any]:
        """Generate dummy evaluation for testing"""
        logger.info(f"🔧 Generating dummy evaluation for {len(qa_pairs)} answers")
        
        # Generate realistic scores (70% average for better testing)
        scores = []
        feedbacks = []
        
        for i, qa in enumerate(qa_pairs, 1):
            # Simulate realistic scoring - not completely random
            score = 1 if random.random() > 0.3 else 0  # 70% correct rate
            scores.append(score)
            
            # Generate contextual feedback
            if user_type == "dev":
                if score == 1:
                    feedback = f"Question {i}: Good solution! Code demonstrates solid understanding of algorithms and best practices. Consider edge case handling for production use."
                else:
                    feedback = f"Question {i}: Implementation needs improvement. Review algorithm efficiency and error handling. Consider time complexity optimization."
            else:
                if score == 1:
                    feedback = f"Question {i}: Correct! Shows good understanding of the concept and its practical applications."
                else:
                    feedback = f"Question {i}: Incorrect. Review the fundamental concepts and their relationships. Focus on practical applications."
            
            feedbacks.append(feedback)
        
        total_correct = sum(scores)
        
        # Generate comprehensive evaluation report
        percentage = (total_correct / len(qa_pairs)) * 100
        
        if user_type == "dev":
            if percentage >= 80:
                performance = "Excellent"
                summary = "Strong coding skills with good problem-solving approach. Ready for senior-level challenges."
            elif percentage >= 60:
                performance = "Good"
                summary = "Solid foundation with room for improvement in advanced algorithms and system design."
            else:
                performance = "Needs Improvement"
                summary = "Focus on fundamental programming concepts and practice problem-solving techniques."
            
            evaluation_report = f"""DEVELOPER ASSESSMENT REPORT

OVERALL PERFORMANCE: {performance} ({percentage:.1f}%)
SCORE: {total_correct}/{len(qa_pairs)}

ANALYSIS:
{summary}

DETAILED FEEDBACK:
{chr(10).join(f"Q{i+1}: {fb}" for i, fb in enumerate(feedbacks))}

RECOMMENDATIONS:
- Practice algorithmic problem solving
- Focus on code optimization and best practices
- Study system design patterns
- Improve error handling and edge case management

This assessment evaluates practical coding skills, problem-solving ability, and technical knowledge relevant to software development roles."""
        
        else:
            if percentage >= 80:
                performance = "Excellent"
                summary = "Strong conceptual understanding with excellent analytical skills. Ready for leadership roles."
            elif percentage >= 60:
                performance = "Good"
                summary = "Good grasp of concepts with solid analytical thinking. Some areas need strengthening."
            else:
                performance = "Needs Improvement"
                summary = "Review fundamental concepts and focus on practical applications in your field."
            
            evaluation_report = f"""NON-DEVELOPER ASSESSMENT REPORT

OVERALL PERFORMANCE: {performance} ({percentage:.1f}%)
SCORE: {total_correct}/{len(qa_pairs)}

ANALYSIS:
{summary}

DETAILED FEEDBACK:
{chr(10).join(f"Q{i+1}: {fb}" for i, fb in enumerate(feedbacks))}

RECOMMENDATIONS:
- Strengthen foundational knowledge in technology concepts
- Practice analytical thinking and problem-solving
- Stay updated with industry trends and best practices
- Focus on practical applications in your domain

This assessment evaluates conceptual understanding, analytical thinking, and technology awareness for non-technical roles."""
        
        return {
            "scores": scores,
            "feedbacks": feedbacks, 
            "total_correct": total_correct,
            "evaluation_report": evaluation_report
        }
    
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
                    time.sleep(2 ** attempt)
        
        raise Exception(f"LLM call failed after {retries} attempts: {last_error}")
    
    def _parse_batch_questions(self, response: str, user_type: str) -> List[Dict[str, Any]]:
        """Parse batch-generated questions from LLM response"""
        try:
            questions = []
            
            # Split response by question markers
            question_blocks = re.split(r'=== QUESTION \d+ ===', response)[1:]
            
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
            
            question_data["question"] = "\n".join(question_lines).strip()
            
            if user_type == "non_dev":
                question_data["options"] = options if len(options) == 4 else None
            
            if not question_data["question"] or len(question_data["question"]) < 50:
                return None
            
            if user_type == "non_dev" and not question_data["options"]:
                return None
            
            return question_data
            
        except Exception as e:
            logger.error(f"Single question parsing failed: {e}")
            return None
    
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
            
            if not parsed["scores"] or len(parsed["scores"]) != len(qa_pairs):
                raise Exception("Failed to parse scores correctly")
            
            if not parsed["feedbacks"] or len(parsed["feedbacks"]) != len(qa_pairs):
                raise Exception("Failed to parse feedback correctly")
            
            for i, qa in enumerate(qa_pairs):
                qa["correct"] = bool(parsed["scores"][i])
                qa["feedback"] = parsed["feedbacks"][i]
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Evaluation parsing failed: {e}")
            raise Exception(f"Evaluation parsing failed: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check AI service health"""
        if self.use_dummy:
            return {
                "status": "healthy",
                "mode": "dummy",
                "client_ready": True,
                "message": "Running in dummy data mode - server unavailable"
            }
        
        try:
            if not self.client:
                return {"status": "error", "message": "Client not initialized"}
            
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
                    "mode": "live",
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