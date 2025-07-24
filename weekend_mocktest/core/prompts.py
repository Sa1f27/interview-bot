# weekend_mocktest/core/prompts.py
from typing import List, Dict, Any
from .config import config

class PromptTemplates:
    """Centralized prompt template management"""
    
    @staticmethod
    def create_batch_questions_prompt(user_type: str, context: str, 
                                    question_count: int = None) -> str:
        """Create prompt for batch question generation"""
        if question_count is None:
            question_count = config.QUESTIONS_PER_TEST
        
        if user_type == "dev":
            return PromptTemplates._dev_batch_prompt(context, question_count)
        else:
            return PromptTemplates._non_dev_batch_prompt(context, question_count)
    
    @staticmethod
    def _dev_batch_prompt(context: str, question_count: int) -> str:
        """Developer questions batch generation prompt"""
        return f"""Generate {question_count} diverse programming questions based on this context. Create practical, challenging questions that test real development skills.

CONTEXT:
{context}

REQUIREMENTS:
- Generate exactly {question_count} questions
- Mix question types: practical coding, debugging, system design, algorithm challenges
- Each question should be complete and standalone
- Include clear problem statements with specific requirements
- Base questions on concepts from the provided context
- Make questions progressively challenging (easy to hard)
- Use real-world scenarios when possible

FORMAT each question as:
=== QUESTION N ===
## Title: [Brief descriptive title]
## Difficulty: [Easy/Medium/Hard]
## Type: [Practical/Algorithm/Debugging/System Design]
## Question:
[Detailed question with clear requirements, constraints, and expected approach]

Example structure to follow:
=== QUESTION 1 ===
## Title: Array Processing Challenge
## Difficulty: Easy
## Type: Practical
## Question:
Write a function that takes an array of integers and returns...
[Include specific requirements, constraints, input/output examples]

Generate all {question_count} questions now:"""

    @staticmethod
    def _non_dev_batch_prompt(context: str, question_count: int) -> str:
        """Non-developer questions batch generation prompt"""
        return f"""Generate {question_count} multiple-choice questions based on this context. Focus on conceptual understanding, analysis, and practical application of technical concepts.

CONTEXT:
{context}

REQUIREMENTS:
- Generate exactly {question_count} questions
- Each question should have exactly 4 options (A, B, C, D) with only 1 correct answer
- Mix question types: conceptual, analytical, practical application
- Test deep understanding of concepts from the context
- Include nuanced distractors based on common misconceptions
- Make questions progressively challenging
- Focus on 'why' and 'how' rather than just 'what'

FORMAT each question as:
=== QUESTION N ===
## Title: [Brief descriptive title]
## Difficulty: [Easy/Medium/Hard]
## Type: [Conceptual/Analytical/Applied]
## Question:
[Clear, specific question based on context]
## Options:
A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option]

Example structure to follow:
=== QUESTION 1 ===
## Title: Software Development Lifecycle
## Difficulty: Easy
## Type: Conceptual
## Question:
Based on the context, which approach best describes the primary benefit of implementing continuous integration in software development?
## Options:
A) It reduces the need for testing by automating deployments
B) It enables early detection of integration issues through frequent code merging
C) It eliminates the need for version control systems
D) It automatically fixes bugs without human intervention

Generate all {question_count} questions now:"""

    @staticmethod
    def create_evaluation_prompt(user_type: str, qa_pairs: List[Dict[str, Any]]) -> str:
        """Create prompt for evaluating test answers"""
        qa_text = []
        for i, qa in enumerate(qa_pairs, 1):
            question_preview = qa['question'][:200] + "..." if len(qa['question']) > 200 else qa['question']
            answer_preview = qa['answer'][:150] + "..." if len(qa['answer']) > 150 else qa['answer']
            qa_text.append(f"Q{i}: {question_preview}\nAnswer: {answer_preview}")
        
        qa_content = "\n\n".join(qa_text)
        
        if user_type == "dev":
            return PromptTemplates._dev_evaluation_prompt(qa_content, len(qa_pairs))
        else:
            return PromptTemplates._non_dev_evaluation_prompt(qa_content, len(qa_pairs))
    
    @staticmethod
    def _dev_evaluation_prompt(qa_content: str, question_count: int) -> str:
        """Developer answers evaluation prompt"""
        return f"""Evaluate this developer assessment comprehensively. Analyze code quality, problem-solving approach, technical accuracy, and best practices.

ASSESSMENT CONTENT:
{qa_content}

EVALUATION CRITERIA:
- Code correctness and functionality
- Algorithm efficiency and optimization
- Code readability and structure
- Best practices and conventions
- Problem-solving approach
- Technical explanation quality
- Edge case consideration

Provide evaluation in this EXACT format:
SCORES: [comma-separated 1s and 0s for each question - 1 for correct/good, 0 for incorrect/poor]
FEEDBACK: [detailed feedback for each question separated by |]
OVERALL_SCORE: [score out of 10]
PERFORMANCE_LEVEL: [Excellent/Good/Average/Needs Improvement]
STRENGTHS: [specific strengths observed in coding and problem-solving]
IMPROVEMENTS: [specific areas needing improvement]
RECOMMENDATIONS: [actionable recommendations for skill development]

Be thorough, specific, and constructive. Focus on technical accuracy and development best practices."""

    @staticmethod
    def _non_dev_evaluation_prompt(qa_content: str, question_count: int) -> str:
        """Non-developer answers evaluation prompt"""
        return f"""Evaluate this non-developer assessment comprehensively. Focus on conceptual understanding, analytical thinking, and practical application knowledge.

ASSESSMENT CONTENT:
{qa_content}

EVALUATION CRITERIA:
- Conceptual accuracy and understanding
- Analytical reasoning quality
- Practical application knowledge
- Technical terminology usage
- Problem-solving approach
- Business/technical alignment

Provide evaluation in this EXACT format:
SCORES: [comma-separated 1s and 0s for each question - 1 for correct, 0 for incorrect]
FEEDBACK: [detailed feedback for each question separated by |]
OVERALL_SCORE: [score out of 10]
PERFORMANCE_LEVEL: [Excellent/Good/Average/Needs Improvement]
STRENGTHS: [specific strengths observed in understanding and analysis]
IMPROVEMENTS: [specific areas needing improvement]
RECOMMENDATIONS: [actionable recommendations for learning and development]

Be thorough, specific, and constructive. Focus on conceptual understanding and practical application."""

    @staticmethod
    def create_context_enhancement_prompt(raw_context: str) -> str:
        """Create prompt to enhance context for better question generation"""
        return f"""Analyze and enhance this technical context to make it more suitable for generating diverse, challenging questions.

ORIGINAL CONTEXT:
{raw_context}

ENHANCEMENT REQUIREMENTS:
- Identify key technical concepts and themes
- Extract practical applications and use cases
- Highlight potential problem areas and challenges
- Suggest question angles: conceptual, practical, analytical
- Ensure context supports both beginner and advanced questions

Enhanced context should maintain original meaning while being more structured for question generation.

ENHANCED CONTEXT:"""

    @staticmethod
    def validate_questions_prompt(questions_text: str, user_type: str) -> str:
        """Create prompt to validate generated questions"""
        return f"""Review and validate these {user_type} questions for quality, clarity, and appropriateness.

QUESTIONS TO VALIDATE:
{questions_text}

VALIDATION CRITERIA:
- Question clarity and specificity
- Appropriate difficulty progression
- Relevance to user type ({user_type})
- Technical accuracy
- Option quality (for MCQ)
- Practical applicability

Provide validation in this format:
VALIDATION_STATUS: [PASS/FAIL]
ISSUES_FOUND: [list of specific issues if any]
QUALITY_SCORE: [1-10]
RECOMMENDATIONS: [suggestions for improvement]"""

class PromptFormatter:
    """Utility class for formatting prompts and responses"""
    
    @staticmethod
    def format_context_for_llm(context: str, max_length: int = 2000) -> str:
        """Format context for optimal LLM processing"""
        if len(context) <= max_length:
            return context
        
        # Truncate at sentence boundary if possible
        truncated = context[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        # Use the later boundary for better context preservation
        boundary = max(last_period, last_newline)
        if boundary > max_length * 0.8:  # If boundary is in last 20%
            return context[:boundary + 1]
        
        return truncated + "..."
    
    @staticmethod
    def extract_difficulty_from_question(question_text: str) -> str:
        """Extract difficulty level from question text"""
        question_lower = question_text.lower()
        
        if "## difficulty: easy" in question_lower:
            return "Easy"
        elif "## difficulty: medium" in question_lower:
            return "Medium"
        elif "## difficulty: hard" in question_lower:
            return "Hard"
        else:
            return "Medium"  # Default
    
    @staticmethod
    def extract_question_type(question_text: str) -> str:
        """Extract question type from question text"""
        question_lower = question_text.lower()
        
        type_patterns = {
            "practical": ["practical", "implementation", "coding"],
            "conceptual": ["conceptual", "understanding", "theory"],
            "analytical": ["analytical", "analysis", "evaluate"],
            "algorithm": ["algorithm", "complexity", "optimization"],
            "debugging": ["debugging", "error", "fix"],
            "system design": ["system", "design", "architecture"]
        }
        
        for question_type, patterns in type_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                return question_type.title()
        
        return "General"  # Default
    
    @staticmethod
    def clean_llm_response(response: str) -> str:
        """Clean and format LLM response for processing"""
        # Remove extra whitespace
        cleaned = ' '.join(response.split())
        
        # Fix common formatting issues
        cleaned = cleaned.replace('```', '')
        cleaned = cleaned.replace('**', '')
        
        return cleaned.strip()
    
    @staticmethod
    def validate_response_format(response: str, expected_sections: List[str]) -> Dict[str, bool]:
        """Validate if response contains expected sections"""
        validation = {}
        response_upper = response.upper()
        
        for section in expected_sections:
            validation[section] = section.upper() in response_upper
        
        validation['all_present'] = all(validation.values())
        return validation