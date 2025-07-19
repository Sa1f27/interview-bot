"""
Prompt templates for Daily Standup application
Contains all AI prompts used throughout the system
"""

from .config import config

class Prompts:
    """Central prompt template repository"""
    
    @staticmethod
    def summary_splitting_prompt(summary: str) -> str:
        """Prompt for splitting summary into semantic chunks"""
        return f"""Split this technical summary into {config.SUMMARY_CHUNKS} meaningful, cohesive chunks for interview questions. Each chunk should focus on a specific aspect or topic.

Summary: {summary}

Return only the chunks separated by '###CHUNK###' markers. Each chunk should be 2-4 sentences covering a distinct topic."""
    
    @staticmethod
    def base_questions_prompt(chunk_content: str) -> str:
        """Prompt for generating base questions for a chunk"""
        return f"""Generate exactly {config.BASE_QUESTIONS_PER_CHUNK} insightful interview questions about this technical content. Questions should be specific, engaging, and encourage detailed responses.

Content: {chunk_content}

Format: Return only the questions, one per line, numbered 1. 2. etc."""
    
    @staticmethod
    def followup_analysis_prompt(chunk_content: str, user_response: str) -> str:
        """Prompt for analyzing user response and generating follow-ups"""
        return f"""Analyze this user response about: "{chunk_content[:100]}..."

User Response: "{user_response}"

Does this response need follow-up questions for clarity or deeper insight? If yes, generate 1-2 specific follow-up questions. If the response is complete and clear, respond with "COMPLETE".

If follow-ups needed, format as:
FOLLOWUP: Question 1
FOLLOWUP: Question 2

If complete, respond with: COMPLETE"""
    
    @staticmethod
    def greeting_responses(user_input: str, greeting_count: int) -> str:
        """Prompts for greeting stage responses"""
        prompts = [
            f"User said: '{user_input}'. Respond warmly and ask how their work is going. Keep it brief and natural (max 2 sentences).",
            f"User said: '{user_input}'. Acknowledge and transition to asking about their recent technical work. Be encouraging and brief."
        ]
        return prompts[min(greeting_count, len(prompts) - 1)]
    
    @staticmethod
    def technical_response_prompt(context: str, user_input: str, next_question: str) -> str:
        """Prompt for technical stage responses"""
        return f"""You're conducting a technical standup interview. 

Recent conversation context:
{context}

User just said: "{user_input}"

Your next planned question: "{next_question}"

Acknowledge their response briefly and naturally transition to the next question. Keep it conversational and engaging. Maximum 2 sentences + the question."""
    
    @staticmethod
    def evaluation_prompt(key_points: list) -> str:
        """Prompt for generating session evaluation"""
        points_text = "\n".join(key_points[:5])
        return f"""Evaluate this standup interview based on key points:

{points_text}

Provide brief evaluation (2-3 sentences) and score out of 10.
Format: [Evaluation] Score: X/10"""
    
    @staticmethod
    def chunk_transition_message(next_question: str) -> str:
        """Standard message for transitioning between chunks"""
        return f"Great insights! Now let me ask you about another aspect of your work: {next_question}"
    
    @staticmethod
    def session_completion_message() -> str:
        """Standard message for session completion"""
        return "Thank you for sharing all those details about your work. You've provided excellent insights into your technical progress."
    
    @staticmethod
    def final_completion_message() -> str:
        """Final message sent at session end"""
        return "Thank you for participating in today's standup! Your responses have been recorded."
    
    @staticmethod
    def conclusion_response(user_input: str) -> str:
        """Standard conclusion response"""
        return "Thank you for participating in today's standup. Your technical insights have been recorded successfully."
    
    @staticmethod
    def clarification_message() -> str:
        """Standard clarification request message"""
        return "Could you repeat that more clearly?"

# Global prompts instance
prompts = Prompts()