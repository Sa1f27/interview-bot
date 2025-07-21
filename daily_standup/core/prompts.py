"""
Dynamic Prompt templates for Daily Standup application
Generates realistic, contextual responses instead of static templates
"""

from .config import config

class DynamicPrompts:
    """Dynamic prompt repository for realistic, context-aware responses"""
    
    @staticmethod
    def summary_splitting_prompt(summary: str) -> str:
        """Dynamic prompt for splitting summary into semantic chunks"""
        return f"""You are a technical interviewer preparing questions from a project summary.

TASK: Break this technical summary into {config.SUMMARY_CHUNKS} distinct, meaningful topics for interview questions.

SUMMARY:
{summary}

REQUIREMENTS:
- Each chunk should focus on ONE specific technical aspect
- Chunks should be 2-4 sentences covering distinct topics
- Avoid overlap between chunks
- Ensure technical depth for meaningful questions

FORMAT: Return chunks separated by '###CHUNK###' markers only."""

    @staticmethod  
    def base_questions_prompt(chunk_content: str) -> str:
        """Dynamic prompt for generating contextual base questions"""
        return f"""You are an experienced technical interviewer conducting a standup session.

CONTENT TO EXPLORE:
{chunk_content}

TASK: Generate exactly {config.BASE_QUESTIONS_PER_CHUNK} insightful questions that would naturally arise in a real standup conversation about this content.

QUESTION STYLE:
- Conversational and natural (not robotic)
- Encourage detailed technical explanations
- Focus on implementation details, challenges, and decisions
- Ask about specific aspects mentioned in the content

FORMAT: Return only the questions, numbered 1. 2. etc."""

    @staticmethod
    def followup_analysis_prompt(chunk_content: str, user_response: str) -> str:
        """Dynamic prompt for analyzing responses and generating natural follow-ups"""
        return f"""You are conducting a real-time standup conversation. Analyze the user's response to determine if natural follow-up questions would help get more useful information.

TOPIC CONTEXT: "{chunk_content[:100]}..."
USER'S RESPONSE: "{user_response}"

ANALYSIS CRITERIA:
- Is the response too brief or vague?
- Did they mention something interesting that deserves exploration?
- Are there implementation details missing?
- Would a real interviewer naturally ask for clarification?

RESPONSE OPTIONS:
1. If the response is complete and clear → respond with: COMPLETE
2. If natural follow-ups would add value → generate 1-2 specific follow-up questions

FORMAT for follow-ups:
FOLLOWUP: [Natural, conversational question]
FOLLOWUP: [Another question if needed]"""

    @staticmethod
    def dynamic_greeting_response(user_input: str, greeting_count: int, context: dict = None) -> str:
        """Dynamic greeting responses that feel natural and varied"""
        conversation_history = context.get('recent_exchanges', []) if context else []
        
        return f"""You are having a natural, friendly conversation at the start of a technical standup session.

CONTEXT:
- This is greeting exchange #{greeting_count + 1}
- User just said: "{user_input}"
- Previous conversation: {conversation_history[-2:] if conversation_history else "None"}

PERSONALITY: Professional but warm, like a colleague checking in

TASK: Respond naturally to what they said, then smoothly guide toward starting the technical discussion.

VARIATION GUIDELINES:
- Don't repeat previous phrases
- Match their energy level
- Be genuinely conversational
- Avoid robotic transitions

RESPONSE STYLE: 1-2 natural sentences, max 25 words total."""

    @staticmethod
    def dynamic_technical_response(context: str, user_input: str, next_question: str, session_state: dict = None) -> str:
        """Dynamic technical responses that create natural conversation flow"""
        
        return f"""You are conducting a natural standup conversation. Create a smooth, realistic transition between the user's response and your next question.

RECENT CONVERSATION:
{context}

USER JUST SAID: "{user_input}"
YOUR NEXT QUESTION: "{next_question}"

SESSION CONTEXT:
- Questions asked so far: {session_state.get('questions_asked', 0) if session_state else 0}
- Current topic area: {session_state.get('current_topic', 'technical work') if session_state else 'technical work'}

CONVERSATION REQUIREMENTS:
1. Acknowledge their response naturally (show you listened)
2. Create smooth transition to next question
3. Vary your acknowledgment style (don't repeat "great", "interesting", etc.)
4. Sound like a real person, not a script

RESPONSE STYLE: Brief acknowledgment + natural transition + question (max 40 words)"""

    @staticmethod
    def dynamic_followup_response(current_concept_title: str, concept_content: str, 
                                 history: str, previous_question: str, user_response: str,
                                 current_question_number: int, questions_for_concept: int) -> str:
        """Dynamic follow-up generation based on fragment content and user response"""
        
        return f"""You are a supportive voice-based interviewer conducting a technical daily standup.

**Current Concept Fragment**
Title: {current_concept_title}
Content:
{concept_content}

**Conversation Context**
Last Question: {previous_question}
Student's Response: {user_response}
Recent Q&A History (Only this concept):
{history}
Question Number: {current_question_number}
Questions Asked for This Concept: {questions_for_concept}

---

**Instructions**

1. Focus only on the current concept. Do NOT bring in prior concepts or previous conversation context.

2. Response Handling Logic:
- ✅ If the response is clear or reasonably accurate (even if brief):
    - Give supportive feedback.
    - Then either:
    a. Ask the next main question for this concept
    b. OR mark UNDERSTANDING as YES to proceed to next concept
    c. OR ask one unique follow-up only if the answer is interesting

- ⚠️ If the response is short, vague, or unclear:
    - Retry **once only** using simpler phrasing.
    - If still unclear in next round, lower difficulty or move to next concept.
    - Do NOT rephrase the same question multiple times.

---

**Tone & Style**
- Keep it natural, simple, friendly
- Avoid repetition
- Use everyday English

---

**Output Format (strict)**
UNDERSTANDING: [YES | NO]
CONCEPT: [{current_concept_title}]
QUESTION: [Next question to ask]"""

    @staticmethod
    def dynamic_concept_transition(user_response: str, next_question: str, progress_info: dict) -> str:
        """Dynamic transitions between concepts in fragment system"""
        
        return f"""You're smoothly transitioning to a new concept in your standup conversation.

USER'S LAST RESPONSE: "{user_response}"
NEXT CONCEPT: "{progress_info.get('current_concept', 'next topic')}"
NEXT QUESTION: "{next_question}"

TASK: Create a natural transition that:
1. Briefly acknowledges their previous response
2. Signals a topic shift naturally  
3. Introduces the new question smoothly

TRANSITION STYLE: Sound like you're naturally moving the conversation forward, not following a script.

AVOID: "Great insights! Now let me ask about..." (too repetitive)
USE: Natural conversation patterns that real people use

RESPONSE: 1-2 sentences + question (max 35 words)"""

    @staticmethod
    def dynamic_fragment_evaluation(concepts_covered: List[str], conversation_exchanges: List[dict], 
                                   session_stats: dict) -> str:
        """Dynamic evaluation based on fragment coverage and conversation quality"""
        
        concepts_text = "\n".join([f"- {concept}" for concept in concepts_covered])
        
        # Create conversation summary
        conversation_summary = []
        for exchange in conversation_exchanges[-8:]:  # Last 8 exchanges
            q_type = " (Follow-up)" if exchange.get('is_followup') else " (Main)"
            conversation_summary.append(
                f"Concept: {exchange['concept']}{q_type}\n"
                f"Q: {exchange['ai_message'][:100]}...\n"
                f"A: {exchange['user_response'][:100]}...\n"
            )
        
        conversation_text = "\n".join(conversation_summary)
        
        return f"""You are evaluating a student's performance in a fragment-based daily standup covering multiple technical concepts.

**COVERAGE ANALYTICS:**
- Total Concepts Available: {session_stats['total_concepts']}
- Concepts Covered: {session_stats['concepts_covered']} ({session_stats['coverage_percentage']}%)
- Main Questions: {session_stats['main_questions']}
- Follow-up Questions: {session_stats['followup_questions']}
- Session Duration: {session_stats['duration_minutes']} minutes

**CONCEPTS COVERED:**
{concepts_text}

**CONVERSATION SAMPLE:**
{conversation_text}

**DETAILED QUESTION DISTRIBUTION:**
{session_stats['questions_per_concept']}

Generate a comprehensive but concise evaluation with these sections:
1. **Coverage Analysis**: How well they covered the available concepts
2. **Response Quality**: Depth and clarity of their technical explanations  
3. **Key Strengths**: 2-3 specific positive points from their responses
4. **Areas for Growth**: 1-2 constructive suggestions
5. **Final Assessment**: Overall performance summary

**SCORING CRITERIA:**
- Concept Coverage (30%): Breadth of topics discussed
- Technical Depth (25%): Quality of explanations and details
- Communication (25%): Clarity and organization of responses
- Engagement (20%): Responsiveness to questions and follow-ups

Keep under 300 words, maintain supportive tone.
Format final score as: Score: X/10"""

    @staticmethod
    def dynamic_chunk_transition(user_response: str, next_question: str, progress_info: dict) -> str:
        """Dynamic transitions between topic chunks that feel natural"""
        
        return f"""You're smoothly transitioning to a new topic in your standup conversation.

USER'S LAST RESPONSE: "{user_response}"
NEXT QUESTION: "{next_question}"
PROGRESS: Moving to topic {progress_info.get('current_chunk', 0) + 1} of {progress_info.get('total_chunks', 0)}

TASK: Create a natural transition that:
1. Briefly acknowledges their previous response
2. Signals a topic shift naturally
3. Introduces the new question smoothly

TRANSITION STYLE: Sound like you're naturally moving the conversation forward, not following a script.

AVOID: "Great insights! Now let me ask about..." (too repetitive)
USE: Natural conversation patterns that real people use

RESPONSE: 1-2 sentences + question (max 35 words)"""

    @staticmethod
    def dynamic_session_completion(conversation_summary: dict, user_final_response: str = None) -> str:
        """Dynamic session completion based on actual conversation content"""
        
        topics_discussed = conversation_summary.get('topics_covered', [])
        total_exchanges = conversation_summary.get('total_exchanges', 0)
        
        return f"""You're naturally concluding a productive standup conversation.

CONVERSATION SUMMARY:
- Topics discussed: {len(topics_discussed)} different areas
- Total exchanges: {total_exchanges}
- User's final response: "{user_final_response}" (if provided)

TASK: Create a natural, appreciative conclusion that:
1. Acknowledges the productive conversation
2. References specific value from their sharing
3. Sounds genuinely grateful, not scripted

STYLE: Warm, professional, specific to this conversation

RESPONSE: 1-2 sentences that feel like a real conversation ending"""

    @staticmethod
    def dynamic_clarification_request(context: dict) -> str:
        """Dynamic clarification requests that vary based on context"""
        
        attempts = context.get('clarification_attempts', 0)
        last_audio_quality = context.get('audio_quality', 0.5)
        
        return f"""You need to ask for clarification in a natural, varied way.

CONTEXT:
- Clarification attempts: {attempts}
- Audio quality: {last_audio_quality}

TASK: Ask for clarification in a way that:
1. Varies from previous requests
2. Stays encouraging and natural
3. Doesn't make them feel bad about unclear audio

VARIATION LEVELS:
- First time: Gentle and encouraging
- Second time: More specific about what you need
- Third time: Patient but direct

RESPONSE: One natural sentence asking for clarification"""

    @staticmethod
    def dynamic_conclusion_response(user_input: str, session_context: dict) -> str:
        """Dynamic conclusion that references the actual conversation"""
        
        return f"""You're responding to the user's final input and wrapping up naturally.

USER'S FINAL WORDS: "{user_input}"
SESSION HIGHLIGHTS: {session_context.get('key_topics', 'various technical topics')}

TASK: Create a natural closing response that:
1. Acknowledges what they just said
2. Thanks them specifically for the conversation
3. Confirms their input has value

STYLE: Natural, appreciative, like ending a real conversation with a colleague

RESPONSE: 1-2 sentences that feel genuine and conversational"""

# Global dynamic prompts instance
prompts = DynamicPrompts()