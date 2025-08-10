from pydantic import BaseModel, Field
from agents import Agent
from typing import List, Optional

HOW_MANY_QUESTIONS = 3

# Instructions for the initial question
INITIAL_INSTRUCTIONS = """You are a helpful research assistant. Given a user's initial query, ask the first question to better understand what they really want. 
Your question should be focused and help clarify the user's intent. Be specific and ask about the most important aspect first."""

# Instructions for follow-up questions
FOLLOW_UP_INSTRUCTIONS = """You are a helpful research assistant conducting an interactive questioning session. 
Based on the user's initial query and their previous answers, ask the next question to further clarify their needs.

Context so far:
- Initial query: {initial_query}
- Previous answers: {previous_answers}

Ask the next question that will provide the most valuable insight to understand what the user really wants. 
Focus on areas that haven't been clarified yet. Be specific and build upon the information you already have."""

# Instructions for the final summary
SUMMARY_INSTRUCTIONS = """You are a helpful research assistant. Based on the user's initial query and all their answers to your questions, 
provide a clear summary of what the user actually wants. This should be a refined, specific understanding of their needs."""


class Question(BaseModel):
    question: str = Field(description="The question to ask.")
    reasoning: str = Field(description="Your reasoning for why this question is important to ask at this point.")


class QuestionPlan(BaseModel):
    questions: List[Question] = Field(description="A list of questions to ask to best answer the query.")


class UserAnswer(BaseModel):
    answer: str = Field(description="The user's answer to the question.")


class RefinedQuery(BaseModel):
    original_query: str = Field(description="The user's original query.")
    refined_understanding: str = Field(description="Your refined understanding of what the user really wants based on all the answers.")
    key_clarifications: List[str] = Field(description="Key points that were clarified through the questioning process.")


# Agent for asking the first question
initial_questioner = Agent(
    name="InitialQuestioner",
    instructions=INITIAL_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=Question,
)

# Agent for asking follow-up questions
follow_up_questioner = Agent(
    name="FollowUpQuestioner",
    instructions=FOLLOW_UP_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=Question,
)

# Agent for providing the final summary
summary_agent = Agent(
    name="SummaryAgent",
    instructions=SUMMARY_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=RefinedQuery,
)


class InteractiveQuestioner:
    """Interactive questioning agent that asks questions one by one to understand user needs."""
    
    def __init__(self, max_questions: int = HOW_MANY_QUESTIONS):
        self.max_questions = max_questions
        self.questions_asked = []
        self.user_answers = []
        self.initial_query = ""
    
    async def start_questioning(self, initial_query: str) -> Question:
        """Start the questioning process with the first question."""
        self.initial_query = initial_query
        question_data = await initial_questioner.run(initial_query)
        self.questions_asked.append(question_data)
        return question_data
    
    async def ask_follow_up(self, user_answer: str) -> Optional[Question]:
        """Ask the next follow-up question based on the user's answer."""
        self.user_answers.append(user_answer)
        
        # If we've reached the max questions, don't ask more
        if len(self.questions_asked) >= self.max_questions:
            return None
        
        # Prepare context for the follow-up question
        context = {
            "initial_query": self.initial_query,
            "previous_answers": self.user_answers
        }
        
        # Create a custom instruction with the context
        custom_instructions = FOLLOW_UP_INSTRUCTIONS.format(
            initial_query=self.initial_query,
            previous_answers="\n".join([f"Q{i+1}: {q.question}\nA{i+1}: {a}" 
                                      for i, (q, a) in enumerate(zip(self.questions_asked, self.user_answers))])
        )
        
        # Update the agent's instructions temporarily
        follow_up_questioner.instructions = custom_instructions
        
        question_data = await follow_up_questioner.run(
            f"Ask the next question based on: {user_answer}"
        )
        self.questions_asked.append(question_data)
        return question_data
    
    async def get_final_summary(self) -> RefinedQuery:
        """Get the final refined understanding of what the user wants."""
        # Prepare context for the summary
        context = {
            "initial_query": self.initial_query,
            "all_answers": self.user_answers
        }
        
        # Create a custom instruction with the context
        custom_instructions = SUMMARY_INSTRUCTIONS + f"""

Context:
- Initial query: {self.initial_query}
- All answers: {chr(10).join([f'Q{i+1}: {q.question}{chr(10)}A{i+1}: {a}' for i, (q, a) in enumerate(zip(self.questions_asked, self.user_answers))])}
"""
        
        # Update the agent's instructions temporarily
        summary_agent.instructions = custom_instructions
        
        summary = await summary_agent.run(
            f"Provide final summary based on: {self.initial_query} and answers: {self.user_answers}"
        )
        return summary
    
    def get_questioning_progress(self) -> dict:
        """Get the current progress of the questioning session."""
        return {
            "questions_asked": len(self.questions_asked),
            "answers_received": len(self.user_answers),
            "max_questions": self.max_questions,
            "is_complete": len(self.questions_asked) >= self.max_questions
        }