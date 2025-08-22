from schema import Question, ResearchContext
from agents import Agent, Runner
from tools.base import BaseTool


# Instructions for follow-up questions
FOLLOW_UP_INSTRUCTIONS_TEMPLATE = """You are a helpful research assistant conducting an interactive questioning session.
Based on the user's initial query and their previous answers, ask the next question to further clarify their needs.

Context so far:
- Initial query: {initial_query}
- Previous answers: {previous_answers}

Ask the next question that will provide the most valuable insight to understand what the user really wants.
Focus on areas that haven't been clarified yet. Be specific and build upon the information you already have."""


class FollowUpQuestionTool(BaseTool[ResearchContext, Question]):
    def __init__(self):
        super().__init__(
            input_type=ResearchContext,
            output_type=Question,
            name="ask_follow_up_question",
            description="Ask a follow-up question based on previous context and user answers"
        )
        self.agent = Agent(
            name="FollowUpQuestionAgent",
            model="gpt-5",
            instructions=""  # will be set dynamically
        )

    async def run_with_context(self, context: ResearchContext, **kwargs) -> Question:
        # Format the Q&A history
        formatted_history = "\n".join(
            [f"Q{i+1}: {item.question.question}\nA{i+1}: {item.answer.answer}"
                for i, item in enumerate(context.qa_history)]
        )

        self.agent.instructions = FOLLOW_UP_INSTRUCTIONS_TEMPLATE.format(
            initial_query=context.initial_query,
            previous_answers=formatted_history
        )

        result = await Runner.run(self.agent, context)
        return result.parsed_output or result.final_output
