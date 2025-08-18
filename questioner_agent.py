from schema import Question, ResearchContext
from agents import Agent


HOW_MANY_QUESTIONS = 3

# Instructions for the initial question
INITIAL_INSTRUCTIONS = """You are a helpful research assistant. Given a user's initial query, ask the first question to better understand what they really want.
Your question should be focused and help clarify the user's intent. Be specific and ask about the most important aspect first."""

# Instructions for follow-up questions
FOLLOW_UP_INSTRUCTIONS_TEMPLATE = """You are a helpful research assistant conducting an interactive questioning session.
Based on the user's initial query and their previous answers, ask the next question to further clarify their needs.

Context so far:
- Initial query: {initial_query}
- Previous answers: {previous_answers}

Ask the next question that will provide the most valuable insight to understand what the user really wants.
Focus on areas that haven't been clarified yet. Be specific and build upon the information you already have."""


# Tool for asking the first question
class InitialQuestionTool:
    input_type = str
    output_type = Question

    def __new__(cls):
        agent = Agent(
            name="InitialQuestioner",
            instructions=INITIAL_INSTRUCTIONS,
            model="gpt-5-mini",
            output_type=cls.output_type,
        )

        tool = agent.as_tool(
            tool_name="ask_initial_question",
            tool_description="Ask the first clarifying question to understand the user's initial query"
        )

        return tool


# Tool for asking follow-up questions
class FollowUpQuestionTool:
    input_type = ResearchContext
    output_type = Question

    def __new__(cls):
        # Create a "blank" agent with dummy instructions — we'll override per run
        agent = Agent(
            name="FollowUpQuestioner",
            instructions="Placeholder — will be overwritten at runtime",
            model="gpt-5-mini",
            output_type=cls.output_type,
        )

        tool = agent.as_tool(
            tool_name="ask_follow_up_question",
            tool_description="Ask a follow-up question based on previous context and user answers"
        )

        async def run_with_context(context: ResearchContext) -> Question:
            """
            Generate a question from richer context.
            """
            initial_query = context.initial_query
            qa_history = context.qa_history

            # Format the Q&A history
            formatted_history = "\n".join(
                [f"Q{i+1}: {item.question.question}\nA{i+1}: {item.answer.answer}"
                 for i, item in enumerate(qa_history)]
            )

            # Overwrite instructions dynamically
            agent.instructions = FOLLOW_UP_INSTRUCTIONS_TEMPLATE.format(
                initial_query=initial_query,
                previous_answers=formatted_history
            )

            return await agent.run("")

        tool.run = run_with_context
        return tool
