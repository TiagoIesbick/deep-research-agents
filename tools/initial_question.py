from schema import Question, ResearchContext
from agents import Agent


# Instructions for the initial question
INITIAL_INSTRUCTIONS = """You are a helpful research assistant. Given a user's initial query, ask the first question to better understand what they really want.
Your question should be focused and help clarify the user's intent. Be specific and ask about the most important aspect first."""


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