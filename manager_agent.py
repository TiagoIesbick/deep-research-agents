from questioner_agent import InitialQuestionTool, FollowUpQuestionTool, HOW_MANY_QUESTIONS
from planner_agent import WebSearchPlannerTool, HOW_MANY_SEARCHES
from schema import ResearchContext, Question, QAItem, WebSearchPlan
from agents import Agent, Runner, trace
from typing import Union


class ManagerAgent:
    def __init__(self):
        self.context = ResearchContext(initial_query="", qa_history=[])

        self.tools = [
            InitialQuestionTool(),
            FollowUpQuestionTool(),
            WebSearchPlannerTool(),
        ]

        self.instructions = f"""
You are the Manager Agent.
Your task is to guide a research process in exactly three Q&A turns, then delegate to the planning tool.

You have three tools:

1. ask_initial_question(initial_query: str) → Question
   - Use this ONLY once, at the very beginning.
   - This produces the first clarifying question for the user.

2. ask_follow_up_question(context: ResearchContext) → Question
   - Use this exactly {HOW_MANY_QUESTIONS - 1} times, after the initial question, to further clarify scope.
   - Each follow-up must build directly on the user’s previous answer.
   - Do not repeat or rephrase earlier questions.

3. plan_web_searches(context: ResearchContext) → WebSearchPlan
   - Use this immediately after the {HOW_MANY_QUESTIONS}° user answer.
   - You must NOT propose or invent search terms yourself.
   - Instead, call this tool and allow it to generate exactly {HOW_MANY_SEARCHES} search terms.

Rules:
- You must ask exactly {HOW_MANY_QUESTIONS} questions: one initial, {HOW_MANY_QUESTIONS - 1} follow-ups.
- After the {HOW_MANY_QUESTIONS}° answer, always call `plan_web_searches`.
- Never ask more than {HOW_MANY_QUESTIONS} questions.
- Never generate search terms yourself — always use the tool.
- Stay focused and concise, ensuring each question meaningfully narrows the research topic.

Goal:
At the end of {HOW_MANY_QUESTIONS} clarifying Q&A turns, produce a well-scoped research plan by invoking the `plan_web_searches` tool.
"""

        self.agent = Agent(
            name="Research Manager",
            instructions=self.instructions,
            tools=self.tools,
            model="gpt-5",
            output_type=Union[Question, WebSearchPlan]
        )

    async def run(self):
        """Run the manager with the current context (does not overwrite user input)."""
        role_map = {"Agent": "assistant", "User": "user"}

        # Build conversation history for Runner
        input_data = [{"role": "user", "content": self.context.initial_query}]

        for qa_item in self.context.qa_history:
            # Agent's question
            input_data.append({
                "role": role_map[qa_item.question.role],
                "content": qa_item.question.question
            })
            # User's answer (if present)
            if qa_item.answer:
                input_data.append({
                    "role": role_map[qa_item.answer.role],
                    "content": qa_item.answer.answer
                })

        with trace("Research Manager Session"):
            print('[input to Runner.run]:', input_data)
            result = await Runner.run(self.agent, input_data)
            print('[result]:', result)

        # If the result is a Question → add a QAItem (without answer yet)
        # Handle tool outputs properly
        output = result.final_output
        if isinstance(output, Question):
            self.context.qa_history.append(QAItem(question=output))
            print('[question]:', self.context.qa_history)
        elif isinstance(output, WebSearchPlan):
            print('[web search plan]:', output)

        return output