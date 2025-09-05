from tools.question_generator import question_generator_tool
from tools.search_terms_generator import search_terms_generator_tool
from tools.search_executor import execute_search_plan
from schema import ResearchContext, Question, QAItem, WebSearchPlan, ExecutedSearchPlan
from agents import Agent, Runner, trace
from typing import Union


HOW_MANY_QUESTIONS = 3


MANAGER_INSTRUCTIONS = f"""
Role:
- You are the Manager Agent.

Task:
- Guide a research process in exactly {HOW_MANY_QUESTIONS} Q&A turns, then delegate to the generate_search_terms, and finally execute the search plan.

**Ground truth context (DO NOT reconstruct it yourself):**
- RESEARCH_CONTEXT_JSON is provided below. When calling tools that require a 'context' argument, pass **exactly** this JSON string.

Operational rules:
- If Q&A turns < {HOW_MANY_QUESTIONS} → call generate_question(context=RESEARCH_CONTEXT_JSON).
- If Q&A turns == {HOW_MANY_QUESTIONS} → call generate_search_terms(context=RESEARCH_CONTEXT_JSON).
- After generate_search_terms returns a WebSearchPlan → immediately call execute_search_plan(plan).
- Never generate question text or search terms or summaries yourself; only tools produce them.
- Do not modify or re-serialize the context.

Goal:
- After exactly {HOW_MANY_QUESTIONS} clarifying Q&A turns, produce a well-scoped research plan by invoking the generate_search_terms tool, then enrich it with executed search results by invoking execute_search_plan.
"""


class ManagerAgent:
    def __init__(self):
        self.context = ResearchContext(initial_query="", qa_history=[])

        self.tools = [
            question_generator_tool,
            search_terms_generator_tool,
            execute_search_plan
        ]

        self.agent = Agent(
            name="Research Manager",
            instructions=MANAGER_INSTRUCTIONS,
            tools=self.tools,
            model="gpt-5",
            output_type=Union[Question, WebSearchPlan, ExecutedSearchPlan]
        )

    async def run(self):
        """Run the manager with the current context (does not overwrite user input)."""
        with trace("Research Manager Session"):
            result = await Runner.run(self.agent, self.context.to_input_data())
            print('[result]:', result)

        # If the result is a Question → add a QAItem (without answer yet)
        # Handle tool outputs properly
        output = result.final_output
        if isinstance(output, Question):
            self.context.qa_history.append(QAItem(question=output))
            print('[question]:', self.context.qa_history)
        elif isinstance(output, WebSearchPlan):
            print('[web search plan]:', output)
        elif isinstance(output, ExecutedSearchPlan):
            print('[executed search plan]:', output)

        return output
