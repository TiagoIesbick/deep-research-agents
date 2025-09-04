from tools.question_generator import question_generator_tool
from tools.search_terms import search_terms_generator_tool
from schema import ResearchContext, Question, QAItem, WebSearchPlan
from agents import Agent, Runner, trace
from typing import Union


HOW_MANY_QUESTIONS = 3


MANAGER_INSTRUCTIONS = f"""
Role:
- You are the Manager Agent.

Task:
- Guide a research process in exactly {HOW_MANY_QUESTIONS} Q&A turns, then delegate to the generate_search_terms tool.

**Ground truth context (DO NOT reconstruct it yourself):**
- RESEARCH_CONTEXT_JSON is provided below. When calling tools that require a 'context' argument, pass **exactly** this JSON string.

Operational rules:
- If Q&A turns < {HOW_MANY_QUESTIONS} → call generate_question(context=RESEARCH_CONTEXT_JSON).
- If Q&A turns == {HOW_MANY_QUESTIONS} → call generate_search_terms(context=RESEARCH_CONTEXT_JSON).
- Never generate question text or search terms yourself; only tools produce them.
- Do not modify or re-serialize the context.

Goal:
- After exactly {HOW_MANY_QUESTIONS} clarifying Q&A turns, produce a well-scoped research plan by invoking the generate_search_terms tool.
"""


class ManagerAgent:
    def __init__(self):
        self.context = ResearchContext(initial_query="", qa_history=[])

        self.tools = [
            question_generator_tool,
            search_terms_generator_tool
        ]

        self.agent = Agent(
            name="Research Manager",
            instructions=MANAGER_INSTRUCTIONS,
            tools=self.tools,
            model="gpt-5",
            output_type=Union[Question, WebSearchPlan]
        )

    async def run(self):
        """Run the manager with the current context (does not overwrite user input)."""
        role_map = {"Agent": "assistant", "User": "user"}

        # Build the conversation transcript (for the agent's awareness)
        transcript = [{"role": "user", "content": self.context.initial_query}]

        for qa in self.context.qa_history:
            transcript.append({
                "role": role_map[qa.question.role],
                "content": qa.question.question
            })
            if qa.answer:
                transcript.append({
                    "role": role_map[qa.answer.role],
                    "content": qa.answer.answer
                })

        # Inject the canonical context string
        ctx_json = self.context.to_json_str()
        system_ctx = f"RESEARCH_CONTEXT_JSON:\n```json\n{ctx_json}\n```"

        with trace("Research Manager Session"):
            input_data = [{"role": "system", "content": system_ctx}] + transcript
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
