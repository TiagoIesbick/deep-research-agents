from questioner_agent import InitialQuestionTool, FollowUpQuestionTool, HOW_MANY_QUESTIONS
from planner_agent import WebSearchPlannerTool
from schema import ResearchContext, Question, QAItem
from agents import Agent, Runner, trace


class ManagerAgent:
    def __init__(self):
        self.context = ResearchContext(initial_query="", qa_history=[])

        self.tools = [
            InitialQuestionTool(),
            FollowUpQuestionTool(),
            WebSearchPlannerTool(),
        ]

        self.instructions = f"""
You are the Manager Agent responsible for clarifying a user's request
and producing the best possible set of web searches.

Workflow:

1. INITIAL QUESTION
   - Call 'ask_initial_question' with the initial query from context.
   - Wait for the answer, then store in context as {{question, answer}}.

2. FOLLOW-UP QUESTIONS
   - Call 'ask_follow_up_question' with the full context until
     you have {HOW_MANY_QUESTIONS} total Q&A entries (including the first).
   - After each answer, update context.

3. PLAN WEB SEARCHES
   - Call 'plan_web_searches' with the full context:
        initial_query: context.initial_query
        qa_history: context.qa_history
   - Return ONLY the output from 'plan_web_searches'.

Rules:
- Never invent your own questions — only use the tools.
- Always update the context in order.
- Always pass the *entire* current context to 'ask_follow_up_question' and 'plan_web_searches'.
"""

        self.agent = Agent(
            name="Research Manager",
            instructions=self.instructions,
            tools=self.tools,
            model="gpt-5"
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
        if isinstance(result, Question):
            print('[question]:', result.final_output)
            self.context.qa_history.append(QAItem(question=result))

        return result