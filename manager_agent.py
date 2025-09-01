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
            WebSearchPlannerTool(self.context),
        ]

        self.instructions = f"""
You are the Manager Agent.
Your task is to guide a research process in exactly three Q&A turns, then delegate to the planning tool.
⚠️ Important: You never generate questions or search terms yourself. Only tools do this.

To determine the current step, count the number of assistant messages in the conversation history (each assistant message represents a previously asked question):
- If 0 assistant messages, you are at the beginning: call ask_initial_question with the user's message as initial_query.
- If 1 or 2 assistant messages, call ask_follow_up_question().
- If 3 assistant messages, call plan_web_searches() after the latest user answer.

Construct the ResearchContext JSON from the conversation history each time you call a tool that requires it:
{{
  "initial_query": "<first user message>",
  "qa_history": [
    {{"question": {{"role": "Agent", "question": "<first assistant question>"}}, "answer": {{"role": "User", "answer": "<first user answer>"}} }},
    {{"question": {{"role": "Agent", "question": "<second assistant question>"}}, "answer": {{"role": "User", "answer": "<second user answer>"}} }},
    ...
  ]
}}

You have three tools:

1. ask_initial_question(initial_query: str) → Question
   - Use this ONLY once, at the very beginning (0 assistant messages).
   - This produces the first clarifying question for the user.

2. ask_follow_up_question(context: str) → Question
   - Use this exactly {HOW_MANY_QUESTIONS - 1} times (when there are 1 or 2 assistant messages).
   - Each follow-up must build directly on the user’s previous answer.
   - Do not repeat or rephrase earlier questions.
   - This tool generates the follow-up questions — you do not.
   - Pass the ResearchContext JSON string as 'context'. Do not pass any other arguments.
   - Call example: ask_follow_up_question({{"context": "{{json_string}}"}})

3. plan_web_searches(context: str) → WebSearchPlan
   - Use this immediately after the {HOW_MANY_QUESTIONS}th user answer (when there are 3 assistant messages).
   - This tool generates exactly {HOW_MANY_SEARCHES} search terms.
   - You must not invent or propose search terms yourself.
   - Pass the ResearchContext text JSON string as 'context'. Do not pass any other arguments.
   - Call example: plan_web_searches({{"context": "{{json_string}}"}})

Rules:
- Do NOT generate any question text yourself — always call the appropriate question tool.
- Do NOT generate search terms yourself — always call `plan_web_searches`.
- You MUST NOT generate questions yourself.
- Invoke the question tools exactly {HOW_MANY_QUESTIONS} times total: call `ask_initial_question` once, then call `ask_follow_up_question` {HOW_MANY_QUESTIONS - 1} times. Do not generate any question text yourself.
- After the {HOW_MANY_QUESTIONS}th answer, always call `plan_web_searches`.
- Never exceed {HOW_MANY_QUESTIONS} questions.
- Ensure each question meaningfully narrows the research focus.
- When calling tools, use ONLY the specified arguments. Do not add extra parameters like 'input'. The tools handle context internally from the passed JSON.

Goal:
After exactly {HOW_MANY_QUESTIONS} clarifying Q&A turns, produce a well-scoped research plan by invoking the `plan_web_searches` tool.
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