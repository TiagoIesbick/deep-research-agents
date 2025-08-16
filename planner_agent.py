from schema import ResearchContext, WebSearchPlan
from agents import Agent


HOW_MANY_SEARCHES = 3

# Instructions for the final summary
PLANNER_INSTRUCTIONS_TEMPLATE = f"""You are a helpful research assistant.
Based on the user's initial query and their Q&A history,
provide {HOW_MANY_SEARCHES} search terms to best answer the user's query.

Initial query:
{{initial_query}}

Q&A history:
{{qa_history}}
"""


class WebSearchPlannerTool:
    def __new__(cls):
        # Create the agent
        agent = Agent(
            name="PlannerAgent",
            instructions="Placeholder — will be overwritten at runtime",
            model="gpt-5-mini",
            output_type=WebSearchPlan,
        )

        # Create the base tool
        tool = agent.as_tool(
            tool_name="plan_web_searches",
            tool_description="Plan web searches to best answer a research query"
        )

        # Wrap run() so we can feed richer context
        async def run_with_context(context: ResearchContext) -> WebSearchPlan:
            """
            Generate a plan of web searches from richer context.
            """
            initial_query = context.initial_query
            qa_history = context.qa_history

            # Format the Q&A history
            formatted_history = "\n".join(
                [f"Q{i+1}: {item.question.question}\nA{i+1}: {item.answer}"
                 for i, item in enumerate(qa_history)]
            )

            agent.instructions = PLANNER_INSTRUCTIONS_TEMPLATE.format(
                initial_query=initial_query,
                qa_history=formatted_history
            )

            return await agent.run("")

        # Return the actual tool object from as_tool
        tool.run = run_with_context
        return tool
