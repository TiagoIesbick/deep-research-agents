from tkinter import E
from tools.search import search_tool
from tools.tool_wrapper import tool_from_agent
from schema import ExecutedSearchPlan


SEARCH_EXECUTOR_INSTRUCTIONS = """
Role:
- You are the Search Executor.

Input:
- the user input will be a WebSearchPlan (list of queries).

Task:
- Run the search tool for each query in parallel.
- Return an ExecutedSearchPlan containing each query and its summary.
"""

search_executor_tool = tool_from_agent(
    agent_name="SearchExecutor",
    agent_instructions=SEARCH_EXECUTOR_INSTRUCTIONS,
    output_type=ExecutedSearchPlan,
    tool_name="execute_search",
    tool_description="Run the search tool for each query in parallel.",
    tools=[search_tool]
)
