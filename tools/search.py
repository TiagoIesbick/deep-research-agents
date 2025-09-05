from agents import WebSearchTool, ModelSettings
from tools.tool_wrapper import tool_from_agent
from schema import SearchResult


SEARCH_INSTRUCTIONS = """
Role:
- You are a research assistant.

Task:
- Given a search term, you search the web for that term and produce a concise summary of the results.

Operational rules:
- The summary must 2-3 paragraphs and less than 300 words.
- Capture the main points.
- This will be consumed by someone synthesizing a report, so its vital you capture the essence and ignore any fluff.
- Do not include any additional commentary other than the summary itself.
"""


search_tool = tool_from_agent(
    agent_name="SearchAgent",
    agent_instructions=SEARCH_INSTRUCTIONS,
    output_type=SearchResult,
    tool_name="web_search",
    tool_description="Search the web for a given query and return a concise 2–3 paragraph summary.",
    tools=[WebSearchTool(search_context_size="low")],
    model_settings=ModelSettings(tool_choice="required"),
)
