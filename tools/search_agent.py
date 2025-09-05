from agents import Agent, WebSearchTool, Runner


SEARCH_INSTRUCTIONS = """
Role:
- You are a research assistant.

Task:
- Given a query, you must use the `web_search_preview` tool to retrieve information, thenproduce a concise summary of the results.

Operational rules:
- Always call the `web_search_preview` tool when given a query.
- From the `web_search_preview` tool results, produce a concise summary of the results.
- The summary must 2-3 paragraphs and less than 300 words.
- Capture the main points.
- This will be consumed by someone synthesizing a report, so its vital you capture the essence and ignore any fluff.
- Do not include any additional commentary other than the summary itself.
"""


search_agent = Agent(
    name="SearchAgent",
    instructions=SEARCH_INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="low")],
    model="gpt-5-mini",
    output_type=str
)

async def run_search(query: str) -> str:
    """Run a single search with controlled summarization."""
    result = await Runner.run(search_agent, f"query: {query}")
    return result.final_output
