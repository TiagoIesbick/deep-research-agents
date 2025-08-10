from agents import Runner, trace, gen_trace_id
import asyncio


class ResearchManager:
    async def run(self, query: str):
        trace_id = gen_trace_id()
        with trace("Deep Research", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            print("Starting research...")