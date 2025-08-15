from agents import Runner, trace, gen_trace_id
from questioner_agent import InteractiveQuestioner, Question, RefinedQuery
from planner_agent import WebSearchPlannerTool, WebSearchPlan
import asyncio


class ResearchManager:
    """Manages the complete research workflow using tools."""
    
    def __init__(self):
        self.questioner = InteractiveQuestioner()
        self.search_planner = WebSearchPlannerTool()
    
    async def run(self, query: str):
        """Run the complete research workflow."""
        trace_id = gen_trace_id()
        with trace("Deep Research", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            
            print("Starting research...")
            yield "Starting research..."
            
            # Step 1: Interactive questioning to understand the query
            yield "Step 1: Understanding your query through questions..."
            initial_question = await self.questioner.start_questioning(query)
            yield f"Question 1: {initial_question.question}\nReasoning: {initial_question.reasoning}"
            
            # Simulate user answers for demonstration
            # In a real app, you'd get these from user input
            sample_answers = [
                "I need comprehensive information about the latest developments",
                "Focus on practical applications and real-world examples",
                "Include both technical details and business implications"
            ]
            
            # Ask follow-up questions
            for i, answer in enumerate(sample_answers):
                if i < 2:  # Only ask 2 follow-ups for demo
                    follow_up = await self.questioner.ask_follow_up(answer)
                    if follow_up:
                        yield f"Question {i+2}: {follow_up.question}\nReasoning: {follow_up.reasoning}"
                        yield f"Your answer: {answer}"
            
            # Get final refined understanding
            yield "Generating refined understanding..."
            refined_query = await self.questioner.get_final_summary()
            yield f"Refined Understanding: {refined_query.refined_understanding}"
            yield f"Key Clarifications: {', '.join(refined_query.key_clarifications)}"
            
            # Step 2: Plan web searches based on refined understanding
            yield "Step 2: Planning web searches..."
            search_plan = await self.search_planner.run(refined_query.refined_understanding)
            yield "Web Search Plan:"
            for i, search in enumerate(search_plan.searches):
                yield f"  Search {i+1}: {search.query}\n  Reason: {search.reason}"
            
            # Step 3: Execute searches (placeholder for now)
            yield "Step 3: Executing web searches..."
            yield "This would involve actual web scraping/search API calls"
            
            # Step 4: Synthesize results (placeholder for now)
            yield "Step 4: Synthesizing research results..."
            yield "This would involve analyzing and combining all gathered information"
            
            yield "Research workflow completed!"
            print("Research completed!")