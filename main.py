import gradio as gr
from dotenv import load_dotenv
import asyncio
from research_manager import ResearchManager
from tool_registry import get_tool_registry, list_tools

load_dotenv(override=True)


def run_research_sync(query: str):
    """Synchronous wrapper for the research workflow."""
    async def run_research():
        manager = ResearchManager()
        results = []
        async for result in manager.run(query):
            results.append(result)
        return "\n".join(results)
    
    # Run the async function in the event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop, create a new one
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_research())
                return future.result()
        else:
            return asyncio.run(run_research())
    except RuntimeError:
        # Fallback for when no event loop is available
        return asyncio.run(run_research())


def list_available_tools():
    """List all available tools in the system."""
    return list_tools()


def main():
    print("Deep Research Agents - Tool-Based Architecture")
    print("=" * 50)
    
    # Show available tools
    print("\nAvailable Tools:")
    print(list_available_tools())
    
    # Create Gradio interface
    with gr.Blocks(title="Deep Research Agents") as demo:
        gr.Markdown("# 🔍 Deep Research Agents")
        gr.Markdown("AI-powered research using a tool-based architecture")
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Research Query",
                    placeholder="Enter your research question here...",
                    lines=3
                )
                run_btn = gr.Button("🚀 Start Research", variant="primary")
            
            with gr.Column():
                tools_info = gr.Markdown("**Available Tools:**\n" + list_available_tools())
        
        output = gr.Textbox(
            label="Research Results",
            lines=20,
            interactive=False
        )
        
        run_btn.click(
            fn=run_research_sync,
            inputs=[query_input],
            outputs=[output]
        )
    
    demo.launch()


if __name__ == "__main__":
    main()
