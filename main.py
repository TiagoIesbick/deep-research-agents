import gradio as gr
from dotenv import load_dotenv
from manager_agent import ManagerAgent
from schema import ResearchContext
import asyncio


load_dotenv(override=True)

manager = ManagerAgent()

async def agent_chat(user_message, chat_history=None):
    if chat_history is None:
        chat_history = []
        manager.context = ResearchContext(initial_query=user_message, qa_history=[])
    else:
        manager.context.initial_query = manager.context.initial_query or user_message

    # Update initial query only on first message
    if not manager.context.initial_query:
        manager.context.initial_query = user_message

    # Let the agent reason, call tools, and update qa_history
    result = await manager.run(manager.context.initial_query)

    # Append messages to chat history
    chat_history.append(("User", user_message))
    chat_history.append(("Agent", str(result)))

    return chat_history, chat_history


# Gradio synchronous wrapper for async
def sync_agent_chat(user_message, chat_history=None):
    return asyncio.run(agent_chat(user_message, chat_history))

def main():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label="Research Agent")
        msg = gr.Textbox(label="Your message")
        state = gr.State([])

        msg.submit(sync_agent_chat, inputs=[msg, state], outputs=[chatbot, state])

    demo.launch()


if __name__ == "__main__":
    main()
