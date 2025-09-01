import gradio as gr
from dotenv import load_dotenv
from manager_agent import ManagerAgent
from schema import ResearchContext, Answer, Question, WebSearchPlan
import asyncio


load_dotenv(override=True)

manager = ManagerAgent()

async def agent_chat(user_message: str, chat_history: list[dict[str, str]] | None = None):
    if not chat_history:
        # First user message is the initial query
        manager.context = ResearchContext(initial_query=user_message, qa_history=[])
        chat_history = []
    else:
        # For follow-up: attach the user's message as the answer to the last QAItem
        if manager.context.qa_history and manager.context.qa_history[-1].answer is None:
            manager.context.qa_history[-1].answer = Answer(answer=user_message)

    print('[manager context]:', manager.context)

    # Run the manager agent with the updated context
    result = await manager.run()

    print('[manager result]:', result)

    # Append user message and agent response to chat history
    chat_history.append({"role": "user", "content": user_message})
    print('[chat user]:', chat_history)
    if isinstance(result, Question):
        print('[result question]:', result.question)
        chat_history.append({"role": "assistant", "content": result.question}) # agent asked a question
        print('[chat agent question]:', chat_history)
    elif isinstance(result, WebSearchPlan):
        print('[web search plan]:', result)
        searches = "\n".join([search.query for search in result.searches])
        chat_history.append({"role": "assistant", "content": searches}) # final web search plan
        print('[chat agent]:', chat_history)

    return chat_history, chat_history


# Gradio synchronous wrapper for async
def sync_agent_chat(user_message: str, chat_history: list[dict[str, str]] | None = None):
    return asyncio.run(agent_chat(user_message, chat_history))

def main():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label="Research Agent", type="messages")
        msg = gr.Textbox(label="Your message")
        state = gr.State([])

        msg.submit(sync_agent_chat, inputs=[msg, state], outputs=[chatbot, state])

    demo.launch()


if __name__ == "__main__":
    main()
