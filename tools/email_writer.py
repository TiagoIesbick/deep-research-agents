from tools.tool_wrapper import tool_from_agent
from tools.send_email import send_email


EMAIL_WRITER_INSTRUCTIONS = """
You are able to send a nicely formatted HTML email based on a detailed report.
You will be provided with a detailed report. You should use your tool to send one email, providing the
report converted into clean, well presented HTML with an appropriate subject line.
"""

email_writer_tool = tool_from_agent(
    agent_name="EmailWriter",
    instructions=EMAIL_WRITER_INSTRUCTIONS,
    tools=[send_email],
    model="gpt-5-mini",
)
