from questioner_agent import InitialQuestionTool, FollowUpQuestionTool
from planner_agent import WebSearchPlannerTool


class ManagerAgent:
    def __init__(self):
        self.initial_question_tool = InitialQuestionTool()
        self.follow_up_question_tool = FollowUpQuestionTool()
        self.web_search_planner_tool = WebSearchPlannerTool()