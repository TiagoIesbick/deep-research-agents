from pydantic import BaseModel, Field
from typing import List


class Question(BaseModel):
    reasoning: str = Field(description="Your reasoning for why this question is important to ask at this point.")
    question: str = Field(description="The question to ask.")


class QAItem(BaseModel):
    question: Question
    answer: str = Field(description="The user's answer to the corresponding question.")


class ResearchContext(BaseModel):
    initial_query: str = Field(..., description="The user's original query.")
    qa_history: List[QAItem] = Field(default_factory=list, description="List of asked questions and their answers.")


class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: List[WebSearchItem] = Field(default_factory=list, description="A list of web searches to perform to best answer the query.")
