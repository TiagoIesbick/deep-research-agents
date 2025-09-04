from typing import Literal
from pydantic import BaseModel, Field, model_validator
import json


RoleAgent = Literal["Agent"]
RoleUser = Literal["User"]


class Question(BaseModel):
    role: RoleAgent = Field(default="Agent", description="Always 'Agent'")
    reasoning: str = Field(description="Why this question matters now.")
    question: str = Field(description="One precise clarifying question.")


class Answer(BaseModel):
    role: RoleUser = Field(default="User", description="Always 'User'")
    answer: str = Field(description="User's answer.")


class QAItem(BaseModel):
    question: Question
    answer: Answer | None = None


class ResearchContext(BaseModel):
    initial_query: str = Field(..., description="Original user query.")
    qa_history: list[QAItem] = Field(default_factory=list)

    def to_json_str(self) -> str:
        """Compact JSON string for feeding into prompts."""
        return json.dumps(self.model_dump(exclude_none=True), separators=(",", ":"))


class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_no_duplicates(self):
        qs = [s.query.strip().lower() for s in self.searches]
        if len(qs) != len(set(qs)):
            raise ValueError("Duplicate search queries are not allowed.")
        return self
