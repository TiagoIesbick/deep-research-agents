from pydantic import BaseModel, Field


class Question(BaseModel):
    role: str = Field(default="Agent", description="Always 'Agent' for AI questions")
    reasoning: str = Field(description="Your reasoning for why this question is important to ask at this point.")
    question: str = Field(description="The question to ask.")


class Answer(BaseModel):
    role: str = Field(default="User", description="Always 'User' for answers")
    answer: str = Field(description="The user's answer to the corresponding question.")


class QAItem(BaseModel):
    question: Question
    answer: Answer | None = None


class ResearchContext(BaseModel):
    initial_query: str = Field(..., description="The user's original query.")
    qa_history: list[QAItem] = Field(default_factory=list, description="List of asked questions and their answers.")

    def to_json_str(self) -> str:
        """Compact JSON string for feeding into prompts."""
        return self.model_dump_json(indent=2, exclude_none=True)

    def to_text_summary(self) -> str:
        """Readable text version for LLM reasoning."""
        summary = [f"Initial query: {self.initial_query}"]
        for i, item in enumerate(self.qa_history, start=1):
            summary.append(f"Q{i}: {item.question.question}")
            if item.answer:
                summary.append(f"A{i}: {item.answer.answer}")
        return "\n".join(summary)


class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(default_factory=list, description="A list of web searches to perform to best answer the query.")
