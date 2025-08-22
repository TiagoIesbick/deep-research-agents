from typing import Generic, TypeVar, Type
from pydantic import BaseModel


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseTool(Generic[InputT, OutputT]):
    def __init__(self, *, input_type: Type[InputT], output_type: Type[OutputT], name: str, description: str):
        self.input_type = input_type
        self.output_type = output_type
        self.name = name
        self.description = description

    def as_tool(self):
        """Return a dict/tool spec usable by an Agent."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_type.model_json_schema(),
            "output_schema": self.output_type.model_json_schema(),
            "run": self.run_with_context,
        }

    async def run_with_context(self, input_obj: InputT, **kwargs) -> OutputT:
        """Override in subclasses."""
        raise NotImplementedError
