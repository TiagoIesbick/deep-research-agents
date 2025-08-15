from typing import Dict, Type, List, Any
from questioner_agent import InitialQuestionTool, FollowUpQuestionTool, SummaryTool, ResearchTool
from planner_agent import WebSearchPlannerTool


class ToolRegistry:
    """Central registry for all available tools in the research system."""
    
    def __init__(self):
        self._tools: Dict[str, ResearchTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools."""
        self.register_tool(InitialQuestionTool())
        self.register_tool(FollowUpQuestionTool())
        self.register_tool(SummaryTool())
        self.register_tool(WebSearchPlannerTool())
    
    def register_tool(self, tool: ResearchTool):
        """Register a new tool."""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> ResearchTool:
        """Get a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found. Available tools: {list(self._tools.keys())}")
        return self._tools[name]
    
    def get_all_tools(self) -> Dict[str, ResearchTool]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def get_tool_names(self) -> List[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all registered tools."""
        return {name: tool.description for name, tool in self._tools.items()}
    
    def list_tools(self) -> str:
        """Get a formatted list of all available tools."""
        descriptions = self.get_tool_descriptions()
        return "\n".join([f"- {name}: {desc}" for name, desc in descriptions.items()])


# Global tool registry instance
tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return tool_registry


def register_tool(tool: ResearchTool):
    """Register a tool in the global registry."""
    tool_registry.register_tool(tool)


def get_tool(name: str) -> ResearchTool:
    """Get a tool by name from the global registry."""
    return tool_registry.get_tool(name)


def list_tools() -> str:
    """Standalone function to list all available tools."""
    return tool_registry.list_tools()
