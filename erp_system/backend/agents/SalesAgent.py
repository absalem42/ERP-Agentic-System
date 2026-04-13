from __future__ import annotations

from backend.memory.base_memory import RouterGlobalState, SalesEntityMemory
from backend.mcp.tool_registry import ToolRegistry
from backend.tools.sales_tools import SalesTools


class SalesAgent:
    def __init__(self, tools: SalesTools):
        self.tools = tools

    def invoke(self, payload: dict) -> dict:
        result = self.tools.handle(payload["input"])
        return {
            "output": result["message"],
            "rows": result.get("rows"),
            "approval_required": result.get("approval_required"),
        }

    def chat(self, message: str) -> str:
        return self.invoke({"input": message})["output"]


def create_sales_agent_with_chat(
    *,
    state_store: RouterGlobalState | None = None,
    entity_memory: SalesEntityMemory | None = None,
    registry: ToolRegistry | None = None,
    db_path: str | None = None,
):
    state = state_store or RouterGlobalState(db_path)
    memory = entity_memory or SalesEntityMemory(db_path)
    tool_registry = registry or ToolRegistry()
    tools = SalesTools(state, memory, tool_registry, db_path=db_path)
    return SalesAgent(tools)
