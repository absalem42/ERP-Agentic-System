from __future__ import annotations

from backend.memory.base_memory import RouterGlobalState
from backend.mcp.tool_registry import ToolRegistry
from backend.tools.finance_tools import FinanceTools


class FinanceAgent:
    def __init__(self, tools: FinanceTools):
        self.tools = tools

    def invoke(self, payload: dict) -> dict:
        result = self.tools.handle(payload["input"], requested_by=str(payload.get("user_id", "system")))
        return {
            "output": result["message"],
            "approval_required": result.get("approval_required"),
        }


def create_finance_agent(
    *,
    state_store: RouterGlobalState | None = None,
    registry: ToolRegistry | None = None,
    db_path: str | None = None,
):
    state = state_store or RouterGlobalState(db_path)
    tool_registry = registry or ToolRegistry()
    tools = FinanceTools(state, tool_registry, db_path=db_path)
    return FinanceAgent(tools)
