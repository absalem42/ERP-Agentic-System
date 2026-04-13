from __future__ import annotations

from backend.memory.base_memory import AnalyticsReportMemory, RouterGlobalState
from backend.mcp.tool_registry import ToolRegistry
from backend.tools.analytics_tools import AnalyticsTools


class AnalyticsAgent:
    def __init__(self, tools: AnalyticsTools):
        self.tools = tools

    def invoke(self, payload: dict) -> dict:
        result = self.tools.handle(payload["input"])
        return {
            "output": result["message"],
            "rows": result.get("rows"),
            "chart_spec": result.get("chart_spec"),
            "sql": result.get("sql"),
        }


def create_analytics_agent(
    *,
    state_store: RouterGlobalState | None = None,
    report_memory: AnalyticsReportMemory | None = None,
    registry: ToolRegistry | None = None,
    db_path: str | None = None,
):
    state = state_store or RouterGlobalState(db_path)
    reports = report_memory or AnalyticsReportMemory(db_path)
    tool_registry = registry or ToolRegistry()
    tools = AnalyticsTools(state, reports, tool_registry, db_path=db_path)
    return AnalyticsAgent(tools)
