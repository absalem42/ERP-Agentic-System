from __future__ import annotations

from backend.memory.base_memory import RouterGlobalState
from backend.mcp.tool_registry import ToolRegistry
from backend.tools.router_tools import RouterTools


class RouterAgent:
    def __init__(
        self,
        *,
        state_store: RouterGlobalState,
        registry: ToolRegistry,
        sales_agent,
        finance_agent,
        inventory_agent,
        analytics_agent,
        system_info_provider,
    ):
        self.state_store = state_store
        self.registry = registry
        self.sales_agent = sales_agent
        self.finance_agent = finance_agent
        self.inventory_agent = inventory_agent
        self.analytics_agent = analytics_agent
        self.system_info_provider = system_info_provider
        self.tools = RouterTools(state_store, registry)

    def invoke(self, payload: dict) -> dict:
        message = payload["input"]
        conversation_id = payload.get("conversation_id")
        last_module = self.state_store.get_active_module(conversation_id) if conversation_id else None
        classification = self.tools.classifier_tool(message, last_module)
        label = classification["label"]

        if label == "router":
            result = self._handle_router_query(message)
            agent_used = "router"
        elif label == "finance":
            result = self.finance_agent.invoke(payload)
            agent_used = "finance"
        elif label == "inventory":
            result = self.inventory_agent.invoke(payload)
            agent_used = "inventory"
        elif label == "analytics":
            result = self.analytics_agent.invoke(payload)
            agent_used = "analytics"
        else:
            result = self.sales_agent.invoke(payload)
            agent_used = "sales"

        if conversation_id:
            self.state_store.set_active_module(conversation_id, agent_used)

        return {
            "output": result["output"],
            "agent_used": agent_used,
            "approval_required": result.get("approval_required"),
            "chart_spec": result.get("chart_spec"),
            "rows": result.get("rows"),
        }

    def _handle_router_query(self, message: str) -> dict:
        lowered = message.lower()
        if "tool list" in lowered or "registry" in lowered:
            registry_result = self.tools.registry_tool()
            return {"output": f"Registered tools: {len(registry_result['tools'])}", "rows": registry_result["tools"]}
        if "approval" in lowered:
            approvals = self.state_store.list_approvals(status="pending")
            return {"output": f"Pending approvals: {len(approvals)}", "rows": approvals}
        return {"output": self.system_info_provider()}


def create_simple_router_agent(
    *,
    state_store: RouterGlobalState,
    registry: ToolRegistry,
    sales_agent,
    finance_agent,
    inventory_agent,
    analytics_agent,
    system_info_provider,
):
    return RouterAgent(
        state_store=state_store,
        registry=registry,
        sales_agent=sales_agent,
        finance_agent=finance_agent,
        inventory_agent=inventory_agent,
        analytics_agent=analytics_agent,
        system_info_provider=system_info_provider,
    )
