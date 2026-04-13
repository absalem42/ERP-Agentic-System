from __future__ import annotations

from typing import Any

from backend.config.llm import coerce_text, get_llm, has_llm_credentials
from backend.memory.base_memory import RouterGlobalState
from backend.mcp.tool_registry import ToolRegistry
from backend.tools.common import find_keyword


class RouterTools:
    def __init__(self, state_store: RouterGlobalState, registry: ToolRegistry):
        self.state_store = state_store
        self.registry = registry
        self.registry.register_tool(
            name="classifier_tool",
            handler=self.classifier_tool,
            description="Classify a user message into router, sales, finance, inventory, or analytics",
            input_schema={"message": "str", "last_module": "str | None"},
            module="router",
        )
        self.registry.register_tool(
            name="registry_tool",
            handler=self.registry_tool,
            description="List registered MCP-style tools",
            input_schema={"module": "str | None"},
            module="router",
        )

    def _log(self, tool_name: str, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        self.state_store.log_tool_call("router", tool_name, payload, result)
        return result

    def classifier_tool(self, message: str, last_module: str | None = None) -> dict[str, Any]:
        lowered = message.lower()
        keyword_routes = [
            ("finance", ["invoice", "payment", "ledger", "journal", "account", "finance", "anomaly"]),
            ("inventory", ["inventory", "stock", "supplier", "purchase order", "procurement", "reorder", "receipt"]),
            ("analytics", ["analytics", "report", "revenue", "trend", "metric", "kpi", "chart", "sql", "insight"]),
            ("sales", ["customer", "lead", "order", "ticket", "support", "crm"]),
            ("router", ["approval", "tool list", "system", "health", "status"]),
        ]
        for label, keywords in keyword_routes:
            if find_keyword(lowered, keywords):
                return self._log(
                    "classifier_tool",
                    {"message": message, "last_module": last_module},
                    {"label": label, "reason": "keyword"},
                )

        if has_llm_credentials():
            prompt = (
                "Classify the ERP user message into exactly one label: router, sales, finance, inventory, analytics.\n"
                "Return only the label.\n"
                f"Message: {message}\nLabel:"
            )
            response = coerce_text(get_llm().invoke(prompt)).lower()
            for label, _ in keyword_routes:
                if label in response:
                    return self._log(
                        "classifier_tool",
                        {"message": message, "last_module": last_module},
                        {"label": label, "reason": "llm"},
                    )

        label = last_module or "sales"
        return self._log(
            "classifier_tool",
            {"message": message, "last_module": last_module},
            {"label": label, "reason": "fallback"},
        )

    def registry_tool(self, module: str | None = None) -> dict[str, Any]:
        result = {"tools": self.registry.list_tools(module)}
        return self._log("registry_tool", {"module": module}, result)
