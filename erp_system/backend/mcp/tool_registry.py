from __future__ import annotations

from typing import Any

from .mcp_adapter import MCPAdapter, ToolDefinition, mcp_registry


class ToolRegistry:
    def __init__(self, adapter: MCPAdapter | None = None):
        self.adapter = adapter or mcp_registry

    def register_tool(
        self,
        name: str,
        handler,
        description: str,
        input_schema: dict[str, Any] | None = None,
        *,
        module: str = "shared",
        read_only: bool = True,
        requires_approval: bool = False,
    ) -> ToolDefinition:
        return self.adapter.register_tool(
            name=name,
            handler=handler,
            description=description,
            input_schema=input_schema,
            module=module,
            read_only=read_only,
            requires_approval=requires_approval,
        )

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self.adapter.get_tool(name)

    def get_tool_info(self, name: str) -> dict[str, Any]:
        return self.adapter.get_tool_info(name)

    def list_tools(self, module: str | None = None) -> list[dict[str, Any]]:
        return self.adapter.list_tools(module)

    def clear(self) -> None:
        self.adapter.clear()
