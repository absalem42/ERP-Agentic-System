from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable


ToolHandler = Callable[..., Any]


@dataclass
class ToolDefinition:
    name: str
    module: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    read_only: bool = True
    requires_approval: bool = False
    handler: ToolHandler | None = None

    def call(self, *args: Any, **kwargs: Any) -> Any:
        if self.handler is None:
            raise RuntimeError(f"Tool {self.name} has no handler")
        return self.handler(*args, **kwargs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "description": self.description,
            "input_schema": self.input_schema,
            "read_only": self.read_only,
            "requires_approval": self.requires_approval,
        }


class MCPAdapter:
    def __init__(self):
        self.tools: dict[str, ToolDefinition] = {}

    def register_tool(
        self,
        name: str,
        handler: ToolHandler,
        description: str,
        input_schema: dict[str, Any] | None = None,
        *,
        module: str = "shared",
        read_only: bool = True,
        requires_approval: bool = False,
    ) -> ToolDefinition:
        tool = ToolDefinition(
            name=name,
            module=module,
            description=description,
            input_schema=input_schema or {},
            read_only=read_only,
            requires_approval=requires_approval,
            handler=handler,
        )
        self.tools[name] = tool
        print(f"Registered tool: {name}", file=sys.stdout)
        return tool

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self.tools.get(name)

    def list_tools(self, module: str | None = None) -> list[dict[str, Any]]:
        tools = self.tools.values()
        if module:
            tools = [tool for tool in tools if tool.module == module]
        return [tool.as_dict() for tool in tools]

    def get_tool_info(self, name: str) -> dict[str, Any]:
        tool = self.get_tool(name)
        if tool is None:
            return {"error": f"Tool '{name}' not found"}
        return tool.as_dict()

    def call_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        tool = self.get_tool(name)
        if tool is None:
            return {"error": f"Tool '{name}' not found"}
        return tool.call(*args, **kwargs)

    def clear(self) -> None:
        self.tools.clear()


mcp_registry = MCPAdapter()
