from typing import Dict, Any, Callable, List
import json
import sys
from datetime import datetime


def safe_print(message: str) -> None:
    """Print safely even when stdout does not support Unicode."""
    stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        print(message)
    except UnicodeEncodeError:
        sanitized = message.encode(encoding, errors="replace").decode(encoding)
        print(sanitized)

class MCPAdapter:
    """Basic in-process MCP adapter for tool registration and execution"""
    
    def __init__(self):
        self.tools = {}
        self.call_logs = []
    
    def register_tool(self, name: str, func: Callable, description: str, parameters: Dict = None):
        """Register a tool with the MCP system"""
        self.tools[name] = {
            'function': func,
            'description': description,
            'parameters': parameters or {},
            'registered_at': datetime.now().isoformat()
        }
        safe_print(f"Registered tool: {name}")
    
    def list_tools(self) -> List[str]:
        """List all registered tools"""
        return list(self.tools.keys())
    
    def get_tool_info(self, name: str) -> Dict:
        """Get information about a specific tool"""
        if name in self.tools:
            tool = self.tools[name].copy()
            tool.pop('function')  # Don't return the actual function
            return tool
        return {"error": f"Tool '{name}' not found"}
    
    def call_tool(self, name: str, *args, **kwargs) -> Any:
        """Execute a registered tool"""
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found"}
        
        try:
            result = self.tools[name]['function'](*args, **kwargs)
            
            # Log the call
            self.call_logs.append({
                'tool': name,
                'timestamp': datetime.now().isoformat(),
                'args': str(args),
                'kwargs': str(kwargs),
                'success': True
            })
            
            return result
        except Exception as e:
            # Log the error
            self.call_logs.append({
                'tool': name,
                'timestamp': datetime.now().isoformat(),
                'args': str(args),
                'kwargs': str(kwargs),
                'success': False,
                'error': str(e)
            })
            
            return {"error": str(e)}
    
    def get_call_logs(self, limit: int = 10) -> List[Dict]:
        """Get recent tool call logs"""
        return self.call_logs[-limit:]
    
    def clear_logs(self):
        """Clear call logs"""
        self.call_logs.clear()

# Global MCP instance
mcp_registry = MCPAdapter()
