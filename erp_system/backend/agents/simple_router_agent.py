"""
Simple Router Agent - Intelligent Query Routing for ERP System

This module implements the main router agent that intelligently routes user queries
to appropriate specialized agents based on the request content and context.
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.tools import tool

from agents.SalesAgent import create_sales_agent_with_chat

# Import Analytics Agent (with error handling for dependencies)
try:
    from agents.AnalyticsAgent import create_analytics_agent

    ANALYTICS_AVAILABLE = True
except ImportError as exc:
    print(f"Analytics Agent import failed: {exc}")
    create_analytics_agent = None
    ANALYTICS_AVAILABLE = False

from config.llm import get_llm
from mcp.tool_registry import ToolRegistry
from memory.base_memory import RouterGlobalState

tool_registry = ToolRegistry()
global_state = RouterGlobalState()


@lru_cache(maxsize=1)
def get_sales_agent():
    return create_sales_agent_with_chat()


@lru_cache(maxsize=1)
def get_analytics_agent():
    if not ANALYTICS_AVAILABLE or create_analytics_agent is None:
        return None
    try:
        return create_analytics_agent()
    except Exception as exc:
        print(f"Analytics Agent initialization failed: {exc}")
        return None


@tool
def execute_with_sales_agent(user_request: str) -> str:
    """Route requests to the Sales Agent for customer, lead, and order management."""
    try:
        sales_agent = get_sales_agent()
        result = sales_agent.invoke({"input": user_request})
        return result["output"]
    except Exception as exc:
        return f"Sales Agent Error: {exc}"


@tool
def get_system_info() -> str:
    """Get system information and health status."""
    try:
        from db import get_db

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM customers")
            customers = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM orders")
            orders = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM leads")
            leads = cursor.fetchone()[0]

        return f"""📊 **System Status:**
• Database: Connected ✅
• Total Customers: {customers}
• Total Orders: {orders}
• Active Leads: {leads}
• System Health: Operational
"""
    except Exception as exc:
        return f"System Info Error: {exc}"


@tool
def execute_with_analytics_agent(user_request: str) -> str:
    """Route analytics, reporting, and data analysis queries to the Analytics Agent."""
    analytics_agent = get_analytics_agent()
    if analytics_agent is None:
        try:
            sales_agent = get_sales_agent()
            result = sales_agent.invoke({"input": f"Analyze and provide insights on: {user_request}"})
            return f"📊 Analytics (via Sales Agent): {result['output']}"
        except Exception as exc:
            return f"Analytics Agent Error: Analytics Agent is not available and fallback failed: {exc}"

    try:
        result = analytics_agent.invoke({"input": user_request})
        response = result["output"]
        print(f"Analytics Agent Response: {response[:200]}...")
        return response
    except Exception as exc:
        return f"Analytics Agent Error: {exc}"


tool_registry.register_tool(execute_with_sales_agent)
if ANALYTICS_AVAILABLE:
    tool_registry.register_tool(execute_with_analytics_agent)
tool_registry.register_tool(get_system_info)


def create_simple_router_agent():
    llm = get_llm()
    tools = tool_registry.get_tools()
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
    )

    prompt_template = """You are a Router Agent for Helios Dynamics ERP system.

Your job is to route user requests to the appropriate agent and return their EXACT response.

Chat History: {chat_history}

TOOLS:
------
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT INSTRUCTIONS:
1. If the user asks about customers, leads, orders, sales, or CRM - use execute_with_sales_agent
2. If the user asks about analytics, reports, revenue, metrics, SQL queries, or data analysis - use execute_with_analytics_agent
3. If the user asks about system info, health, or status - use get_system_info
4. When you get a response from a tool, return EXACTLY what the tool returned in your Final Answer
5. Do NOT add wrapper text in your Final Answer

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input", "agent_scratchpad", "chat_history"],
        partial_variables={
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools]),
        },
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
    )


class RouterAgent:
    """Router Agent with memory management."""

    def __init__(self, global_state: RouterGlobalState = None, user_id: str = "default_user", session_id: str = None):
        self.user_id = user_id
        self.session_id = session_id
        self.global_state = global_state or RouterGlobalState()
        self.conversation_id = self.global_state.get_or_create_conversation(
            user_id=user_id,
            session_id=session_id,
            agent_type="router",
        )
        self.executor = create_simple_router_agent()

    def chat(self, user_input: str) -> str:
        self.global_state.add_message(self.conversation_id, "human", user_input)

        try:
            result = self.executor.invoke({"input": user_input})
            response = result["output"]
            self.global_state.add_message(self.conversation_id, "ai", response)

            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if len(step) >= 2:
                        tool_name = step[0].tool if hasattr(step[0], "tool") else "unknown"
                        self.global_state.log_tool_call("router", tool_name, step[0], step[1])

            return response
        except Exception as exc:
            error_msg = f"Router Error: {exc}"
            self.global_state.add_message(self.conversation_id, "ai", error_msg)
            return error_msg

    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.global_state.get_conversation_history(self.conversation_id, limit)

    async def get_system_info(self, _: str) -> str:
        return get_system_info("")


executor = None


def run_simple_router_agent():
    """Interactive simple router agent."""
    print("\nSimple Router Agent Ready")
    executor = create_simple_router_agent()

    try:
        while True:
            user_input = input("Router > ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            try:
                result = executor.invoke({"input": user_input})
                print(f"\n{result['output']}\n")
            except Exception as exc:
                print(f"Error: {exc}")
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    run_simple_router_agent()
