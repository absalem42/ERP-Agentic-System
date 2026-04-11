"""
Simple Router Agent - Intelligent Query Routing for ERP System

This module implements the main router agent that intelligently routes user queries
to appropriate specialized agents based on the request content and context.

Key Features:
- Natural language understanding for query classification
- Intelligent routing to specialized agents (Sales, Finance, Inventory)
- System status monitoring and health checks
- Error handling and fallback responses
- Memory management for conversation context

Architecture:
- Uses LangChain ReAct pattern for reasoning and action selection
- Integrates with MCP (Model Context Protocol) for tool management
- Supports multiple LLM providers (Google Gemini, Ollama)
- Implements conversation memory for context awareness
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.memory import ConversationBufferWindowMemory
from config.llm import get_llm
from mcp.tool_registry import ToolRegistry
from agents.SalesAgent import create_sales_agent_with_chat
# Import Analytics Agent (with error handling for dependencies)
try:
    from agents.AnalyticsAgent import create_analytics_agent
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Analytics Agent import failed: {e}")
    create_analytics_agent = None
    ANALYTICS_AVAILABLE = False
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
    """
    Route requests to the Sales Agent for customer, lead, and order management.
    
    This tool handles all customer-related queries including:
    - Customer information retrieval and management
    - Lead tracking and scoring
    - Order history and analysis
    - Revenue reporting and analytics
    
    Args:
        user_request (str): The user's natural language request
        
    Returns:
        str: Formatted response from the Sales Agent with business data
        
    Example queries:
        - "show recent customers"
        - "display leads with high scores"
        - "customer summary report"
    """
    try:
        sales_agent = get_sales_agent()
        result = sales_agent.invoke({"input": user_request})
        response = result['output']
        return response
    except Exception as e:
        return f"Sales Agent Error: {str(e)}"

@tool
def get_system_info() -> str:
    """Get system information and health status"""
    try:
        from db import get_db
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get basic counts
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
    except Exception as e:
        return f"System Info Error: {str(e)}"

@tool
def execute_with_analytics_agent(user_request: str) -> str:
    """
    Route analytics, reporting, and data analysis queries to the Analytics Agent
    
    Use this for:
    - SQL queries and data analysis  
    - Revenue/financial reporting
    - Business metrics and KPIs
    - Data visualization requests
    - Executive dashboard queries
    - Natural language to SQL conversion
    
    Args:
        user_request (str): The user's analytics or reporting request
        
    Returns:
        str: Formatted response from the Analytics Agent with data insights
        
    Example queries:
        - "What's our total revenue this month?"
        - "Show me top performing customers"
        - "Analyze sales trends"
        - "Create a chart of monthly orders"
    """
    analytics_agent = get_analytics_agent()
    if analytics_agent is None:
        try:
            sales_agent = get_sales_agent()
            result = sales_agent.invoke({"input": f"Analyze and provide insights on: {user_request}"})
            response = result['output']
            return f"📊 Analytics (via Sales Agent): {response}"
        except Exception as e:
            return f"Analytics Agent Error: Analytics Agent is not available and fallback failed: {str(e)}"
    
    try:
        # Invoke the analytics agent with the user request
        result = analytics_agent.invoke({"input": user_request})
        response = result['output']
        print(f"Analytics Agent Response: {response[:200]}...")  # Log first 200 chars for debugging
        return response
    except Exception as e:
        return f"Analytics Agent Error: {str(e)}"

# Register tools with the registry
tool_registry.register_tool(execute_with_sales_agent)
if ANALYTICS_AVAILABLE:
    tool_registry.register_tool(execute_with_analytics_agent)
tool_registry.register_tool(get_system_info)

# Create the router agent
def create_simple_router_agent():
    llm = get_llm()
    
    # Get tools from registry
    tools = tool_registry.get_tools()
    
    # Initialize memory with k=5 as per specifications
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )
    
    # ReAct prompt template with required variables
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
5. Do NOT add "Agent Response:" or any wrapper text in your Final Answer
6. Do NOT say "Here is the information you requested" - just return the actual data

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input", "agent_scratchpad", "chat_history"],
        partial_variables={
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools])
        }
    )
    
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return executor

class RouterAgent:
    """Router Agent with full memory management"""
    
    def __init__(self, global_state: RouterGlobalState = None, user_id: str = "default_user", session_id: str = None):
        self.user_id = user_id
        self.session_id = session_id
        self.global_state = global_state or RouterGlobalState()
        self.conversation_id = self.global_state.get_or_create_conversation(
            user_id=user_id, 
            session_id=session_id,
            agent_type="router"
        )
        self.executor = create_simple_router_agent()
    
    def chat(self, user_input: str) -> str:
        """Chat with memory tracking"""
        # Log user message
        self.global_state.add_message(self.conversation_id, "human", user_input)
        
        try:
            # Get response from agent
            result = self.executor.invoke({"input": user_input})
            response = result['output']
            
            # Log agent response
            self.global_state.add_message(self.conversation_id, "ai", response)
            
            # Log tool calls if any
            if 'intermediate_steps' in result:
                for step in result['intermediate_steps']:
                    if len(step) >= 2:
                        tool_name = step[0].tool if hasattr(step[0], 'tool') else 'unknown'
                        self.global_state.log_tool_call("router", tool_name, step[0], step[1])
            
            return response
            
        except Exception as e:
            error_msg = f"Router Error: {str(e)}"
            self.global_state.add_message(self.conversation_id, "ai", error_msg)
            return error_msg
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get conversation history with memory"""
        return self.global_state.get_conversation_history(self.conversation_id, limit)
    
    async def get_system_info(self, _: str) -> str:
        """Get system info - for compatibility with old interface"""
        return get_system_info("")

# Export both the executor and the RouterAgent class
executor = create_simple_router_agent()

def run_simple_router_agent():
    """Interactive simple router agent"""
    print("\n🤖 Simple Router Agent Ready! I route your requests to the Sales Agent.")
    print("Examples:")
    print("- 'Show me top 5 customers'")
    print("- 'Last month sales data'")
    print("- 'Customer summary'")
    print("- 'Show recent leads'")
    print("- Type 'quit' to exit\n")
    
    executor = create_simple_router_agent()
    
    try:
        while True:
            user_input = input("Router > ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            try:
                result = executor.invoke({"input": user_input})
                print(f"\n{result['output']}\n")
            except Exception as e:
                print(f"Error: {str(e)}")
                
    except KeyboardInterrupt:
        print("\nGoodbye!")

# Export the executor
executor = create_simple_router_agent()

if __name__ == "__main__":
    run_simple_router_agent()
