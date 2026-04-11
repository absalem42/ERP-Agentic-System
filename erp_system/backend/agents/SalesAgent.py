"""
Sales Management Agent using LangChain
======================================
An agentic AI system for managing customers, leads, orders, and sales operations
using SQL queries and contextual business logic.
"""

import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from db import get_db
from config.llm import get_llm

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set Groq API key with proper error handling
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
else:
    print("WARNING: GROQ_API_KEY not found in environment. Sales agent may fall back to non-AI mode.")

# Import memory systems
from memory.base_memory import SalesEntityMemory, RouterGlobalState

# -------- Database Utilities --------
def execute_sql(query: str, params: tuple = ()) -> List[Dict]:
    """Execute SQL query using the shared database connection"""
    import sqlite3
    conn = sqlite3.connect(os.getenv("DB_PATH", ""))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_table_schema(table_name: str) -> str:
    """Get the schema information for a specific table"""
    schema = execute_sql(f"PRAGMA table_info({table_name})")
    return f"Table: {table_name}\n" + "\n".join([f"  - {col['name']} ({col['type']})" for col in schema])

def get_all_tables() -> List[str]:
    """Get list of all tables in the database"""
    tables = execute_sql("SELECT name FROM sqlite_master WHERE type='table'")
    return [t['name'] for t in tables]

# -------- Tool Functions --------
@tool
def sales_sql_query(question: str, context: Optional[str] = None) -> str:
    """
    Convert a natural language sales question to SQL, execute it, and return results.
    Specialized for sales operations: customers, leads, orders, products, suppliers.
    """
    tables = get_all_tables()
    # Focus on sales-related tables
    sales_tables = ['customers', 'leads', 'orders', 'order_items', 'products', 'suppliers', 'invoices', 'payments']
    relevant_tables = [t for t in tables if any(st in t.lower() for st in sales_tables)]
    
    schema_info = "\n".join([get_table_schema(t) for t in relevant_tables])
    prompt = f"""
    Given the following sales database schema:
    {schema_info}

    Convert this sales question to a SQL query: {question}
    Additional context: {context if context else 'None'}
    
    Focus on sales operations:
    - Customer management and relationships
    - Lead tracking and conversion
    - Order processing and history
    - Product and supplier information
    - Revenue and payment tracking
    
    Return only the SQL query without any explanation.
    SQL Query:
    """
    llm = get_llm()  # Use shared LLM configuration
    response = llm.invoke(prompt)
    # Handle both string and AIMessage responses
    if hasattr(response, 'content'):
        sql_query = response.content.strip()
    else:
        sql_query = str(response).strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    
    try:
        results = execute_sql(sql_query)
        if results:
            df = pd.DataFrame(results)
            return f"Query executed successfully. Results:\n{df.to_string()}\n\nSQL: {sql_query}"
        else:
            return f"Query executed but returned no results.\nSQL: {sql_query}"
    except Exception as e:
        return f"Error executing SQL query: {str(e)}\nGenerated SQL: {sql_query}"

@tool
def customer_management(operation: str, customer_data: Optional[Dict] = None) -> str:
    """
    Manage customer operations: create, update, search, and analyze customers.
    Operations: 'list', 'search', 'summary', 'details', 'create', 'update'
    """
    try:
        if operation == 'list':
            results = execute_sql("""
                SELECT c.*, COUNT(o.id) as order_count,
                       COALESCE(SUM(o.total), 0) as total_spent
                FROM customers c 
                LEFT JOIN orders o ON c.id = o.customer_id 
                GROUP BY c.id 
                ORDER BY c.created_at DESC
                LIMIT 20
            """)
            if not results:
                return "No customers found"
            
            output = "📋 **Recent Customers:**\n\n"
            for customer in results:
                output += f"• **{customer['name']}** ({customer['email']})\n"
                output += f"  📞 {customer.get('phone', 'N/A')}\n"
                output += f"  📦 {customer['order_count']} orders | 💰 ${customer['total_spent']:.2f}\n"
                output += f"  📅 Customer since: {customer.get('created_at', 'N/A')}\n\n"
            return output
            
        elif operation == 'summary':
            total_count = execute_sql("SELECT COUNT(*) as count FROM customers")
            count = total_count[0]['count'] if total_count else 0
            
            new_customers = execute_sql("""
                SELECT COUNT(*) as count FROM customers
                WHERE created_at >= date('now', 'start of month')
            """)
            new_count = new_customers[0]['count'] if new_customers else 0
            
            customer_data = execute_sql("""
                SELECT 
                    COUNT(DISTINCT c.id) as customer_count,
                    COUNT(o.id) as order_count,
                    COALESCE(SUM(o.total), 0) as total_revenue,
                    COALESCE(AVG(o.total), 0) as avg_order_value
                FROM customers c
                LEFT JOIN orders o ON c.id = o.customer_id
            """)
            
            result = "📊 **Customer Summary:**\n\n"
            result += f"Total Customers: {count}\n"
            result += f"New This Month: {new_count}\n"
            
            if customer_data and len(customer_data) > 0:
                data = customer_data[0]
                result += f"Total Orders: {data.get('order_count', 0)}\n"
                result += f"Total Revenue: ${data.get('total_revenue', 0):.2f}\n"
                result += f"Average Order Value: ${data.get('avg_order_value', 0):.2f}\n"
            
            return result
            
        elif operation == 'search' and customer_data:
            search_term = customer_data.get('search_term', '')
            results = execute_sql("""
                SELECT * FROM customers 
                WHERE name LIKE ? OR email LIKE ? OR phone LIKE ?
                ORDER BY created_at DESC
            """, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"))
            
            if not results:
                return f"No customers found matching '{search_term}'"
            
            output = f"🔍 **Search Results for '{search_term}':**\n\n"
            for customer in results:
                output += f"• **{customer['name']}** ({customer['email']})\n"
                output += f"  📞 {customer.get('phone', 'N/A')}\n\n"
            return output
            
        else:
            return f"Unknown operation: {operation}"
            
    except Exception as e:
        return f"Error in customer management: {str(e)}"

@tool
def lead_management(operation: str, lead_data: Optional[Dict] = None) -> str:
    """
    Manage lead operations: track, convert, and analyze sales leads.
    Operations: 'list', 'summary', 'convert', 'update_status'
    """
    try:
        if operation == 'list':
            results = execute_sql("""
                SELECT * FROM leads 
                ORDER BY created_at DESC
                LIMIT 20
            """)
            if not results:
                return "No leads found"
            
            output = "📈 **Recent Leads:**\n\n"
            for lead in results:
                status_emoji = {"new": "🆕", "contacted": "📞", "qualified": "✅", "converted": "🎉", "lost": "❌"}.get(lead.get('status', '').lower(), "📋")
                output += f"{status_emoji} **{lead['name']}** ({lead['email']})\n"
                output += f"  📱 {lead.get('phone', 'N/A')}\n"
                output += f"  🏢 {lead.get('company', 'N/A')}\n"
                output += f"  📊 Status: {lead.get('status', 'N/A')}\n"
                output += f"  💰 Value: ${lead.get('estimated_value', 0):.2f}\n\n"
            return output
            
        elif operation == 'summary':
            summary = execute_sql("""
                SELECT 
                    COUNT(*) as total_leads,
                    COUNT(CASE WHEN status = 'converted' THEN 1 END) as converted,
                    COUNT(CASE WHEN status = 'new' THEN 1 END) as new_leads,
                    AVG(estimated_value) as avg_value,
                    SUM(estimated_value) as total_value
                FROM leads
            """)
            
            if not summary:
                return "No lead data available"
            
            data = summary[0]
            result = "📈 **Lead Summary:**\n\n"
            result += f"Total Leads: {data['total_leads']}\n"
            result += f"Converted: {data['converted']}\n"
            result += f"New Leads: {data['new_leads']}\n"
            result += f"Average Value: ${data.get('avg_value', 0):.2f}\n"
            result += f"Total Pipeline Value: ${data.get('total_value', 0):.2f}\n"
            
            conversion_rate = (data['converted'] / data['total_leads'] * 100) if data['total_leads'] > 0 else 0
            result += f"Conversion Rate: {conversion_rate:.1f}%\n"
            
            return result
            
        else:
            return f"Unknown lead operation: {operation}"
            
    except Exception as e:
        return f"Error in lead management: {str(e)}"

@tool
def order_management(operation: str, order_data: Optional[Dict] = None) -> str:
    """
    Manage order operations: track, analyze, and process sales orders.
    Operations: 'list', 'summary', 'details', 'status_update'
    """
    try:
        if operation == 'list':
            results = execute_sql("""
                SELECT o.*, c.name as customer_name, c.email as customer_email
                FROM orders o
                LEFT JOIN customers c ON o.customer_id = c.id
                ORDER BY o.created_at DESC
                LIMIT 20
            """)
            if not results:
                return "No orders found"
            
            output = "📦 **Recent Orders:**\n\n"
            for order in results:
                status_emoji = {"pending": "⏳", "paid": "✅", "shipped": "🚚", "delivered": "📦", "cancelled": "❌"}.get(order.get('status', '').lower(), "📋")
                output += f"{status_emoji} **Order #{order['id']}** - ${order['total']:.2f}\n"
                output += f"  👤 {order.get('customer_name', 'N/A')} ({order.get('customer_email', 'N/A')})\n"
                output += f"  📊 Status: {order.get('status', 'N/A')}\n"
                output += f"  📅 Created: {order.get('created_at', 'N/A')}\n\n"
            return output
            
        elif operation == 'summary':
            summary = execute_sql("""
                SELECT 
                    COUNT(*) as total_orders,
                    COUNT(CASE WHEN status = 'paid' THEN 1 END) as paid_orders,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_orders,
                    COUNT(CASE WHEN status = 'shipped' THEN 1 END) as shipped_orders,
                    SUM(total) as total_revenue,
                    AVG(total) as avg_order_value
                FROM orders
            """)
            
            if not summary:
                return "No order data available"
            
            data = summary[0]
            result = "📦 **Order Summary:**\n\n"
            result += f"Total Orders: {data['total_orders']}\n"
            result += f"Paid Orders: {data['paid_orders']}\n"
            result += f"Pending Orders: {data['pending_orders']}\n"
            result += f"Shipped Orders: {data['shipped_orders']}\n"
            result += f"Total Revenue: ${data.get('total_revenue', 0):.2f}\n"
            result += f"Average Order Value: ${data.get('avg_order_value', 0):.2f}\n"
            
            return result
            
        else:
            return f"Unknown order operation: {operation}"
            
    except Exception as e:
        return f"Error in order management: {str(e)}"

@tool
def sales_reporting(report_type: str, params: Optional[Dict] = None) -> str:
    """
    Generate sales reports and analytics.
    Report types: 'revenue', 'top_customers', 'top_products', 'conversion_funnel'
    """
    try:
        if report_type == 'revenue':
            period = params.get('period', 'month') if params else 'month'
            
            if period == 'month':
                results = execute_sql("""
                    SELECT 
                        strftime('%Y-%m', created_at) as period,
                        COUNT(*) as order_count,
                        SUM(total) as revenue
                    FROM orders
                    WHERE created_at >= date('now', '-12 months')
                    GROUP BY strftime('%Y-%m', created_at)
                    ORDER BY period DESC
                """)
            else:
                results = execute_sql("""
                    SELECT 
                        date(created_at) as period,
                        COUNT(*) as order_count,
                        SUM(total) as revenue
                    FROM orders
                    WHERE created_at >= date('now', '-30 days')
                    GROUP BY date(created_at)
                    ORDER BY period DESC
                """)
            
            if not results:
                return "No revenue data available"
            
            output = f"💰 **Revenue Report ({period}ly):**\n\n"
            for row in results[:10]:  # Show top 10 periods
                output += f"📅 {row['period']}: ${row['revenue']:.2f} ({row['order_count']} orders)\n"
            return output
            
        elif report_type == 'top_customers':
            limit = params.get('limit', 10) if params else 10
            results = execute_sql("""
                SELECT 
                    c.name, c.email,
                    COUNT(o.id) as order_count,
                    SUM(o.total) as total_spent
                FROM customers c
                LEFT JOIN orders o ON c.id = o.customer_id
                GROUP BY c.id
                ORDER BY total_spent DESC
                LIMIT ?
            """, (limit,))
            
            if not results:
                return "No customer data available"
            
            output = f"🏆 **Top {limit} Customers by Revenue:**\n\n"
            for i, row in enumerate(results, 1):
                output += f"{i}. **{row['name']}** ({row['email']})\n"
                output += f"   💰 ${row.get('total_spent', 0):.2f} ({row['order_count']} orders)\n\n"
            return output
            
        else:
            return f"Unknown report type: {report_type}"
            
    except Exception as e:
        return f"Error generating report: {str(e)}"

# -------- Agent System Prompt --------
SALES_AGENT_SYSTEM = """You are the Sales Management Agent for Helios Dynamics.

Your responsibilities:
- Manage customer relationships and data
- Track and convert sales leads
- Process and analyze sales orders
- Generate sales reports and insights
- Support sales operations

Available tools: {tools}
Tool names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

# -------- Sales Agent Wrapper --------
class SalesAgentWrapper:
    """Wrapper class for Sales Agent to provide consistent interface"""
    
    def __init__(self):
        self.executor = create_sales_agent()
        self.entity_memory = SalesEntityMemory()
        self.global_state = RouterGlobalState()
    
    def chat(self, message: str) -> str:
        """Chat interface that mimics other agents"""
        try:
            result = self.executor.invoke({"input": message})
            return result.get('output', str(result))
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def invoke(self, request_data: dict) -> dict:
        """Direct invoke method for compatibility"""
        return self.executor.invoke(request_data)

# -------- Build the Sales Agent --------
def create_sales_agent():
    """Create and configure the Sales Agent"""
    llm = get_llm()  # Use shared LLM configuration
    tools = [sales_sql_query, customer_management, lead_management, order_management, sales_reporting]
    memory = ConversationBufferMemory()
    prompt = PromptTemplate.from_template(SALES_AGENT_SYSTEM)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        memory=memory
    )
    return executor

def create_sales_agent_with_chat():
    """Create Sales Agent with chat interface"""
    return SalesAgentWrapper()

if __name__ == "__main__":
    executor = create_sales_agent()
    print("🛍️ Helios Dynamics - Sales Management Agent Ready!")
    print("Ask me about customers, leads, orders, or sales reports.")
    try:
        while True:
            user_input = input("Sales Agent > ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            try:
                result = executor.invoke({"input": user_input})
                print(f"\n{result['output']}\n")
            except Exception as e:
                print(f"Error: {str(e)}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
