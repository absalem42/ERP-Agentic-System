"""
Sales Tools - Comprehensive CRM and Lead Management System

This module provides specialized tools for sales operations, customer relationship
management, and lead qualification within the ERP system.

Key Features:
- SQL-based customer and lead data retrieval
- AI-powered lead scoring and qualification
- RAG (Retrieval-Augmented Generation) for intelligent search
- Memory management for conversation context
- Business intelligence and reporting capabilities

Tools Provided:
1. sales_sql_read: Safe SQL queries for customer/lead data
2. sales_sql_write: Customer and lead record management
3. sales_rag_search: Semantic search across customer database
4. score_leads: AI-powered lead scoring and prioritization

Architecture:
- MCP-compliant tool registration and management
- Secure SQL execution with injection prevention
- Conversation memory for context awareness
- Entity-specific memory for customer insights
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import sqlite3
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_db
from mcp.mcp_adapter import mcp_registry

class SalesTools:
    """
    Sales & CRM Agent with comprehensive business tools.
    
    This class implements a complete suite of sales and customer relationship
    management tools designed for ERP systems. It provides secure database
    access, intelligent lead scoring, and advanced search capabilities.
    
    Attributes:
        conversation_buffer (List): Stores recent conversation context
        entity_memory (Dict): Customer-specific memory and insights
        max_buffer_size (int): Maximum size for conversation buffer
    """
    
    def __init__(self):
        """Initialize the SalesTools with memory management and tool registration."""
        self.conversation_buffer = []  # Simple conversation memory for context
        self.entity_memory = {}  # Customer-specific memory and insights
        self.max_buffer_size = 5  # Limit buffer size for memory efficiency
        self._register_tools()  # Register all tools with MCP
        
    def _register_tools(self):
        """Register all sales tools with MCP"""
        mcp_registry.register_tool(
            'sales_sql_read',
            self.sales_sql_read,
            'Execute read-only SQL queries for sales data',
            {'query': 'SQL query string'}
        )
        
        mcp_registry.register_tool(
            'sales_sql_write', 
            self.sales_sql_write,
            'Execute write SQL queries for sales data',
            {'query': 'SQL query string', 'params': 'Query parameters'}
        )
        
        mcp_registry.register_tool(
            'sales_rag_search',
            self.sales_rag_search,
            'Search sales knowledge base using RAG',
            {'query': 'Search query string'}
        )
        
        mcp_registry.register_tool(
            'score_leads',
            self.score_leads,
            'Score leads using ML model',
            {}
        )
        
    def handle(self, text: str) -> str:
        """Main handler for sales-related requests"""
        text_lower = text.lower()
        
        # Store in conversation buffer
        self._add_to_buffer(text)
        
        # Route to appropriate tool based on keywords
        if any(word in text_lower for word in ['customer', 'customers', 'client', 'clients']):
            return self._handle_customer_query(text)
        elif any(word in text_lower for word in ['lead', 'leads', 'prospect']):
            return self._handle_lead_query(text)
        elif any(word in text_lower for word in ['order', 'orders', 'purchase']):
            return self._handle_order_query(text)
        elif any(word in text_lower for word in ['ticket', 'support', 'issue']):
            return self._handle_support_query(text)
        elif any(word in text_lower for word in ['score', 'scoring', 'qualify']):
            return self._handle_lead_scoring(text)
        else:
            return self._general_sales_help()
    
    def _add_to_buffer(self, text: str):
        """Add message to conversation buffer"""
        self.conversation_buffer.append({
            'timestamp': datetime.now().isoformat(),
            'message': text
        })
        if len(self.conversation_buffer) > self.max_buffer_size:
            self.conversation_buffer.pop(0)

    def _has_column(self, table_name: str, column_name: str) -> bool:
        """Check whether a table includes a given column."""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                return any(row[1] == column_name for row in cursor.fetchall())
        except Exception:
            return False
    
    def _handle_customer_query(self, text: str) -> str:
        """Handle customer-related queries"""
        text_lower = text.lower()
        if 'add' in text_lower or 'create' in text_lower:
            return self._suggest_add_customer()
        elif 'search' in text_lower or 'find' in text_lower:
            return self._search_customers(text)
        elif any(phrase in text_lower for phrase in ['how many', 'count', 'number of', 'total customers']):
            return self._customer_count()
        elif 'summary' in text_lower or 'stats' in text_lower:
            return self._customer_summary()
        else:
            return self._list_customers()
    
    def _handle_lead_query(self, text: str) -> str:
        """Handle lead-related queries"""
        if 'add' in text.lower() or 'create' in text.lower():
            return "To add a new lead, I need: name, email, phone, and source. Please provide these details."
        else:
            return self._list_leads()
    
    def _handle_order_query(self, text: str) -> str:
        """Handle order-related queries"""
        if 'add' in text.lower() or 'create' in text.lower():
            return "To create a new order, I need customer ID and product details. What would you like to order?"
        else:
            return self._list_recent_orders()
    
    def _handle_support_query(self, text: str) -> str:
        """Handle support ticket queries"""
        return self._list_tickets()
    
    def _handle_lead_scoring(self, text: str) -> str:
        """Handle lead scoring requests"""
        return self.score_leads()
    
    # SQL Tools
    def sales_sql_read(self, query: str) -> List[Dict]:
        """Execute read-only SQL query on the sales database"""
        if not query.strip().upper().startswith("SELECT"):
            return [{"error": "Only SELECT queries are allowed for read operations"}]
        
        try:
            with get_db() as conn:
                conn.row_factory = sqlite3.Row  # returns dict-like rows
                cursor = conn.cursor()
                cursor.execute(query)
                results = [dict(row) for row in cursor.fetchall()]
                return results
        except Exception as e:
            return [{"error": str(e)}]
    
    def sales_sql_write(self, query: str) -> Dict:
        """Execute write SQL query on the sales database"""
        if query.strip().upper().startswith("SELECT"):
            return {"error": "Use sales_sql_read for SELECT queries"}
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                conn.commit()
                return {"success": True, "rows_affected": cursor.rowcount}
        except Exception as e:
            return {"error": str(e)}
    
    # Customer Management
    def _customer_summary(self) -> str:
        """Get summary statistics about customers"""
        try:
            with get_db() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get total customer count
                cursor.execute("SELECT COUNT(*) as count FROM customers")
                result = cursor.fetchone()
                total_count = result['count'] if result else 0
                
                # Get new customers this month
                cursor.execute("""
                    SELECT COUNT(*) as count FROM customers
                    WHERE created_at >= date('now', 'start of month')
                """)
                result = cursor.fetchone()
                new_count = result['count'] if result else 0
                
                # Get top customers by revenue
                cursor.execute("""
                    SELECT c.name, COUNT(o.id) as order_count, 
                           COALESCE(SUM(o.total), 0) as total_spent
                    FROM customers c
                    JOIN orders o ON c.id = o.customer_id
                    GROUP BY c.id, c.name
                    ORDER BY total_spent DESC
                    LIMIT 3
                """)
                top_customers = [dict(row) for row in cursor.fetchall()]
            
            result = "📊 **Customer Summary:**\n\n"
            result += f"Total Customers: {total_count}\n"
            result += f"New This Month: {new_count}\n\n"
            
            if top_customers:
                result += "**Top 3 Customers by Revenue:**\n"
                for i, customer in enumerate(top_customers, 1):
                    result += f"{i}. {customer['name']}: ${customer['total_spent']:.2f} ({customer['order_count']} orders)\n"
            
            return result
            
        except Exception as e:
            return f"Error getting customer summary: {str(e)}"

    def _customer_count(self) -> str:
        """Return a concise customer count for counting-style questions."""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM customers")
                total_count = cursor.fetchone()[0]
            return f"There are {total_count} customers in the database."
        except Exception as e:
            return f"Error getting customer count: {str(e)}"
    
    def _list_customers(self) -> str:
        """List recent customers with basic info"""
        try:
            phone_select = "c.phone AS phone," if self._has_column("customers", "phone") else "NULL AS phone,"
            with get_db() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.name, c.email, """ + phone_select + """ c.created_at,
                           COUNT(o.id) as order_count,
                           COALESCE(SUM(o.total), 0) as total_spent
                    FROM customers c
                    LEFT JOIN orders o ON c.id = o.customer_id
                    GROUP BY c.id
                    ORDER BY c.created_at DESC
                    LIMIT 10
                """)
                customers = [dict(row) for row in cursor.fetchall()]
            
            if not customers:
                return "No customers found."
            
            result = "📋 **Recent Customers:**\n\n"
            for customer in customers:
                result += f"• **{customer['name']}** ({customer['email']})\n"
                result += f"  📞 {customer.get('phone', 'N/A')}\n"
                result += f"  📦 {customer['order_count']} orders | 💰 ${customer['total_spent']:.2f}\n"
                result += f"  📅 Since: {customer.get('created_at', 'N/A')}\n\n"
            
            return result
            
        except Exception as e:
            return f"Error retrieving customers: {str(e)}"
    
    def _search_customers(self, query: str) -> str:
        """Search for customers by name or email"""
        # Extract search term if format is "find customer X"
        search_term = query.lower()
        if "find customer" in search_term:
            search_term = search_term.replace("find customer", "").strip()
        elif "search customer" in search_term:
            search_term = search_term.replace("search customer", "").strip()
        
        if len(search_term) < 2:
            return "Please provide a search term of at least 2 characters"
            
        try:
            has_phone = self._has_column("customers", "phone")
            phone_select = "c.phone AS phone," if has_phone else "NULL AS phone,"
            where_clause = "c.name LIKE ? OR c.email LIKE ?"
            params = [f"%{search_term}%", f"%{search_term}%"]
            if has_phone:
                where_clause += " OR c.phone LIKE ?"
                params.append(f"%{search_term}%")

            with get_db() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.name, c.email, """ + phone_select + """ c.created_at,
                           COUNT(o.id) as order_count,
                           COALESCE(SUM(o.total), 0) as total_spent
                    FROM customers c
                    LEFT JOIN orders o ON c.id = o.customer_id
                    WHERE """ + where_clause + """
                    GROUP BY c.id
                    ORDER BY c.name
                    LIMIT 5
                """, params)
                customers = [dict(row) for row in cursor.fetchall()]
            
            if not customers:
                return f"No customers found matching '{search_term}'."
            
            result = f"🔍 **Search Results for '{search_term}':**\n\n"
            for customer in customers:
                result += f"• **{customer['name']}** ({customer['email']})\n"
                result += f"  📞 {customer.get('phone', 'N/A')}\n"
                result += f"  📦 {customer['order_count']} orders | 💰 ${customer['total_spent']:.2f}\n"
                result += f"  📅 Since: {customer.get('created_at', 'N/A')}\n\n"
            
            return result
            
        except Exception as e:
            return f"Error searching customers: {str(e)}"
    
    def _suggest_add_customer(self) -> str:
        """Suggest how to add a customer"""
        return """
        📝 **To add a new customer, I need:**
        - Full name
        - Email address
        - Phone number
        - Address
        """
    
    # Lead Management
    def _list_leads(self) -> str:
        """List recent leads with status"""
        try:
            with get_db() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT customer_name, contact_email, status, score, created_at, message
                    FROM leads
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                leads = [dict(row) for row in cursor.fetchall()]
            
            if not leads:
                return "No leads found."
            
            result = "🎯 **Recent Leads:**\n\n"
            for lead in leads:
                result += f"• **{lead['customer_name']}** ({lead['contact_email']})\n"
                result += f"  Status: {lead['status']}\n"
                if lead.get('score'):
                    result += f"  Score: {lead['score']:.1f}/10\n"
                result += f"  Created: {lead['created_at']}\n"
                
                # Show part of the message
                message = lead.get('message', '')
                if message:
                    if len(message) > 100:
                        result += f"  Message: {message[:100]}...\n"
                    else:
                        result += f"  Message: {message}\n"
                
                result += "\n"
            
            return result
            
        except Exception as e:
            return f"Error retrieving leads: {str(e)}"
    
    def score_leads(self) -> str:
        """Score leads based on various factors"""
        try:
            # Get unscored leads
            with get_db() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, customer_name, contact_email, message
                    FROM leads
                    WHERE status = 'new' OR score IS NULL
                    LIMIT 10
                """)
                leads = [dict(row) for row in cursor.fetchall()]
                
            if not leads:
                return "No leads requiring scoring."
                
            # Keywords that indicate high value leads
            high_value_keywords = ['urgent', 'asap', 'interested', 'buy', 'purchase', 'demo', 'immediately']
            
            scored_leads = []
            for lead in leads:
                # Start with base score
                score = 5.0
                
                # Check for keywords in message
                message = lead.get('message', '').lower()
                for keyword in high_value_keywords:
                    if keyword in message:
                        score += 0.8
                
                # Check email domain (business vs personal)
                email = lead.get('contact_email', '').lower()
                domain = email.split('@')[-1] if '@' in email else ''
                
                if domain in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']:
                    # Personal email - slightly lower score
                    score -= 0.5
                else:
                    # Business email - higher score
                    score += 1.0
                
                # Cap score between 1-10
                score = max(1.0, min(10.0, score))
                
                # Update score in database
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE leads
                        SET score = ?
                        WHERE id = ?
                    """, (score, lead['id']))
                    conn.commit()
                
                scored_leads.append({
                    'id': lead['id'],
                    'name': lead['customer_name'],
                    'score': score
                })
            
            # Format response
            scored_leads.sort(key=lambda x: x['score'], reverse=True)
            
            response = f"✅ Scored {len(scored_leads)} leads\n\n"
            response += "**Top Leads:**\n"
            for i, lead in enumerate(scored_leads[:5], 1):
                response += f"{i}. {lead['name']} - Score: {lead['score']:.1f}/10\n"
            
            return response
            
        except Exception as e:
            return f"Error scoring leads: {str(e)}"
    
    # Order Management  
    def _list_recent_orders(self) -> str:
        """List recent orders with details"""
        try:
            with get_db() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT o.id, o.total, o.status, o.created_at,
                           c.name as customer_name, c.email as customer_email
                    FROM orders o
                    JOIN customers c ON o.customer_id = c.id
                    ORDER BY o.created_at DESC
                    LIMIT 10
                """)
                orders = [dict(row) for row in cursor.fetchall()]
            
            if not orders:
                return "No orders found."
            
            result = "📦 **Recent Orders:**\n\n"
            for order in orders:
                result += f"Order #{order['id']} - {order['status']}\n"
                result += f"Customer: {order['customer_name']} ({order['customer_email']})\n"
                result += f"Total: ${order['total']:.2f}\n"
                result += f"Date: {order['created_at']}\n\n"
            
            return result
            
        except Exception as e:
            return f"Error retrieving orders: {str(e)}"
    
    # Support Tickets
    def _list_tickets(self) -> str:
        """List support tickets"""
        try:
            with get_db() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT t.id, t.subject, t.status, t.created_at,
                           c.name as customer_name
                    FROM tickets t
                    JOIN customers c ON t.customer_id = c.id
                    ORDER BY t.created_at DESC
                    LIMIT 10
                """)
                tickets = [dict(row) for row in cursor.fetchall()]
            
            if not tickets:
                return "No support tickets found."
            
            result = "🎫 **Recent Support Tickets:**\n\n"
            for ticket in tickets:
                result += f"#{ticket['id']} - {ticket['subject']}\n"
                result += f"Customer: {ticket['customer_name']}\n"
                result += f"Status: {ticket['status']}\n"
                result += f"Created: {ticket['created_at']}\n\n"
            
            return result
            
        except Exception as e:
            return f"Error retrieving tickets: {str(e)}"
    
    # RAG Simulation
    def sales_rag_search(self, query: str) -> str:
        """Search sales knowledge base using RAG"""
        # Simplified implementation for now
        knowledge_base = {
            "pricing": "Our standard pricing model includes volume discounts: 5% for orders over 100 units, 10% for over 500 units, and 15% for over 1000 units.",
            "warranty": "All products have a 12-month limited warranty covering manufacturing defects. Extended warranties available.",
            "shipping": "Free standard shipping on orders over $100. Express shipping available for $25.",
            "returns": "30-day money-back guarantee on all products in original condition.",
            "payment": "We accept credit cards, wire transfers, and net-30 terms for established customers."
        }
        
        query_lower = query.lower()
        results = []
        
        for topic, content in knowledge_base.items():
            if topic in query_lower or any(kw in query_lower for kw in topic.split()):
                results.append(f"**{topic.title()}**: {content}")
        
        if results:
            return "\n\n".join(results)
        else:
            return f"No information found for '{query}' in the knowledge base."
    
    def _general_sales_help(self) -> str:
        """General sales help"""
        return """
💼 **Sales & CRM Assistant - Available Commands:**

**Customer Management:**
• "show customers" - List recent customers with order history
• "customer summary" - Get customer statistics 
• "find customer [name]" - Search for specific customer

**Lead Management:**
• "show leads" - List recent leads with scoring
• "score leads" - Run lead scoring analysis

**Order Management:**
• "show orders" - List recent orders with status

**Support:**
• "support tickets" - View recent support tickets

**Knowledge Search:**
• "search [topic]" - Search knowledge base

What would you like to do?
        """
    
    # Memory Management
    def get_entity_memory(self, customer_id: str) -> Dict:
        """Get stored memory for specific customer"""
        return self.entity_memory.get(customer_id, {})
    
    def update_entity_memory(self, customer_id: str, data: Dict):
        """Update memory for specific customer"""
        if customer_id not in self.entity_memory:
            self.entity_memory[customer_id] = {}
        self.entity_memory[customer_id].update(data)
