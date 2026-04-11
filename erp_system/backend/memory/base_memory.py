"""
Base Memory Management System for ERP Agents
Implements required memory systems according to specifications
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

def get_db_path():
    """Get database path"""
    configured_path = os.getenv("DB_PATH")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).parent.parent.parent / "databases" / "erp.db"

class RouterGlobalState:
    """Manages router's global state and persistence"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(get_db_path())
        self._init_tables()
    
    def _init_tables(self):
        """Initialize required memory tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if conversations table exists and has required columns
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [col[1] for col in cursor.fetchall()]
        table_exists = 'conversations' in [table[0] for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        
        if table_exists:
            if 'session_id' not in columns:
                cursor.execute('ALTER TABLE conversations ADD COLUMN session_id TEXT')
                print("✅ Added missing session_id column to conversations table")
            
            if 'agent_type' not in columns:
                cursor.execute('ALTER TABLE conversations ADD COLUMN agent_type TEXT')
                print("✅ Added missing agent_type column to conversations table")
            
            if 'created_at' not in columns:
                cursor.execute('ALTER TABLE conversations ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
                print("✅ Added missing created_at column to conversations table")
        
        # Check messages table columns
        cursor.execute("PRAGMA table_info(messages)")
        msg_columns = [col[1] for col in cursor.fetchall()]
        msg_table_exists = 'messages' in [table[0] for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        
        if msg_table_exists:
            if 'role' not in msg_columns:
                cursor.execute('ALTER TABLE messages ADD COLUMN role TEXT')
                print("✅ Added missing role column to messages table")
            
            if 'content' not in msg_columns:
                cursor.execute('ALTER TABLE messages ADD COLUMN content TEXT')
                print("✅ Added missing content column to messages table")
            
            if 'timestamp' not in msg_columns:
                cursor.execute('ALTER TABLE messages ADD COLUMN timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
                print("✅ Added missing timestamp column to messages table")
        
        # Create required tables for memory
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'default_user',
                session_id TEXT,
                agent_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_type TEXT,
                status TEXT DEFAULT 'pending',
                requested_by TEXT,
                approved_by TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT,
                tool_name TEXT,
                input_data TEXT,
                output_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_or_create_conversation(self, user_id: str = "default_user", session_id: str = None, agent_type: str = "router"):
        """Get or create conversation session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute('''
                SELECT id FROM conversations WHERE session_id = ? AND user_id = ?
            ''', (session_id, user_id))
            result = cursor.fetchone()
            if result:
                conn.close()
                return result[0]
        
        # Create new conversation
        cursor.execute('''
            INSERT INTO conversations (user_id, session_id, agent_type, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (user_id, session_id or f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", agent_type))
        
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return conversation_id
    
    def add_message(self, conversation_id: int, role: str, content: str):
        """Add message to conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (conversation_id, role, content, timestamp)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (conversation_id, role, content))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, conversation_id: int, limit: int = 10) -> List[Dict]:
        """Get conversation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT role, content, timestamp FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (conversation_id, limit))
        
        messages = []
        for role, content, timestamp in cursor.fetchall():
            messages.append({
                'role': role,
                'content': content,
                'timestamp': timestamp
            })
        
        conn.close()
        return list(reversed(messages))  # Return in chronological order
    
    def log_tool_call(self, agent_type: str, tool_name: str, input_data: Any, output_data: Any):
        """Log tool call for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tool_calls (agent_type, tool_name, input_data, output_data, timestamp)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (agent_type, tool_name, json.dumps(input_data), json.dumps(output_data)))
        
        conn.commit()
        conn.close()

class SalesEntityMemory:
    """Manages customer entity memory for Sales Agent"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(get_db_path())
        self._init_customer_kv_table()
    
    def _init_customer_kv_table(self):
        """Initialize customer key-value memory table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customer_kv (
                customer_id INTEGER,
                key TEXT,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (customer_id, key)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def set_customer_info(self, customer_id: int, key: str, value: str):
        """Store customer entity information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO customer_kv (customer_id, key, value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (customer_id, key, value))
        
        conn.commit()
        conn.close()
    
    def get_customer_info(self, customer_id: int, key: str = None) -> Any:
        """Retrieve customer entity information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if key:
            cursor.execute('''
                SELECT value FROM customer_kv WHERE customer_id = ? AND key = ?
            ''', (customer_id, key))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        else:
            cursor.execute('''
                SELECT key, value FROM customer_kv WHERE customer_id = ?
            ''', (customer_id,))
            result = dict(cursor.fetchall())
            conn.close()
            return result
    
    def update_last_interaction(self, customer_id: int, interaction_type: str):
        """Update customer's last interaction info"""
        self.set_customer_info(customer_id, "last_interaction", interaction_type)
        self.set_customer_info(customer_id, "last_interaction_date", datetime.now().isoformat())

class AnalyticsReportMemory:
    """Manages saved reports for Analytics Agent"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(get_db_path())
        self._init_saved_reports_table()
    
    def _init_saved_reports_table(self):
        """Initialize saved_reports table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS saved_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_name TEXT UNIQUE,
                sql_query TEXT,
                parameters TEXT,
                created_by TEXT DEFAULT 'analytics_agent',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_run TIMESTAMP,
                run_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_report(self, report_name: str, sql_query: str, parameters: dict = None, created_by: str = "analytics_agent") -> str:
        """Save a report for future use"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO saved_reports (report_name, sql_query, parameters, created_by, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (report_name, sql_query, json.dumps(parameters or {}), created_by))
            
            conn.commit()
            conn.close()
            return f"Report '{report_name}' saved successfully"
        except sqlite3.IntegrityError:
            conn.close()
            return f"Report '{report_name}' already exists"
    
    def get_saved_report(self, report_name: str) -> Optional[Dict]:
        """Retrieve a saved report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT report_name, sql_query, parameters, created_by, created_at, last_run, run_count
            FROM saved_reports WHERE report_name = ?
        ''', (report_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'name': result[0],
                'sql_query': result[1], 
                'parameters': json.loads(result[2]) if result[2] else {},
                'created_by': result[3],
                'created_at': result[4],
                'last_run': result[5],
                'run_count': result[6]
            }
        return None
    
    def update_report_run(self, report_name: str):
        """Update report run statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE saved_reports 
            SET last_run = CURRENT_TIMESTAMP, run_count = run_count + 1
            WHERE report_name = ?
        ''', (report_name,))
        
        conn.commit()
        conn.close()
