from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from backend.db import get_db


class RouterGlobalState:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path
        self._ensure_support_columns()

    def _ensure_support_columns(self) -> None:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("PRAGMA table_info(conversations)")
            conversation_columns = {row[1] for row in cursor.fetchall()}
            if conversation_columns and "session_id" not in conversation_columns:
                cursor.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT")
            if conversation_columns and "agent_type" not in conversation_columns:
                cursor.execute("ALTER TABLE conversations ADD COLUMN agent_type TEXT")
            if conversation_columns and "created_at" not in conversation_columns:
                cursor.execute("ALTER TABLE conversations ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

            cursor.execute("PRAGMA table_info(messages)")
            message_columns = {row[1] for row in cursor.fetchall()}
            if message_columns and "role" not in message_columns:
                cursor.execute("ALTER TABLE messages ADD COLUMN role TEXT")
            if message_columns and "timestamp" not in message_columns:
                cursor.execute("ALTER TABLE messages ADD COLUMN timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

            conn.commit()

    def get_or_create_conversation(
        self,
        user_id: int | str = 1,
        session_id: str | None = None,
        agent_type: str = "router",
    ) -> int:
        normalized_session = session_id or f"{agent_type}-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM conversations WHERE session_id = ? AND user_id = ?",
                (normalized_session, user_id),
            )
            row = cursor.fetchone()
            if row:
                return int(row[0])

            cursor.execute(
                """
                INSERT INTO conversations (user_id, started_at, session_id, agent_type, created_at)
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, CURRENT_TIMESTAMP)
                """,
                (user_id, normalized_session, agent_type),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def set_active_module(self, conversation_id: int, module: str) -> None:
        with get_db(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET agent_type = ?, created_at = CURRENT_TIMESTAMP WHERE id = ?",
                (module, conversation_id),
            )
            conn.commit()

    def get_active_module(self, conversation_id: int) -> str | None:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT agent_type FROM conversations WHERE id = ?", (conversation_id,))
            row = cursor.fetchone()
            return str(row[0]) if row and row[0] else None

    def add_message(self, conversation_id: int, sender: str, content: str, role: str | None = None) -> None:
        resolved_role = role or sender
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO messages (conversation_id, sender, content, created_at, role, timestamp)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, CURRENT_TIMESTAMP)
                """,
                (conversation_id, sender, content, resolved_role),
            )
            conn.commit()

    def get_conversation_history(self, conversation_id: int, limit: int = 20) -> list[dict[str, Any]]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT sender, content, COALESCE(timestamp, created_at) AS created
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            )
            rows = cursor.fetchall()
        history = [
            {"sender": row[0], "content": row[1], "created_at": row[2]}
            for row in reversed(rows)
        ]
        return history

    def create_approval(self, module: str, payload: dict[str, Any], requested_by: str) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO approvals (module, payload_json, status, requested_by, created_at)
                VALUES (?, ?, 'pending', ?, CURRENT_TIMESTAMP)
                """,
                (module, json.dumps(payload), requested_by),
            )
            approval_id = int(cursor.lastrowid)
            conn.commit()
        return {
            "id": approval_id,
            "module": module,
            "status": "pending",
            "requested_by": requested_by,
            "payload_json": payload,
        }

    def list_approvals(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = """
            SELECT id, module, payload_json, status, requested_by, decided_by, created_at, decided_at
            FROM approvals
        """
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
        approvals = []
        for row in rows:
            approvals.append(
                {
                    "id": row[0],
                    "module": row[1],
                    "payload_json": json.loads(row[2]) if row[2] else {},
                    "status": row[3],
                    "requested_by": row[4],
                    "decided_by": row[5],
                    "created_at": row[6],
                    "decided_at": row[7],
                }
            )
        return approvals

    def resolve_approval(self, approval_id: int, decision: str, decided_by: str) -> dict[str, Any] | None:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE approvals
                SET status = ?, decided_by = ?, decided_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (decision, decided_by, approval_id),
            )
            conn.commit()
            cursor.execute(
                """
                SELECT id, module, payload_json, status, requested_by, decided_by, created_at, decided_at
                FROM approvals WHERE id = ?
                """,
                (approval_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "module": row[1],
            "payload_json": json.loads(row[2]) if row[2] else {},
            "status": row[3],
            "requested_by": row[4],
            "decided_by": row[5],
            "created_at": row[6],
            "decided_at": row[7],
        }

    def last_tool_call_id(self) -> int:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COALESCE(MAX(id), 0) FROM tool_calls")
            return int(cursor.fetchone()[0])

    def log_tool_call(self, agent: str, tool_name: str, input_data: Any, output_data: Any) -> int:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_calls (agent, tool_name, input_json, output_json, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (agent, tool_name, json.dumps(input_data), json.dumps(output_data)),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_tool_calls(self, limit: int = 50, since_id: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT id, agent, tool_name, input_json, output_json, created_at
            FROM tool_calls
        """
        params: list[Any] = []
        if since_id is not None:
            query += " WHERE id > ?"
            params.append(since_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
        tool_calls = []
        for row in reversed(rows):
            tool_calls.append(
                {
                    "id": row[0],
                    "agent": row[1],
                    "tool_name": row[2],
                    "input_json": json.loads(row[3]) if row[3] else {},
                    "output_json": json.loads(row[4]) if row[4] else {},
                    "created_at": row[5],
                }
            )
        return tool_calls


class SalesEntityMemory:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path

    def set_customer_info(self, customer_id: int, key: str, value: str) -> None:
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO customer_kv (customer_id, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT(customer_id, key) DO UPDATE SET value = excluded.value
                """,
                (customer_id, key, value),
            )
            conn.commit()

    def get_customer_info(self, customer_id: int, key: str | None = None) -> Any:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            if key is not None:
                cursor.execute(
                    "SELECT value FROM customer_kv WHERE customer_id = ? AND key = ?",
                    (customer_id, key),
                )
                row = cursor.fetchone()
                return row[0] if row else None
            cursor.execute("SELECT key, value FROM customer_kv WHERE customer_id = ?", (customer_id,))
            return {row[0]: row[1] for row in cursor.fetchall()}


class AnalyticsReportMemory:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path

    def save_report(self, title: str, sql: str) -> str:
        with get_db(self.db_path) as conn:
            conn.execute(
                "INSERT INTO saved_reports (title, sql, created_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (title, sql),
            )
            conn.commit()
        return f"Saved report '{title}'"

    def get_saved_report(self, title: str) -> dict[str, Any] | None:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, sql, created_at FROM saved_reports WHERE lower(title) = lower(?)",
                (title,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {"id": row[0], "title": row[1], "sql": row[2], "created_at": row[3]}

    def list_reports(self, limit: int = 50) -> list[dict[str, Any]]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, sql, created_at FROM saved_reports ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        return [{"id": row[0], "title": row[1], "sql": row[2], "created_at": row[3]} for row in rows]
