from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from backend.config.llm import coerce_text, get_llm, has_llm_credentials
from backend.db import get_db
from backend.memory.base_memory import RouterGlobalState, SalesEntityMemory
from backend.mcp.tool_registry import ToolRegistry
from backend.tools.common import extract_json_payload, format_rows, looks_like_json_payload, parse_llm_json_object


class SalesTools:
    def __init__(
        self,
        state_store: RouterGlobalState,
        entity_memory: SalesEntityMemory,
        registry: ToolRegistry,
        db_path: str | None = None,
    ):
        self.state_store = state_store
        self.entity_memory = entity_memory
        self.registry = registry
        self.db_path = db_path
        self.registry.register_tool(
            name="sales_query_tool",
            handler=self.sales_query_tool,
            description="Read customer, lead, order, and ticket information for the sales domain",
            input_schema={"message": "str"},
            module="sales",
        )
        self.registry.register_tool(
            name="create_lead_tool",
            handler=self.create_lead_tool,
            description="Create a lead in the sales domain",
            input_schema={"customer_name": "str", "contact_email": "str", "message": "str"},
            module="sales",
            read_only=False,
        )
        self.registry.register_tool(
            name="create_order_tool",
            handler=self.create_order_tool,
            description="Create an order and order items",
            input_schema={"customer_id": "int", "items": "list[dict]", "status": "str"},
            module="sales",
            read_only=False,
        )
        self.registry.register_tool(
            name="update_lead_tool",
            handler=self.update_lead_tool,
            description="Update a lead status",
            input_schema={"lead_id": "int", "status": "str"},
            module="sales",
            read_only=False,
        )
        self.registry.register_tool(
            name="create_ticket_tool",
            handler=self.create_ticket_tool,
            description="Create a support ticket",
            input_schema={"customer_id": "int", "subject": "str", "body": "str"},
            module="sales",
            read_only=False,
        )
        self.registry.register_tool(
            name="lead_score_tool",
            handler=self.lead_score_tool,
            description="Score new leads and persist scores",
            input_schema={"lead_id": "int | None"},
            module="sales",
            read_only=False,
        )
        self.registry.register_tool(
            name="sales_rag_tool",
            handler=self.sales_rag_tool,
            description="Retrieve sales playbook and CRM context from glossary/documents",
            input_schema={"query": "str"},
            module="sales",
        )

    def _log(self, tool_name: str, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        self.state_store.log_tool_call("sales", tool_name, payload, result)
        return result

    def sales_query_tool(self, message: str) -> dict[str, Any]:
        lowered = message.lower()
        rows: list[dict[str, Any]] = []
        title = "Sales Query"
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            if "how many customers" in lowered or "count customers" in lowered:
                cursor.execute("SELECT COUNT(*) AS customer_count FROM customers")
                rows = [dict(cursor.fetchone())]
                title = "Customer Count"
            elif "lead" in lowered:
                cursor.execute(
                    """
                    SELECT id, customer_name, contact_email, COALESCE(score, 0) AS score, status, created_at
                    FROM leads ORDER BY id DESC LIMIT 10
                    """
                )
                rows = [dict(row) for row in cursor.fetchall()]
                title = "Recent Leads"
            elif "order" in lowered:
                cursor.execute(
                    """
                    SELECT o.id, c.name AS customer_name, o.total, o.status, o.created_at
                    FROM orders o JOIN customers c ON c.id = o.customer_id
                    ORDER BY o.id DESC LIMIT 10
                    """
                )
                rows = [dict(row) for row in cursor.fetchall()]
                title = "Recent Orders"
            elif "ticket" in lowered:
                cursor.execute(
                    """
                    SELECT t.id, c.name AS customer_name, t.subject, t.status, t.created_at
                    FROM tickets t LEFT JOIN customers c ON c.id = t.customer_id
                    ORDER BY t.id DESC LIMIT 10
                    """
                )
                rows = [dict(row) for row in cursor.fetchall()]
                title = "Recent Tickets"
            else:
                cursor.execute(
                    """
                    SELECT c.id, c.name, c.email, c.phone, COUNT(o.id) AS order_count, COALESCE(SUM(o.total), 0) AS total_spent
                    FROM customers c
                    LEFT JOIN orders o ON o.customer_id = c.id
                    GROUP BY c.id, c.name, c.email, c.phone
                    ORDER BY c.id
                    """
                )
                rows = [dict(row) for row in cursor.fetchall()]
                title = "Customers"
        if title == "Customer Count":
            message_text = f"There are {rows[0]['customer_count']} customers in the database."
        else:
            message_text = f"{title}\n\n{format_rows(rows)}"
        return self._log("sales_query_tool", {"message": message}, {"message": message_text, "rows": rows})

    def create_lead_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        features = {
            "message_length": len(payload.get("message", "")),
            "has_urgent": "urgent" in payload.get("message", "").lower(),
            "has_demo": "demo" in payload.get("message", "").lower(),
        }
        score = self._score_lead_payload(payload, features)
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO leads (customer_name, contact_email, message, score, status, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    payload["customer_name"],
                    payload["contact_email"],
                    payload["message"],
                    score,
                    payload.get("status", "new"),
                ),
            )
            lead_id = int(cursor.lastrowid)
            cursor.execute(
                """
                INSERT INTO ml_features_cache (entity_type, entity_id, feature_json, created_at)
                VALUES ('lead', ?, ?, CURRENT_TIMESTAMP)
                """,
                (lead_id, json.dumps(features)),
            )
            conn.commit()
        result = {"message": f"Lead created with id {lead_id} and score {score:.2f}.", "lead_id": lead_id, "score": score}
        return self._log("create_lead_tool", payload, result)

    def create_order_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        items = payload.get("items", [])
        if not items:
            result = {"message": "Order creation failed: items are required."}
            return self._log("create_order_tool", payload, result)

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            total = 0.0
            resolved_items = []
            for item in items:
                cursor.execute("SELECT id, price, name FROM products WHERE id = ?", (item["product_id"],))
                product = cursor.fetchone()
                if not product:
                    result = {"message": f"Order creation failed: product {item['product_id']} not found."}
                    return self._log("create_order_tool", payload, result)
                quantity = int(item["quantity"])
                line_price = float(item.get("price", product["price"]))
                total += quantity * line_price
                resolved_items.append({"product_id": product["id"], "quantity": quantity, "price": line_price, "name": product["name"]})

            cursor.execute(
                """
                INSERT INTO orders (customer_id, total, status, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (payload["customer_id"], total, payload.get("status", "pending")),
            )
            order_id = int(cursor.lastrowid)
            for item in resolved_items:
                cursor.execute(
                    """
                    INSERT INTO order_items (order_id, product_id, quantity, price)
                    VALUES (?, ?, ?, ?)
                    """,
                    (order_id, item["product_id"], item["quantity"], item["price"]),
                )
            conn.commit()

        self.entity_memory.set_customer_info(
            payload["customer_id"],
            "last_order_date",
            datetime.now(timezone.utc).isoformat(),
        )
        self.entity_memory.set_customer_info(payload["customer_id"], "last_order_total", f"{total:.2f}")
        result = {
            "message": f"Order created with id {order_id} for customer {payload['customer_id']} totaling {total:.2f}.",
            "order_id": order_id,
            "total": total,
        }
        return self._log("create_order_tool", payload, result)

    def create_ticket_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tickets (customer_id, subject, body, status, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (payload["customer_id"], payload["subject"], payload["body"], payload.get("status", "open")),
            )
            ticket_id = int(cursor.lastrowid)
            conn.commit()
        result = {"message": f"Ticket created with id {ticket_id}.", "ticket_id": ticket_id}
        return self._log("create_ticket_tool", payload, result)

    def update_lead_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE leads SET status = ? WHERE id = ?", (payload["status"], payload["lead_id"]))
            if cursor.rowcount == 0:
                result = {"message": f"Lead {payload['lead_id']} not found."}
                return self._log("update_lead_tool", payload, result)
            conn.commit()
        result = {"message": f"Lead {payload['lead_id']} updated to status {payload['status']}."}
        return self._log("update_lead_tool", payload, result)

    def lead_score_tool(self, lead_id: int | None = None) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            if lead_id is None:
                cursor.execute("SELECT id, customer_name, contact_email, message FROM leads WHERE score IS NULL OR status = 'new'")
            else:
                cursor.execute(
                    "SELECT id, customer_name, contact_email, message FROM leads WHERE id = ?",
                    (lead_id,),
                )
            leads = [dict(row) for row in cursor.fetchall()]

        scored = []
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            for lead in leads:
                features = {
                    "message_length": len(lead.get("message", "")),
                    "has_urgent": "urgent" in lead.get("message", "").lower(),
                    "has_demo": "demo" in lead.get("message", "").lower(),
                }
                score = self._score_lead_payload(lead, features)
                cursor.execute("UPDATE leads SET score = ? WHERE id = ?", (score, lead["id"]))
                cursor.execute(
                    """
                    INSERT INTO ml_features_cache (entity_type, entity_id, feature_json, created_at)
                    VALUES ('lead', ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (lead["id"], json.dumps(features)),
                )
                scored.append({"id": lead["id"], "customer_name": lead["customer_name"], "score": round(score, 2)})
            conn.commit()
        result = {"message": f"Scored {len(scored)} leads.", "scored_leads": scored}
        return self._log("lead_score_tool", {"lead_id": lead_id}, result)

    def sales_rag_tool(self, query: str) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT module, path, tags FROM documents
                WHERE module = 'sales' AND (lower(tags) LIKE ? OR lower(path) LIKE ?)
                """,
                (f"%{query.lower()}%", f"%{query.lower()}%"),
            )
            docs = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """
                SELECT term, definition, module FROM glossary
                WHERE module = 'sales' AND lower(term) LIKE ?
                """,
                (f"%{query.lower()}%",),
            )
            glossary = [dict(row) for row in cursor.fetchall()]
        message = f"Sales knowledge results: {len(docs)} documents, {len(glossary)} glossary matches."
        return self._log("sales_rag_tool", {"query": query}, {"message": message, "documents": docs, "glossary": glossary})

    def handle(self, message: str) -> dict[str, Any]:
        lowered = message.lower().strip()
        if lowered.startswith("create lead") and looks_like_json_payload(message):
            return self.create_lead_tool(extract_json_payload(message))
        if lowered.startswith("create order") and looks_like_json_payload(message):
            return self.create_order_tool(extract_json_payload(message))
        if lowered.startswith("update lead") and looks_like_json_payload(message):
            return self.update_lead_tool(extract_json_payload(message))
        if lowered.startswith("create ticket") and looks_like_json_payload(message):
            return self.create_ticket_tool(extract_json_payload(message))
        planned = self._llm_plan(message)
        if planned:
            planned_result = self._dispatch_planned_action(planned, message)
            if planned_result is not None:
                return planned_result
        if "score leads" in lowered or "lead score" in lowered:
            return self.lead_score_tool()
        if any(keyword in lowered for keyword in ["playbook", "sales docs", "crm", "manual", "history", "follow-up"]):
            context = self.sales_rag_tool(message)
            memory_bits: list[str] = []
            for customer_id in (1, 2):
                customer_memory = self.entity_memory.get_customer_info(customer_id)
                if customer_memory:
                    memory_bits.append(f"customer {customer_id}: {customer_memory}")
            memory_text = f"\n\nEntity memory: {'; '.join(memory_bits)}" if memory_bits else ""
            return {"message": f"{context['message']}{memory_text}", "rows": context.get("documents", [])}
        return self.sales_query_tool(message)

    def _llm_plan(self, message: str) -> dict[str, Any] | None:
        if not has_llm_credentials():
            return None

        prompt = (
            "You are the sales planning brain for an ERP agent.\n"
            "Choose exactly one action and return only JSON.\n"
            "Allowed actions:\n"
            "- create_lead: payload needs customer_name, contact_email, message\n"
            "- create_order: payload needs customer_id, status, items[{product_id, quantity, price?}]\n"
            "- update_lead: payload needs lead_id, status\n"
            "- create_ticket: payload needs customer_id, subject, body, status?\n"
            "- lead_score: payload may be empty\n"
            "- sales_rag: payload needs query\n"
            "- sales_query: payload may be empty\n"
            "If the user asks to add or create a lead, choose create_lead.\n"
            "If fields are missing, do not invent numeric IDs.\n"
            f"User message: {message}"
        )
        planned = parse_llm_json_object(coerce_text(get_llm().invoke(prompt)))
        if planned:
            self.state_store.log_tool_call("sales", "sales_llm_plan", {"message": message}, planned)
        return planned

    def _dispatch_planned_action(self, planned: dict[str, Any], message: str) -> dict[str, Any] | None:
        action = str(planned.get("action", "")).strip().lower()
        payload = planned.get("payload") or {}

        if action == "create_lead" and {"customer_name", "contact_email", "message"} <= set(payload):
            return self.create_lead_tool(payload)
        if action == "create_order" and {"customer_id", "items"} <= set(payload):
            return self.create_order_tool(payload)
        if action == "update_lead" and {"lead_id", "status"} <= set(payload):
            return self.update_lead_tool(payload)
        if action == "create_ticket" and {"customer_id", "subject", "body"} <= set(payload):
            return self.create_ticket_tool(payload)
        if action == "lead_score":
            return self.lead_score_tool(payload.get("lead_id"))
        if action == "sales_rag":
            query = payload.get("query") or message
            return self.sales_rag_tool(query)
        if action == "sales_query":
            return self.sales_query_tool(message)
        return None

    def _score_lead_payload(self, payload: dict[str, Any], features: dict[str, Any]) -> float:
        score = 5.0
        message = payload.get("message", "").lower()
        email = payload.get("contact_email", "").lower()
        if "urgent" in message:
            score += 1.5
        if "demo" in message:
            score += 1.0
        if "pricing" in message:
            score += 0.75
        if email.endswith((".gmail.com", ".yahoo.com", ".hotmail.com", ".outlook.com")):
            score -= 0.5
        else:
            score += 0.5
        score += min(features["message_length"] / 100.0, 1.0)
        return max(1.0, min(10.0, score))
