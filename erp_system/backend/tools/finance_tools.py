from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.config.llm import coerce_text, get_llm, has_llm_credentials
from backend.db import get_db
from backend.memory.base_memory import RouterGlobalState
from backend.mcp.tool_registry import ToolRegistry
from backend.tools.common import extract_json_payload, looks_like_json_payload, parse_llm_json_object


class FinanceTools:
    HIGH_RISK_AMOUNT = 10000.0

    def __init__(self, state_store: RouterGlobalState, registry: ToolRegistry, db_path: str | None = None):
        self.state_store = state_store
        self.registry = registry
        self.db_path = db_path
        self.registry.register_tool(
            name="post_invoice_tool",
            handler=self.post_invoice_tool,
            description="Post a finance invoice and supporting invoice lines/order links",
            input_schema={"customer_id": "int", "order_id": "int | None", "lines": "list[dict]"},
            module="finance",
            read_only=False,
            requires_approval=True,
        )
        self.registry.register_tool(
            name="record_payment_tool",
            handler=self.record_payment_tool,
            description="Record a payment and payment allocation against an invoice",
            input_schema={"customer_id": "int", "invoice_id": "int", "amount": "float", "method": "str"},
            module="finance",
            read_only=False,
        )
        self.registry.register_tool(
            name="post_journal_tool",
            handler=self.post_journal_tool,
            description="Post a manual double-entry journal",
            input_schema={"entry_date": "str", "lines": "list[dict]"},
            module="finance",
            read_only=False,
        )
        self.registry.register_tool(
            name="anomaly_detector_tool",
            handler=self.anomaly_detector_tool,
            description="Check whether an invoice payload is risky",
            input_schema={"lines": "list[dict]"},
            module="finance",
        )
        self.registry.register_tool(
            name="policy_rag_tool",
            handler=self.policy_rag_tool,
            description="Retrieve finance policy and glossary context",
            input_schema={"query": "str"},
            module="finance",
        )

    def _log(self, tool_name: str, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        self.state_store.log_tool_call("finance", tool_name, payload, result)
        return result

    def anomaly_detector_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        total_amount = sum(float(line["quantity"]) * float(line["unit_price"]) for line in payload.get("lines", []))
        is_risky = total_amount >= self.HIGH_RISK_AMOUNT
        result = {"is_risky": is_risky, "total_amount": total_amount}
        return self._log("anomaly_detector_tool", payload, result)

    def post_invoice_tool(self, payload: dict[str, Any], requested_by: str = "system", approved: bool = False) -> dict[str, Any]:
        payload = self._normalize_invoice_payload(payload)
        reference_error = self._validate_invoice_references(payload)
        if reference_error:
            result = {"message": reference_error}
            return self._log("post_invoice_tool", payload, result)
        anomaly = self.anomaly_detector_tool(payload)
        total_amount = anomaly["total_amount"]
        if anomaly["is_risky"] and not approved:
            approval = self.state_store.create_approval(
                "finance",
                {"action": "post_invoice", "payload": payload},
                requested_by,
            )
            result = {
                "message": "Invoice requires approval before posting.",
                "approval_required": approval,
                "total_amount": total_amount,
            }
            return self._log("post_invoice_tool", payload, result)

        invoice_number = payload.get("invoice_number") or self._next_invoice_number()
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO invoices (customer_id, invoice_number, issue_date, due_date, total_amount, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    payload["customer_id"],
                    invoice_number,
                    payload.get("issue_date"),
                    payload.get("due_date"),
                    total_amount,
                    payload.get("status", "unpaid"),
                ),
            )
            invoice_id = int(cursor.lastrowid)
            for line in payload.get("lines", []):
                cursor.execute(
                    """
                    INSERT INTO invoice_lines (invoice_id, description, quantity, unit_price)
                    VALUES (?, ?, ?, ?)
                    """,
                    (invoice_id, line["description"], line["quantity"], line["unit_price"]),
                )
            if payload.get("order_id") is not None:
                cursor.execute(
                    "INSERT INTO invoice_orders (invoice_id, order_id) VALUES (?, ?)",
                    (invoice_id, payload["order_id"]),
                )
            conn.commit()

        journal = self.post_journal_tool(
            {
                "entry_date": payload.get("issue_date"),
                "lines": [
                    {"account": "Accounts Receivable", "debit": total_amount, "credit": 0.0},
                    {"account": "Revenue", "debit": 0.0, "credit": total_amount},
                ],
            }
        )
        result = {
            "message": f"Invoice posted with id {invoice_id} and number {invoice_number}.",
            "invoice_id": invoice_id,
            "invoice_number": invoice_number,
            "total_amount": total_amount,
            "journal_entry_id": journal.get("entry_id"),
        }
        return self._log("post_invoice_tool", payload, result)

    def record_payment_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO payments (customer_id, amount, method, received_at)
                VALUES (?, ?, ?, ?)
                """,
                (payload["customer_id"], payload["amount"], payload.get("method"), payload.get("received_at")),
            )
            payment_id = int(cursor.lastrowid)
            cursor.execute(
                """
                INSERT INTO payment_allocations (payment_id, invoice_id, amount)
                VALUES (?, ?, ?)
                """,
                (payment_id, payload["invoice_id"], payload["amount"]),
            )
            cursor.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM payment_allocations WHERE invoice_id = ?",
                (payload["invoice_id"],),
            )
            allocated = float(cursor.fetchone()[0])
            cursor.execute("SELECT total_amount FROM invoices WHERE id = ?", (payload["invoice_id"],))
            invoice_row = cursor.fetchone()
            invoice_total = float(invoice_row[0]) if invoice_row and invoice_row[0] is not None else 0.0
            status = "paid" if allocated >= invoice_total else "partial"
            cursor.execute("UPDATE invoices SET status = ? WHERE id = ?", (status, payload["invoice_id"]))
            conn.commit()

        journal = self.post_journal_tool(
            {
                "entry_date": payload.get("received_at", "")[:10],
                "lines": [
                    {"account": "Cash", "debit": float(payload["amount"]), "credit": 0.0},
                    {"account": "Accounts Receivable", "debit": 0.0, "credit": float(payload["amount"])},
                ],
            }
        )
        result = {
            "message": f"Payment recorded with id {payment_id}.",
            "payment_id": payment_id,
            "invoice_status": status,
            "journal_entry_id": journal.get("entry_id"),
        }
        return self._log("record_payment_tool", payload, result)

    def post_journal_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        line_items = payload.get("lines", [])
        if not line_items:
            result = {"message": "Journal requires at least one line."}
            return self._log("post_journal_tool", payload, result)
        accounts = {line["account"] for line in line_items}
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(accounts))
            cursor.execute(f"SELECT account FROM chart_of_accounts WHERE account IN ({placeholders})", tuple(accounts))
            known_accounts = {row[0] for row in cursor.fetchall()}
            missing_accounts = accounts - known_accounts
            if missing_accounts:
                result = {"message": f"Unknown account(s): {', '.join(sorted(missing_accounts))}."}
                return self._log("post_journal_tool", payload, result)

            debit_total = round(sum(float(line.get("debit", 0.0)) for line in line_items), 2)
            credit_total = round(sum(float(line.get("credit", 0.0)) for line in line_items), 2)
            if debit_total != credit_total:
                result = {"message": "Journal is not balanced."}
                return self._log("post_journal_tool", payload, result)

            cursor.execute(
                "INSERT INTO ledger_entries (entry_date, created_at) VALUES (?, CURRENT_TIMESTAMP)",
                (payload["entry_date"],),
            )
            entry_id = int(cursor.lastrowid)
            for line in line_items:
                cursor.execute(
                    """
                    INSERT INTO ledger_lines (entry_id, account, debit, credit)
                    VALUES (?, ?, ?, ?)
                    """,
                    (entry_id, line["account"], line.get("debit", 0.0), line.get("credit", 0.0)),
                )
            conn.commit()
        result = {"message": f"Journal entry {entry_id} posted.", "entry_id": entry_id}
        return self._log("post_journal_tool", payload, result)

    def policy_rag_tool(self, query: str) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT module, path, tags FROM documents
                WHERE module = 'finance' AND (lower(tags) LIKE ? OR lower(path) LIKE ?)
                """,
                (f"%{query.lower()}%", f"%{query.lower()}%"),
            )
            documents = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """
                SELECT term, definition, module FROM glossary
                WHERE module = 'finance' OR lower(term) LIKE ?
                """,
                (f"%{query.lower()}%",),
            )
            glossary = [dict(row) for row in cursor.fetchall()]
        result = {
            "message": f"Finance policy lookup returned {len(documents)} documents and {len(glossary)} glossary entries.",
            "documents": documents,
            "glossary": glossary,
        }
        return self._log("policy_rag_tool", {"query": query}, result)

    def handle(self, message: str, requested_by: str = "system") -> dict[str, Any]:
        lowered = message.lower().strip()
        if lowered.startswith("post invoice") and looks_like_json_payload(message):
            return self.post_invoice_tool(extract_json_payload(message), requested_by=requested_by)
        if lowered.startswith("record payment") and looks_like_json_payload(message):
            return self.record_payment_tool(extract_json_payload(message))
        if lowered.startswith("post journal") and looks_like_json_payload(message):
            return self.post_journal_tool(extract_json_payload(message))
        if "invoice" in lowered and "vendor" in lowered:
            unsupported = {
                "message": (
                    "The current finance workflow supports customer invoices only. "
                    "Vendor/AP invoices are not modeled in this schema yet. Use an existing customer_id "
                    "for billing workflows."
                )
            }
            return self._log(
                "unsupported_finance_intent",
                {"query": message, "unsupported_intent": "vendor_invoice"},
                unsupported,
            )
        planned = self._llm_plan(message)
        if planned:
            planned_result = self._dispatch_planned_action(planned, message, requested_by)
            if planned_result is not None:
                return planned_result
        return self.policy_rag_tool(message)

    def execute_approved_payload(self, approval_payload: dict[str, Any], requested_by: str = "system") -> dict[str, Any]:
        action = approval_payload.get("action")
        payload = approval_payload.get("payload") if "payload" in approval_payload else approval_payload
        if action == "post_invoice" or {"customer_id", "lines"} <= set(payload):
            return self.post_invoice_tool(payload, requested_by=requested_by, approved=True)
        return {"message": "No executable finance action found for approval."}

    def _normalize_invoice_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        raw_customer_id = normalized.get("customer_id")
        normalized["customer_id"] = self._coerce_int(raw_customer_id)
        if raw_customer_id != normalized["customer_id"]:
            normalized["_raw_customer_id"] = raw_customer_id
        if "order_id" in normalized:
            raw_order_id = normalized.get("order_id")
            normalized["order_id"] = self._coerce_int(raw_order_id)
            if raw_order_id != normalized["order_id"]:
                normalized["_raw_order_id"] = raw_order_id
        issue_date = normalized.get("issue_date") or normalized.get("entry_date")
        if not issue_date:
            due_date = normalized.get("due_date")
            if isinstance(due_date, str) and due_date:
                issue_date = due_date[:10]
            else:
                issue_date = datetime.utcnow().strftime("%Y-%m-%d")
        normalized["issue_date"] = issue_date[:10] if isinstance(issue_date, str) else issue_date
        return normalized

    def _validate_invoice_references(self, payload: dict[str, Any]) -> str | None:
        customer_id = payload.get("customer_id")
        if customer_id is None:
            raw_customer_id = payload.get("_raw_customer_id")
            if raw_customer_id is not None:
                return f"Unknown customer reference: {raw_customer_id}. Provide an existing customer_id."
            return "Invoice payload must include a valid existing customer_id."

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM customers WHERE id = ?", (customer_id,))
            if cursor.fetchone() is None:
                return f"Unknown customer reference: {payload.get('customer_id')}. Provide an existing customer_id."

            order_id = payload.get("order_id")
            if order_id is not None:
                cursor.execute("SELECT 1 FROM orders WHERE id = ?", (order_id,))
                if cursor.fetchone() is None:
                    raw_order_id = payload.get("_raw_order_id", payload.get("order_id"))
                    return f"Unknown order reference: {raw_order_id}. Provide an existing order_id."

        return None

    def _coerce_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
        return None

    def _llm_plan(self, message: str) -> dict[str, Any] | None:
        if not has_llm_credentials():
            return None

        prompt = (
            "You are the finance planning brain for an ERP agent.\n"
            "Choose exactly one action and return only JSON.\n"
            "Allowed actions:\n"
            "- post_invoice: payload needs customer_id, order_id optional, issue_date optional, due_date optional, lines[{description, quantity, unit_price}]\n"
            "- record_payment: payload needs customer_id, invoice_id, amount, method, received_at optional\n"
            "- post_journal: payload needs entry_date and lines[{account, debit, credit}]\n"
            "- policy_rag: payload needs query\n"
            "If the user asks to create or post an invoice, choose post_invoice.\n"
            "Return only JSON.\n"
            f"User message: {message}"
        )
        planned = parse_llm_json_object(coerce_text(get_llm().invoke(prompt)))
        if planned:
            self.state_store.log_tool_call("finance", "finance_llm_plan", {"message": message}, planned)
        return planned

    def _dispatch_planned_action(
        self,
        planned: dict[str, Any],
        message: str,
        requested_by: str,
    ) -> dict[str, Any] | None:
        action = str(planned.get("action", "")).strip().lower()
        payload = planned.get("payload") or {}
        if action == "post_invoice" and {"customer_id", "lines"} <= set(payload):
            return self.post_invoice_tool(payload, requested_by=requested_by)
        if action == "record_payment" and {"customer_id", "invoice_id", "amount"} <= set(payload):
            return self.record_payment_tool(payload)
        if action == "post_journal" and {"entry_date", "lines"} <= set(payload):
            return self.post_journal_tool(payload)
        if action == "policy_rag":
            query = payload.get("query") or message
            return self.policy_rag_tool(query)
        return None

    def _next_invoice_number(self) -> str:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM invoices")
            count = int(cursor.fetchone()[0]) + 1
        return f"INV-{count:05d}"
