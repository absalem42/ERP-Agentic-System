from __future__ import annotations

from datetime import datetime, timezone
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
            name="post_vendor_bill_tool",
            handler=self.post_vendor_bill_tool,
            description="Post a vendor/AP bill and supporting line items",
            input_schema={"vendor_id": "int", "lines": "list[dict]", "due_date": "str | None"},
            module="finance",
            read_only=False,
            requires_approval=True,
        )
        self.registry.register_tool(
            name="record_vendor_payment_tool",
            handler=self.record_vendor_payment_tool,
            description="Record a vendor bill payment and allocation",
            input_schema={"vendor_id": "int", "vendor_bill_id": "int", "amount": "float", "method": "str"},
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
        self.registry.register_tool(
            name="finance_query_tool",
            handler=self.finance_query_tool,
            description="Run finance read queries such as trial balance and payables summaries",
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

    def post_vendor_bill_tool(self, payload: dict[str, Any], requested_by: str = "system", approved: bool = False) -> dict[str, Any]:
        payload = self._normalize_vendor_bill_payload(payload)
        reference_error = self._validate_vendor_references(payload)
        if reference_error:
            result = {"message": reference_error}
            return self._log("post_vendor_bill_tool", payload, result)

        anomaly = self.anomaly_detector_tool(payload)
        total_amount = anomaly["total_amount"]
        if anomaly["is_risky"] and not approved:
            approval = self.state_store.create_approval(
                "finance",
                {"action": "post_vendor_bill", "payload": payload},
                requested_by,
            )
            result = {
                "message": "Vendor bill requires approval before posting.",
                "approval_required": approval,
                "total_amount": total_amount,
            }
            return self._log("post_vendor_bill_tool", payload, result)

        bill_number = payload.get("bill_number") or self._next_vendor_bill_number()
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO vendor_bills (vendor_id, bill_number, issue_date, due_date, total_amount, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    payload["vendor_id"],
                    bill_number,
                    payload.get("issue_date"),
                    payload.get("due_date"),
                    total_amount,
                    payload.get("status", "unpaid"),
                ),
            )
            vendor_bill_id = int(cursor.lastrowid)
            for line in payload.get("lines", []):
                cursor.execute(
                    """
                    INSERT INTO vendor_bill_lines (vendor_bill_id, description, quantity, unit_price)
                    VALUES (?, ?, ?, ?)
                    """,
                    (vendor_bill_id, line["description"], line["quantity"], line["unit_price"]),
                )
            conn.commit()

        expense_account = payload.get("expense_account") or "Office Expense"
        journal = self.post_journal_tool(
            {
                "entry_date": payload.get("issue_date"),
                "lines": [
                    {"account": expense_account, "debit": total_amount, "credit": 0.0},
                    {"account": "Accounts Payable", "debit": 0.0, "credit": total_amount},
                ],
            }
        )
        result = {
            "message": f"Vendor bill posted with id {vendor_bill_id} and number {bill_number}.",
            "vendor_bill_id": vendor_bill_id,
            "bill_number": bill_number,
            "total_amount": total_amount,
            "journal_entry_id": journal.get("entry_id"),
        }
        return self._log("post_vendor_bill_tool", payload, result)

    def record_vendor_payment_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO vendor_bill_payments (vendor_id, amount, method, paid_at)
                VALUES (?, ?, ?, ?)
                """,
                (payload["vendor_id"], payload["amount"], payload.get("method"), payload.get("paid_at")),
            )
            payment_id = int(cursor.lastrowid)
            cursor.execute(
                """
                INSERT INTO vendor_bill_allocations (payment_id, vendor_bill_id, amount)
                VALUES (?, ?, ?)
                """,
                (payment_id, payload["vendor_bill_id"], payload["amount"]),
            )
            cursor.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM vendor_bill_allocations WHERE vendor_bill_id = ?",
                (payload["vendor_bill_id"],),
            )
            allocated = float(cursor.fetchone()[0])
            cursor.execute("SELECT total_amount FROM vendor_bills WHERE id = ?", (payload["vendor_bill_id"],))
            bill_row = cursor.fetchone()
            bill_total = float(bill_row[0]) if bill_row and bill_row[0] is not None else 0.0
            status = "paid" if allocated >= bill_total else "partial"
            cursor.execute("UPDATE vendor_bills SET status = ? WHERE id = ?", (status, payload["vendor_bill_id"]))
            conn.commit()

        journal = self.post_journal_tool(
            {
                "entry_date": payload.get("paid_at", "")[:10],
                "lines": [
                    {"account": "Accounts Payable", "debit": float(payload["amount"]), "credit": 0.0},
                    {"account": "Cash", "debit": 0.0, "credit": float(payload["amount"])},
                ],
            }
        )
        result = {
            "message": f"Vendor payment recorded with id {payment_id}.",
            "payment_id": payment_id,
            "vendor_bill_status": status,
            "journal_entry_id": journal.get("entry_id"),
        }
        return self._log("record_vendor_payment_tool", payload, result)

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

    def finance_query_tool(self, query: str) -> dict[str, Any]:
        lowered = query.lower()
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            if "trial balance" in lowered or ("account" in lowered and "balance" in lowered):
                cursor.execute(
                    """
                    SELECT account,
                           ROUND(SUM(debit), 2) AS total_debit,
                           ROUND(SUM(credit), 2) AS total_credit
                    FROM ledger_lines
                    GROUP BY account
                    ORDER BY account
                    """
                )
                rows = [dict(row) for row in cursor.fetchall()]
                message = "Trial Balance"
            elif "payable" in lowered or "vendor bill" in lowered:
                cursor.execute(
                    """
                    SELECT vb.id, v.name AS vendor_name, vb.bill_number, vb.total_amount, vb.status, vb.due_date
                    FROM vendor_bills vb
                    JOIN vendors v ON v.id = vb.vendor_id
                    ORDER BY vb.id DESC
                    LIMIT 10
                    """
                )
                rows = [dict(row) for row in cursor.fetchall()]
                message = "Vendor Bills"
            else:
                rows = []
                message = "No finance read query matched."
        result = {"message": message, "rows": rows}
        return self._log("finance_query_tool", {"query": query}, result)

    def handle(self, message: str, requested_by: str = "system") -> dict[str, Any]:
        lowered = message.lower().strip()
        if lowered.startswith("post invoice") and looks_like_json_payload(message):
            return self.post_invoice_tool(extract_json_payload(message), requested_by=requested_by)
        if lowered.startswith("post vendor bill") and looks_like_json_payload(message):
            return self.post_vendor_bill_tool(extract_json_payload(message), requested_by=requested_by)
        if lowered.startswith("record payment") and looks_like_json_payload(message):
            return self.record_payment_tool(extract_json_payload(message))
        if lowered.startswith("record vendor payment") and looks_like_json_payload(message):
            return self.record_vendor_payment_tool(extract_json_payload(message))
        if lowered.startswith("post journal") and looks_like_json_payload(message):
            return self.post_journal_tool(extract_json_payload(message))
        if "trial balance" in lowered or ("account" in lowered and "balance" in lowered):
            return self.finance_query_tool(message)
        planned = self._llm_plan(message)
        if planned:
            planned_result = self._dispatch_planned_action(planned, message, requested_by)
            if planned_result is not None:
                return planned_result
        if "vendor" in lowered or "payable" in lowered:
            heuristic_vendor = self._heuristic_vendor_bill_plan(message)
            if heuristic_vendor is not None:
                return self.post_vendor_bill_tool(heuristic_vendor, requested_by=requested_by)
        return self.policy_rag_tool(message)

    def execute_approved_payload(self, approval_payload: dict[str, Any], requested_by: str = "system") -> dict[str, Any]:
        action = approval_payload.get("action")
        payload = approval_payload.get("payload") if "payload" in approval_payload else approval_payload
        if action == "post_invoice" or {"customer_id", "lines"} <= set(payload):
            return self.post_invoice_tool(payload, requested_by=requested_by, approved=True)
        if action == "post_vendor_bill" or {"vendor_id", "lines"} <= set(payload):
            return self.post_vendor_bill_tool(payload, requested_by=requested_by, approved=True)
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
                issue_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        normalized["issue_date"] = issue_date[:10] if isinstance(issue_date, str) else issue_date
        return normalized

    def _normalize_vendor_bill_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        raw_vendor_id = normalized.get("vendor_id")
        normalized["vendor_id"] = self._coerce_int(raw_vendor_id)
        if raw_vendor_id != normalized["vendor_id"]:
            normalized["_raw_vendor_id"] = raw_vendor_id
            if isinstance(raw_vendor_id, str) and raw_vendor_id.strip():
                normalized["vendor_name"] = raw_vendor_id.strip().replace("_", " ")
        issue_date = normalized.get("issue_date")
        if not issue_date:
            due_date = normalized.get("due_date")
            if isinstance(due_date, str) and due_date:
                issue_date = due_date[:10]
            else:
                issue_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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

    def _validate_vendor_references(self, payload: dict[str, Any]) -> str | None:
        payload = self._resolve_vendor_reference(payload)
        vendor_id = payload.get("vendor_id")
        if vendor_id is None:
            raw_vendor_id = payload.get("_raw_vendor_id")
            if raw_vendor_id is not None:
                return f"Unknown vendor reference: {raw_vendor_id}. Provide an existing vendor_id."
            return "Vendor bill payload must include a valid existing vendor_id."

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM vendors WHERE id = ?", (vendor_id,))
            if cursor.fetchone() is None:
                return f"Unknown vendor reference: {payload.get('vendor_id')}. Provide an existing vendor_id."

        return None

    def _resolve_vendor_reference(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("vendor_id") is not None:
            return payload

        vendor_name = payload.get("vendor_name") or payload.get("_raw_vendor_id")
        if not isinstance(vendor_name, str) or not vendor_name.strip():
            return payload

        normalized_name = vendor_name.strip().replace("_", " ")
        if normalized_name.lower() in {"new vendor", "vendor", "new"}:
            normalized_name = "New Vendor"

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM vendors WHERE lower(name) = lower(?)", (normalized_name,))
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    """
                    INSERT INTO vendors (name, email, phone, created_at)
                    VALUES (?, NULL, NULL, CURRENT_TIMESTAMP)
                    """,
                    (normalized_name,),
                )
                conn.commit()
                vendor_id = int(cursor.lastrowid)
            else:
                vendor_id = int(row[0])

        payload["vendor_id"] = vendor_id
        payload["vendor_name"] = normalized_name
        return payload

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
            "- post_vendor_bill: payload needs vendor_id, issue_date optional, due_date optional, lines[{description, quantity, unit_price}]\n"
            "- record_payment: payload needs customer_id, invoice_id, amount, method, received_at optional\n"
            "- record_vendor_payment: payload needs vendor_id, vendor_bill_id, amount, method, paid_at optional\n"
            "- post_journal: payload needs entry_date and lines[{account, debit, credit}]\n"
            "- finance_query: payload needs query for trial balance or payable summaries\n"
            "- policy_rag: payload needs query\n"
            "If the user asks to create or post a vendor bill or accounts payable invoice, choose post_vendor_bill.\n"
            "If the user asks to create or post an invoice for a customer, choose post_invoice.\n"
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
        if action == "post_vendor_bill":
            if payload.get("vendor_id") is None:
                recovered_payload = self._heuristic_vendor_bill_plan(message)
                if recovered_payload is not None:
                    return self.post_vendor_bill_tool(recovered_payload, requested_by=requested_by)
            if "lines" in payload:
                return self.post_vendor_bill_tool(payload, requested_by=requested_by)
        if action == "record_payment" and {"customer_id", "invoice_id", "amount"} <= set(payload):
            return self.record_payment_tool(payload)
        if action == "record_vendor_payment" and {"vendor_id", "vendor_bill_id", "amount"} <= set(payload):
            return self.record_vendor_payment_tool(payload)
        if action == "post_journal" and {"entry_date", "lines"} <= set(payload):
            return self.post_journal_tool(payload)
        if action == "finance_query":
            query = payload.get("query") or message
            return self.finance_query_tool(query)
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

    def _next_vendor_bill_number(self) -> str:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM vendor_bills")
            count = int(cursor.fetchone()[0]) + 1
        return f"BILL-{count:05d}"

    def _heuristic_vendor_bill_plan(self, message: str) -> dict[str, Any] | None:
        lowered = message.lower()
        if "vendor" not in lowered and "payable" not in lowered:
            return None
        amount_match = __import__("re").search(r"(\d+(?:\.\d+)?)", message)
        amount = float(amount_match.group(1)) if amount_match else 0.0
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM vendors ORDER BY id LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    """
                    INSERT INTO vendors (name, email, phone, created_at)
                    VALUES ('New Vendor', NULL, NULL, CURRENT_TIMESTAMP)
                    """
                )
                conn.commit()
                vendor_id = int(cursor.lastrowid)
            else:
                vendor_id = int(row[0])
        return {
            "vendor_id": vendor_id,
            "lines": [{"description": "Vendor bill", "quantity": 1, "unit_price": amount or self.HIGH_RISK_AMOUNT}],
        }
