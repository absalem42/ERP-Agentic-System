import json
import sqlite3
from datetime import datetime


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, prompt: str, **kwargs):
        if not self.responses:
            raise AssertionError(f"Unexpected LLM prompt with no stubbed response left: {prompt}")

        class Response:
            def __init__(self, content):
                self.content = content

        return Response(self.responses.pop(0))


def test_prepare_runtime_db_copies_sample_database(runtime_paths, monkeypatch):
    from backend import runtime

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("DB_PATH", raising=False)

    resolved = runtime.prepare_runtime_db(sample_db=sample_db, runtime_db=runtime_db)

    assert resolved == runtime_db
    assert runtime_db.exists()
    assert runtime_db.read_bytes() == sample_db.read_bytes()
    assert runtime.os.environ["DB_PATH"].replace("\\", "/") == runtime_db.as_posix()


def test_router_records_conversation_and_tool_calls_for_sales_query(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat("show customers", "router", user_id=1, session_id="session-a")

    assert result["agent_used"] == "sales"
    assert "Acme Corp" in result["response"]
    assert result["tool_calls"]
    assert result["approval_required"] is None

    conn = sqlite3.connect(runtime_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversations WHERE session_id = ?", ("session-a",))
    assert cursor.fetchone()[0] == 1
    cursor.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = 1")
    assert cursor.fetchone()[0] == 2
    cursor.execute("SELECT tool_name FROM tool_calls ORDER BY id DESC LIMIT 1")
    assert cursor.fetchone()[0] == "sales_query_tool"
    conn.close()


def test_router_requires_approval_for_large_finance_invoice(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    payload = {
        "customer_id": 1,
        "order_id": 1,
        "issue_date": "2025-03-01",
        "due_date": "2025-03-15",
        "lines": [{"description": "Large machinery", "quantity": 1, "unit_price": 25000.0}],
    }

    result = service.chat(f"post invoice {json.dumps(payload)}", "router", user_id=1, session_id="session-b")

    assert result["agent_used"] == "finance"
    assert result["approval_required"] is not None
    assert result["approval_required"]["status"] == "pending"
    assert "approval" in result["response"].lower()


def test_router_uses_llm_classification_for_non_keyword_message(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService
    import backend.tools.router_tools as router_tools

    sample_db, runtime_db = runtime_paths
    monkeypatch.setattr(router_tools, "has_llm_credentials", lambda: True)
    monkeypatch.setattr(router_tools, "get_llm", lambda: FakeLLM(['{"label":"inventory"}']))

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat(
        "Which products look close to needing replenishment soon?",
        "router",
        user_id=1,
        session_id="router-llm-1",
    )

    assert result["agent_used"] == "inventory"


def test_sales_agent_can_create_lead_order_and_ticket_and_update_customer_memory(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)

    lead_payload = {
        "customer_name": "New Horizon",
        "contact_email": "sales@newhorizon.example",
        "message": "Urgent demo request and pricing",
    }
    lead_result = service.chat(f"create lead {json.dumps(lead_payload)}", "sales", user_id=1, session_id="sales-1")
    assert "lead created" in lead_result["response"].lower()

    order_payload = {
        "customer_id": 1,
        "status": "paid",
        "items": [{"product_id": 1, "quantity": 2}, {"product_id": 2, "quantity": 1}],
    }
    order_result = service.chat(f"create order {json.dumps(order_payload)}", "sales", user_id=1, session_id="sales-1")
    assert "order created" in order_result["response"].lower()

    ticket_payload = {"customer_id": 1, "subject": "Need invoice copy", "body": "Please resend invoice."}
    ticket_result = service.chat(f"create ticket {json.dumps(ticket_payload)}", "sales", user_id=1, session_id="sales-1")
    assert "ticket created" in ticket_result["response"].lower()

    conn = sqlite3.connect(runtime_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM leads WHERE customer_name = 'New Horizon'")
    assert cursor.fetchone()[0] == 1
    cursor.execute("SELECT COUNT(*) FROM orders WHERE customer_id = 1")
    assert cursor.fetchone()[0] == 2
    cursor.execute("SELECT COUNT(*) FROM tickets WHERE subject = 'Need invoice copy'")
    assert cursor.fetchone()[0] == 1
    cursor.execute(
        "SELECT value FROM customer_kv WHERE customer_id = 1 AND key = 'last_order_date'"
    )
    assert cursor.fetchone()[0]
    conn.close()


def test_sales_agent_uses_llm_for_natural_language_lead_creation(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService
    import backend.tools.sales_tools as sales_tools

    sample_db, runtime_db = runtime_paths
    monkeypatch.setattr(sales_tools, "has_llm_credentials", lambda: True, raising=False)
    monkeypatch.setattr(
        sales_tools,
        "get_llm",
        lambda: FakeLLM(
            [
                json.dumps(
                    {
                        "action": "create_lead",
                        "payload": {
                            "customer_name": "New Horizon",
                            "contact_email": "sales@newhorizon.example",
                            "message": "Urgent demo and pricing request",
                        },
                    }
                )
            ]
        ),
        raising=False,
    )

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat(
        "Please add a new lead for New Horizon. Email sales@newhorizon.example. They want an urgent demo and pricing.",
        "sales",
        user_id=1,
        session_id="sales-llm-1",
    )

    assert "lead created" in result["response"].lower()

    conn = sqlite3.connect(runtime_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM leads WHERE customer_name = 'New Horizon'")
    assert cursor.fetchone()[0] == 1
    conn.close()


def test_finance_agent_posts_invoice_allocates_payment_and_balances_ledger(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    invoice_payload = {
        "customer_id": 1,
        "order_id": 1,
        "issue_date": "2025-03-05",
        "due_date": "2025-03-20",
        "lines": [{"description": "Widget Pro", "quantity": 2, "unit_price": 100.0}],
    }

    invoice_result = service.chat(f"post invoice {json.dumps(invoice_payload)}", "finance", user_id=1, session_id="fin-1")
    assert "invoice posted" in invoice_result["response"].lower()

    payment_payload = {
        "customer_id": 1,
        "invoice_id": 1,
        "amount": 200.0,
        "method": "bank_transfer",
        "received_at": "2025-03-06 10:00:00",
    }
    payment_result = service.chat(f"record payment {json.dumps(payment_payload)}", "finance", user_id=1, session_id="fin-1")
    assert "payment recorded" in payment_result["response"].lower()

    conn = sqlite3.connect(runtime_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM invoices")
    assert cursor.fetchone()[0] == 1
    cursor.execute("SELECT COUNT(*) FROM payment_allocations")
    assert cursor.fetchone()[0] == 1
    cursor.execute(
        "SELECT ROUND(SUM(debit), 2), ROUND(SUM(credit), 2) FROM ledger_lines"
    )
    debit_total, credit_total = cursor.fetchone()
    assert debit_total == credit_total == 400.0
    conn.close()


def test_finance_agent_rejects_unknown_account_in_manual_journal(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    journal_payload = {
        "entry_date": "2025-03-07",
        "lines": [
            {"account": "Cash", "debit": 50.0, "credit": 0.0},
            {"account": "Unknown Account", "debit": 0.0, "credit": 50.0},
        ],
    }

    result = service.chat(f"post journal {json.dumps(journal_payload)}", "finance", user_id=1, session_id="fin-2")

    assert "unknown account" in result["response"].lower()


def test_finance_agent_executes_approved_invoice_after_approval(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    payload = {
        "customer_id": 1,
        "order_id": 1,
        "issue_date": "2025-03-01",
        "due_date": "2025-03-15",
        "lines": [{"description": "Large machinery", "quantity": 1, "unit_price": 25000.0}],
    }

    result = service.chat(f"post invoice {json.dumps(payload)}", "finance", user_id=1, session_id="fin-approve-1")
    approval_id = result["approval_required"]["id"]

    approved = service.approve_approval(approval_id, decided_by="tester")

    assert approved is not None
    assert approved["status"] == "approved"

    conn = sqlite3.connect(runtime_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM invoices")
    assert cursor.fetchone()[0] == 1
    conn.close()


def test_finance_agent_approval_executes_for_llm_invoice_without_issue_date(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService
    import backend.tools.finance_tools as finance_tools

    sample_db, runtime_db = runtime_paths
    monkeypatch.setattr(finance_tools, "has_llm_credentials", lambda: True, raising=False)
    monkeypatch.setattr(
        finance_tools,
        "get_llm",
        lambda: FakeLLM(
            [
                json.dumps(
                    {
                        "action": "post_invoice",
                        "payload": {
                            "customer_id": 1,
                            "order_id": 1,
                            "due_date": "2025-03-15",
                            "lines": [
                                {
                                    "description": "Large machinery charge",
                                    "quantity": 1,
                                    "unit_price": 25000.0,
                                }
                            ],
                        },
                    }
                )
            ]
        ),
        raising=False,
    )

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat(
        "Post a new invoice for customer 1 linked to order 1 for a large machinery charge of 25000 AED due on 2025-03-15.",
        "finance",
        user_id=1,
        session_id="fin-approve-llm-1",
    )
    approval_id = result["approval_required"]["id"]

    approved = service.approve_approval(approval_id, decided_by="tester")

    assert approved is not None
    assert approved["status"] == "approved"

    conn = sqlite3.connect(runtime_db)
    cursor = conn.cursor()
    cursor.execute("SELECT issue_date FROM invoices LIMIT 1")
    assert cursor.fetchone()[0]
    conn.close()


def test_finance_agent_rejects_llm_invoice_with_unknown_customer_reference(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService
    import backend.tools.finance_tools as finance_tools

    sample_db, runtime_db = runtime_paths
    monkeypatch.setattr(finance_tools, "has_llm_credentials", lambda: True, raising=False)
    monkeypatch.setattr(
        finance_tools,
        "get_llm",
        lambda: FakeLLM(
            [
                json.dumps(
                    {
                        "action": "post_invoice",
                        "payload": {
                            "customer_id": "new_vendor",
                            "lines": [
                                {
                                    "description": "Goods/Services",
                                    "quantity": 1,
                                    "unit_price": 15000.0,
                                }
                            ],
                        },
                    }
                )
            ]
        ),
        raising=False,
    )

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat(
        "Post an invoice for a new vendor for 15000 AED",
        "finance",
        user_id=1,
        session_id="fin-invalid-customer-1",
    )

    assert "customer invoices only" in result["response"].lower()
    assert result["approval_required"] is None
    assert service.list_approvals() == []


def test_finance_agent_approved_invalid_customer_payload_returns_safe_error(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    approval = service.state_store.create_approval(
        "finance",
        {
            "action": "post_invoice",
            "payload": {
                "customer_id": "new_vendor",
                "lines": [{"description": "Goods/Services", "quantity": 1, "unit_price": 15000.0}],
            },
        },
        requested_by="tester",
    )

    approved = service.approve_approval(approval["id"], decided_by="tester")

    assert approved is not None
    assert approved["status"] == "approved"
    assert "unknown customer" in approved["execution_result"]["message"].lower()

    conn = sqlite3.connect(runtime_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM invoices")
    assert cursor.fetchone()[0] == 0
    conn.close()


def test_inventory_agent_updates_stock_creates_po_and_receives_items(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    adjust_payload = {"product_id": 1, "change_qty": -4, "reason": "sale", "ref_id": 99}
    adjust_result = service.chat(f"adjust stock {json.dumps(adjust_payload)}", "inventory", user_id=1, session_id="inv-1")
    assert "stock updated" in adjust_result["response"].lower()

    po_payload = {"product_id": 2, "quantity": 10}
    po_result = service.chat(f"create purchase order {json.dumps(po_payload)}", "inventory", user_id=1, session_id="inv-1")
    assert "purchase order created" in po_result["response"].lower()

    receive_payload = {"po_id": 1, "product_id": 2, "received_qty": 10, "received_at": "2025-03-08 09:00:00"}
    receive_result = service.chat(f"receive purchase order {json.dumps(receive_payload)}", "inventory", user_id=1, session_id="inv-1")
    assert "receipt recorded" in receive_result["response"].lower()

    conn = sqlite3.connect(runtime_db)
    cursor = conn.cursor()
    cursor.execute("SELECT qty_on_hand FROM stock WHERE product_id = 1")
    assert cursor.fetchone()[0] == 8
    cursor.execute("SELECT qty_on_hand FROM stock WHERE product_id = 2")
    assert cursor.fetchone()[0] == 12
    cursor.execute("SELECT COUNT(*) FROM purchase_orders")
    assert cursor.fetchone()[0] == 1
    cursor.execute("SELECT status FROM purchase_orders WHERE id = 1")
    assert cursor.fetchone()[0] == "received"
    conn.close()


def test_inventory_agent_uses_llm_for_natural_language_purchase_order(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService
    import backend.tools.inventory_tools as inventory_tools

    sample_db, runtime_db = runtime_paths
    monkeypatch.setattr(inventory_tools, "has_llm_credentials", lambda: True, raising=False)
    monkeypatch.setattr(
        inventory_tools,
        "get_llm",
        lambda: FakeLLM(
            [
                json.dumps(
                    {
                        "action": "create_purchase_order",
                        "payload": {"product_id": 2, "quantity": 10},
                    }
                )
            ]
        ),
        raising=False,
    )

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat(
        "Please reorder 10 units of product 2 from the best supplier.",
        "inventory",
        user_id=1,
        session_id="inventory-llm-1",
    )

    assert "purchase order created" in result["response"].lower()

    conn = sqlite3.connect(runtime_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM purchase_orders")
    assert cursor.fetchone()[0] == 1
    conn.close()


def test_analytics_agent_enforces_read_only_sql_and_runs_saved_reports(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)

    reject_result = service.chat("run sql DELETE FROM orders", "analytics", user_id=1, session_id="ana-1")
    assert "read-only" in reject_result["response"].lower()

    save_payload = {
        "title": "monthly revenue",
        "sql": "SELECT strftime('%Y-%m', created_at) AS period, ROUND(SUM(total), 2) AS revenue FROM orders GROUP BY strftime('%Y-%m', created_at) ORDER BY period",
    }
    save_result = service.chat(f"save report {json.dumps(save_payload)}", "analytics", user_id=1, session_id="ana-1")
    assert "saved report" in save_result["response"].lower()

    report_result = service.chat("run report monthly revenue", "analytics", user_id=1, session_id="ana-1")
    assert "monthly revenue" in report_result["response"].lower()
    assert report_result["chart_spec"] is not None
    assert report_result["chart_spec"]["type"] == "bar"


def test_router_handles_revenue_trend_prompt_with_analytics(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat(
        "Show me this month revenue trend",
        "router",
        user_id=1,
        session_id="ana-trend-1",
    )

    assert result["agent_used"] == "analytics"
    assert "no sql mapping" not in result["response"].lower()
    assert result["rows"] == []
    assert result["chart_spec"] is None
    expected_month = datetime.now().strftime("%Y-%m")
    latest_period = "2025-02"
    assert f"no order revenue data is available for {expected_month}" in result["response"].lower()
    assert f"latest available revenue data is from {latest_period}" in result["response"].lower()


def test_analytics_agent_returns_top_products_by_revenue_with_context(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat(
        "What are the top 5 products by revenue and why?",
        "analytics",
        user_id=1,
        session_id="ana-top-products-1",
    )

    assert "read-only" not in result["response"].lower()
    assert len(result["rows"]) == 2
    assert set(result["rows"][0]) >= {"product_name", "revenue"}
    assert result["chart_spec"] is not None
    assert result["chart_spec"]["type"] == "bar"
    assert "based on fulfilled order line revenue" in result["response"].lower()
    assert "context:" in result["response"].lower()


def test_analytics_agent_normalizes_fenced_llm_sql_before_validation(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService
    import backend.tools.analytics_tools as analytics_tools

    sample_db, runtime_db = runtime_paths
    monkeypatch.setattr(analytics_tools, "has_llm_credentials", lambda: True, raising=False)
    monkeypatch.setattr(
        analytics_tools,
        "get_llm",
        lambda: FakeLLM(
            [
                "```sql\nSELECT status, COUNT(*) AS order_count FROM orders GROUP BY status ORDER BY status\n```",
            ]
        ),
        raising=False,
    )

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat(
        "Group orders by status for me",
        "analytics",
        user_id=1,
        session_id="ana-fenced-sql-1",
    )

    assert "read-only" not in result["response"].lower()
    assert result["rows"]
    assert result["tool_calls"][0]["output_json"]["sql"].startswith("SELECT status")


def test_finance_agent_rejects_unsupported_vendor_invoice_prompt_without_llm(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat(
        "Post an invoice for a new vendor for 15000 AED",
        "finance",
        user_id=1,
        session_id="fin-unsupported-vendor-1",
    )

    assert "current finance workflow supports customer invoices" in result["response"].lower()
    assert "existing customer_id" in result["response"].lower()
    assert result["approval_required"] is None


def test_api_smoke_endpoints_expose_agents_approvals_and_audit(runtime_paths, monkeypatch):
    from fastapi.testclient import TestClient

    sample_db, runtime_db = runtime_paths
    monkeypatch.setenv("DB_PATH", str(runtime_db))
    monkeypatch.setenv("ERP_SAMPLE_DB_PATH", str(sample_db))
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    from backend.api import app

    client = TestClient(app)

    chat_response = client.post(
        "/chat",
        json={"message": "show customers", "agent": "router", "user_id": 1, "session_id": "api-1"},
    )
    assert chat_response.status_code == 200
    assert chat_response.json()["agent_used"] == "sales"

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["agents"]["finance"] == "available"

    approvals_response = client.get("/approvals")
    assert approvals_response.status_code == 200
    assert "approvals" in approvals_response.json()

    audit_response = client.get("/audit/tool-calls")
    assert audit_response.status_code == 200
    assert audit_response.json()["tool_calls"]
