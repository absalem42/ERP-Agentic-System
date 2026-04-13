from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from backend.db import get_db
from backend.agents.AnalyticsAgent import create_analytics_agent
from backend.agents.FinanceAgent import create_finance_agent
from backend.agents.InventoryAgent import create_inventory_agent
from backend.agents.SalesAgent import create_sales_agent_with_chat
from backend.agents.simple_router_agent import create_simple_router_agent
from backend.config.llm import get_provider_mode
from backend.memory.base_memory import AnalyticsReportMemory, RouterGlobalState, SalesEntityMemory
from backend.mcp.mcp_adapter import mcp_registry
from backend.mcp.tool_registry import ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_DB = PROJECT_ROOT / "databases" / "erp_sample.db"
DEFAULT_RUNTIME_DB = Path(tempfile.gettempdir()) / "erp_system_demo" / "erp_runtime.db"


def ensure_runtime_schema(db_path: Path | str) -> None:
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS vendors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS vendor_bills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_id INTEGER NOT NULL,
                bill_number TEXT,
                issue_date DATE,
                due_date DATE,
                total_amount REAL,
                status TEXT DEFAULT 'unpaid',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (vendor_id) REFERENCES vendors(id)
            );

            CREATE TABLE IF NOT EXISTS vendor_bill_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_bill_id INTEGER NOT NULL,
                description TEXT,
                quantity INTEGER,
                unit_price REAL,
                FOREIGN KEY (vendor_bill_id) REFERENCES vendor_bills(id)
            );

            CREATE TABLE IF NOT EXISTS vendor_bill_payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_id INTEGER,
                amount REAL,
                method TEXT,
                paid_at DATETIME,
                FOREIGN KEY (vendor_id) REFERENCES vendors(id)
            );

            CREATE TABLE IF NOT EXISTS vendor_bill_allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payment_id INTEGER,
                vendor_bill_id INTEGER,
                amount REAL,
                FOREIGN KEY (payment_id) REFERENCES vendor_bill_payments(id),
                FOREIGN KEY (vendor_bill_id) REFERENCES vendor_bills(id)
            );
            """
        )
        conn.commit()


def resolve_app_version() -> str:
    env_version = os.getenv("ERP_APP_VERSION") or os.getenv("GIT_COMMIT")
    if env_version:
        return env_version[:12]
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "dev"


def prepare_runtime_db(
    sample_db: Path | str | None = None,
    runtime_db: Path | str | None = None,
) -> Path:
    sample_db_path = Path(sample_db or os.getenv("ERP_SAMPLE_DB_PATH") or DEFAULT_SAMPLE_DB)
    runtime_db_path = Path(runtime_db or os.getenv("DB_PATH") or DEFAULT_RUNTIME_DB)
    runtime_db_path.parent.mkdir(parents=True, exist_ok=True)

    if not runtime_db_path.exists():
        shutil.copy2(sample_db_path, runtime_db_path)

    ensure_runtime_schema(runtime_db_path)
    os.environ["DB_PATH"] = str(runtime_db_path)
    return runtime_db_path


class DirectERPService:
    def __init__(
        self,
        sample_db: Path | str | None = None,
        runtime_db: Path | str | None = None,
    ):
        self.runtime_db = prepare_runtime_db(sample_db=sample_db, runtime_db=runtime_db)
        mcp_registry.clear()
        self.state_store = RouterGlobalState(str(self.runtime_db))
        self.entity_memory = SalesEntityMemory(str(self.runtime_db))
        self.report_memory = AnalyticsReportMemory(str(self.runtime_db))
        self.registry = ToolRegistry()

        self.sales_agent = create_sales_agent_with_chat(
            state_store=self.state_store,
            entity_memory=self.entity_memory,
            registry=self.registry,
            db_path=str(self.runtime_db),
        )
        self.finance_agent = create_finance_agent(
            state_store=self.state_store,
            registry=self.registry,
            db_path=str(self.runtime_db),
        )
        self.inventory_agent = create_inventory_agent(
            state_store=self.state_store,
            registry=self.registry,
            db_path=str(self.runtime_db),
        )
        self.analytics_agent = create_analytics_agent(
            state_store=self.state_store,
            report_memory=self.report_memory,
            registry=self.registry,
            db_path=str(self.runtime_db),
        )
        self.router_agent = create_simple_router_agent(
            state_store=self.state_store,
            registry=self.registry,
            sales_agent=self.sales_agent,
            finance_agent=self.finance_agent,
            inventory_agent=self.inventory_agent,
            analytics_agent=self.analytics_agent,
            system_info_provider=self._system_info,
        )

    def get_health(self) -> dict:
        with get_db(self.runtime_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM customers")
            customer_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM approvals WHERE status = 'pending'")
            pending_approvals = cursor.fetchone()[0]

        return {
            "status": "healthy",
            "database": "connected",
            "database_path": str(self.runtime_db),
            "app_version": resolve_app_version(),
            "customer_count": customer_count,
            "pending_approvals": pending_approvals,
            "provider_mode": get_provider_mode(),
            "agents": {
                "router": "available",
                "sales": "available",
                "finance": "available",
                "inventory": "available",
                "analytics": "available",
            },
        }

    def list_agents(self) -> list[dict]:
        return [
            {"name": "router", "description": "Orchestrates routing, policy checks, and audit state."},
            {"name": "sales", "description": "Handles customers, leads, orders, tickets, and sales memory."},
            {"name": "finance", "description": "Posts invoices, payments, journals, and approval-aware finance actions."},
            {"name": "inventory", "description": "Handles stock, suppliers, purchase orders, receipts, and forecasts."},
            {"name": "analytics", "description": "Runs read-only analytics, saved reports, and chart specifications."},
        ]

    def list_approvals(self, status: str | None = None) -> list[dict]:
        return self.state_store.list_approvals(status=status)

    def approve_approval(self, approval_id: int, decided_by: str = "system") -> dict | None:
        approval = self.state_store.resolve_approval(approval_id, "approved", decided_by)
        if approval is None:
            return None

        execution_result = None
        if approval["module"] == "finance":
            execution_result = self.finance_agent.tools.execute_approved_payload(
                approval["payload_json"],
                requested_by=decided_by,
            )
        elif approval["module"] == "inventory":
            execution_result = self.inventory_agent.tools.execute_approved_payload(
                approval["payload_json"],
                requested_by=decided_by,
            )

        if execution_result is not None:
            approval["execution_result"] = execution_result
        return approval

    def reject_approval(self, approval_id: int, decided_by: str = "system") -> dict | None:
        return self.state_store.resolve_approval(approval_id, "rejected", decided_by)

    def list_tool_calls(self, limit: int = 50) -> list[dict]:
        return self.state_store.list_tool_calls(limit=limit)

    def list_saved_reports(self) -> list[dict]:
        return self.report_memory.list_reports()

    def run_saved_report(
        self,
        title: str,
        *,
        user_id: int | str = 1,
        session_id: str | None = None,
    ) -> dict:
        return self.chat(f"run report {title}", "analytics", user_id=user_id, session_id=session_id)

    def chat(
        self,
        message: str,
        agent: str = "router",
        *,
        user_id: int | str = 1,
        session_id: str | None = None,
    ) -> dict:
        started = time.time()
        conversation_id = self.state_store.get_or_create_conversation(
            user_id=user_id,
            session_id=session_id,
            agent_type=agent or "router",
        )
        baseline_tool_id = self.state_store.last_tool_call_id()
        self.state_store.add_message(conversation_id, "user", message, role="user")

        payload = {"input": message, "user_id": user_id, "conversation_id": conversation_id}
        selected_agent = (agent or "router").lower()
        if selected_agent == "sales":
            agent_result = self.sales_agent.invoke(payload)
            agent_used = "sales"
        elif selected_agent == "finance":
            agent_result = self.finance_agent.invoke(payload)
            agent_used = "finance"
        elif selected_agent == "inventory":
            agent_result = self.inventory_agent.invoke(payload)
            agent_used = "inventory"
        elif selected_agent == "analytics":
            agent_result = self.analytics_agent.invoke(payload)
            agent_used = "analytics"
        else:
            agent_result = self.router_agent.invoke(payload)
            agent_used = agent_result.get("agent_used", "router")

        response_text = agent_result["output"]
        self.state_store.add_message(conversation_id, agent_used, response_text, role="assistant")
        tool_calls = self.state_store.list_tool_calls(limit=50, since_id=baseline_tool_id)
        return {
            "response": response_text,
            "agent_used": agent_used,
            "tool_calls": tool_calls,
            "approval_required": agent_result.get("approval_required"),
            "chart_spec": agent_result.get("chart_spec"),
            "rows": agent_result.get("rows"),
            "execution_time": round(time.time() - started, 4),
        }

    def _system_info(self) -> str:
        health = self.get_health()
        return (
            "ERP system status:\n"
            f"- provider_mode: {health['provider_mode']}\n"
            f"- customers: {health['customer_count']}\n"
            f"- pending_approvals: {health['pending_approvals']}\n"
            "- agents: router, sales, finance, inventory, analytics"
        )


@lru_cache(maxsize=1)
def get_direct_service() -> DirectERPService:
    return DirectERPService()
