import os
import shutil
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from .config.llm import has_llm_credentials
from .db import get_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_DB = PROJECT_ROOT / "databases" / "erp_sample.db"
DEFAULT_RUNTIME_DB = Path(tempfile.gettempdir()) / "erp_system_demo" / "erp_public_demo.db"


def _hosted_direct_ai_enabled() -> bool:
    """
    Hosted direct mode prioritizes responsiveness over full agent execution.
    Opt in explicitly with ERP_ENABLE_DIRECT_AI=1 if you want the Streamlit-hosted
    demo to call the full Groq-backed agent stack.
    """
    return os.getenv("ERP_ENABLE_DIRECT_AI", "").strip().lower() in {"1", "true", "yes", "on"}


def _ensure_sample_db_exists(sample_db: Path) -> Path:
    if sample_db.exists():
        return sample_db

    from create_sample_db import copy_if_exists, create_minimal_schema_and_seed

    if not copy_if_exists():
        create_minimal_schema_and_seed()

    return sample_db


def prepare_runtime_db(
    sample_db: Path | str = DEFAULT_SAMPLE_DB,
    runtime_db: Path | str = DEFAULT_RUNTIME_DB,
) -> Path:
    """Copy the demo database to a writable runtime path and export DB_PATH."""
    sample_db_path = _ensure_sample_db_exists(Path(sample_db))
    runtime_db_path = Path(runtime_db)
    runtime_db_path.parent.mkdir(parents=True, exist_ok=True)

    if not runtime_db_path.exists():
        shutil.copy2(sample_db_path, runtime_db_path)

    os.environ["DB_PATH"] = str(runtime_db_path)
    return runtime_db_path


class DirectERPService:
    """Direct runtime service used by hosted Streamlit deployments."""

    def __init__(
        self,
        sample_db: Path | str = DEFAULT_SAMPLE_DB,
        runtime_db: Path | str = DEFAULT_RUNTIME_DB,
    ):
        self.runtime_db = prepare_runtime_db(sample_db=sample_db, runtime_db=runtime_db)

        from .tools.sales_tools import SalesTools

        self.sales_tools = SalesTools()
        self._sales_agent = None
        self._analytics_agent = None
        self._router_agent = None

    def get_health(self) -> Dict:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM customers")
            customer_count = cursor.fetchone()[0]

        return {
            "status": "healthy",
            "database": "connected",
            "customer_count": customer_count,
            "database_path": str(self.runtime_db),
            "mode": "direct",
            "agents": {
                "router": "available",
                "sales": "available",
                "analytics": "available",
            },
        }

    def chat(self, message: str, agent: str = "router") -> Dict:
        start_time = time.time()
        selected_agent = agent or "router"

        if selected_agent == "sales":
            response = self._run_sales(message)
            agent_used = "sales"
        elif selected_agent == "analytics":
            response = self._run_analytics(message)
            agent_used = "analytics"
        else:
            agent_used, response = self._run_router(message)

        return {
            "response": response,
            "agent_used": agent_used,
            "execution_time": time.time() - start_time,
        }

    def _run_router(self, message: str) -> tuple[str, str]:
        router_agent = self._load_router_agent()
        if router_agent is not None:
            try:
                result = router_agent.invoke({"input": message})
                return "router", result["output"]
            except Exception:
                pass

        lower_message = message.lower()
        if any(keyword in lower_message for keyword in ["revenue", "analytics", "report", "top product", "aov"]):
            return "analytics", self._run_analytics(message)
        if any(keyword in lower_message for keyword in ["system", "health", "status"]):
            return "router", self._system_info()
        return "sales", self._run_sales(message)

    def _run_sales(self, message: str) -> str:
        sales_agent = self._load_sales_agent()
        if sales_agent is not None:
            try:
                result = sales_agent.invoke({"input": message})
                return result["output"]
            except Exception:
                pass

        return self.sales_tools.handle(message)

    def _run_analytics(self, message: str) -> str:
        analytics_agent = self._load_analytics_agent()
        if analytics_agent is not None:
            try:
                result = analytics_agent.invoke({"input": message})
                return result["output"]
            except Exception:
                pass

        return self._analytics_fallback(message)

    def _load_sales_agent(self):
        if self._sales_agent is not None:
            return self._sales_agent

        if not has_llm_credentials() or not _hosted_direct_ai_enabled():
            return None

        try:
            from .agents.SalesAgent import create_sales_agent_with_chat

            self._sales_agent = create_sales_agent_with_chat()
            return self._sales_agent
        except Exception:
            return None

    def _load_analytics_agent(self):
        if self._analytics_agent is not None:
            return self._analytics_agent

        if not has_llm_credentials() or not _hosted_direct_ai_enabled():
            return None

        try:
            from .agents.AnalyticsAgent import create_analytics_agent

            self._analytics_agent = create_analytics_agent()
            return self._analytics_agent
        except Exception:
            return None

    def _load_router_agent(self):
        if self._router_agent is not None:
            return self._router_agent

        if not has_llm_credentials() or not _hosted_direct_ai_enabled():
            return None

        try:
            from .agents.simple_router_agent import create_simple_router_agent

            self._router_agent = create_simple_router_agent()
            return self._router_agent
        except Exception:
            return None

    def _system_info(self) -> str:
        with get_db() as conn:
            cursor = conn.cursor()
            stats = {}
            for table_name in ("customers", "orders", "leads"):
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                stats[table_name] = cursor.fetchone()[0]

        return (
            "📊 **System Status:**\n"
            f"• Database: Connected\n"
            f"• Total Customers: {stats['customers']}\n"
            f"• Total Orders: {stats['orders']}\n"
            f"• Active Leads: {stats['leads']}\n"
            "• Deployment Mode: Direct Streamlit Demo\n"
        )

    def _analytics_fallback(self, message: str) -> str:
        lower_message = message.lower()
        if "revenue" in lower_message and "month" in lower_message:
            rows = self.sales_tools.sales_sql_read(
                """
                SELECT
                    strftime('%Y-%m', created_at) AS period,
                    ROUND(SUM(total), 2) AS revenue,
                    COUNT(*) AS orders
                FROM orders
                GROUP BY strftime('%Y-%m', created_at)
                ORDER BY period
                """
            )
            if rows and "error" not in rows[0]:
                lines = ["📈 **Revenue by Month:**", ""]
                for row in rows:
                    lines.append(f"• {row['period']}: ${row['revenue']:.2f} across {row['orders']} orders")
                return "\n".join(lines)

        if "top" in lower_message and "product" in lower_message:
            rows = self.sales_tools.sales_sql_read(
                """
                SELECT
                    p.name AS product_name,
                    ROUND(SUM(oi.quantity * oi.price), 2) AS revenue
                FROM order_items oi
                JOIN products p ON p.id = oi.product_id
                GROUP BY p.id, p.name
                ORDER BY revenue DESC
                LIMIT 5
                """
            )
            if rows and "error" not in rows[0]:
                lines = ["🏆 **Top Products by Revenue:**", ""]
                for index, row in enumerate(rows, start=1):
                    lines.append(f"{index}. {row['product_name']} - ${row['revenue']:.2f}")
                return "\n".join(lines)

        if "aov" in lower_message or "average order value" in lower_message:
            rows = self.sales_tools.sales_sql_read(
                """
                SELECT
                    strftime('%Y-%m', created_at) AS period,
                    ROUND(AVG(total), 2) AS average_order_value
                FROM orders
                GROUP BY strftime('%Y-%m', created_at)
                ORDER BY period
                """
            )
            if rows and "error" not in rows[0]:
                lines = ["📊 **Average Order Value by Month:**", ""]
                for row in rows:
                    lines.append(f"• {row['period']}: ${row['average_order_value']:.2f}")
                return "\n".join(lines)

        return (
            "Analytics fallback is available for revenue by month, top products by revenue, "
            "and average order value by month. Add a GROQ_API_KEY to enable the full analytics agent."
        )


@lru_cache(maxsize=1)
def get_direct_service() -> DirectERPService:
    return DirectERPService()
