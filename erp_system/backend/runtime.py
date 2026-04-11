import os
import shutil
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from .config.llm import get_llm, has_llm_credentials
from .db import get_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_DB = PROJECT_ROOT / "databases" / "erp_sample.db"
DEFAULT_RUNTIME_DB = Path(tempfile.gettempdir()) / "erp_system_demo" / "erp_public_demo.db"


class _HostedLLMAgent:
    def __init__(self, responder):
        self._responder = responder

    def invoke(self, payload: Dict[str, str]) -> Dict[str, str]:
        return {"output": self._responder(payload["input"])}


def _hosted_direct_ai_enabled() -> bool:
    """
    Enable hosted direct AI when Groq credentials are available, unless the
    environment explicitly disables it.
    """
    if not has_llm_credentials():
        return False

    return os.getenv("ERP_ENABLE_DIRECT_AI", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


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
        self._hosted_sales_llm_agent = None
        self._hosted_analytics_llm_agent = None
        self._hosted_router_llm_enabled = None
        self._sales_agent = None
        self._analytics_agent = None
        self._router_agent = None

    def get_health(self) -> Dict:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM customers")
            customer_count = cursor.fetchone()[0]

        llm_mode = "groq-hosted" if _hosted_direct_ai_enabled() else "fallback"

        return {
            "status": "healthy",
            "database": "connected",
            "customer_count": customer_count,
            "database_path": str(self.runtime_db),
            "mode": "direct",
            "llm_mode": llm_mode,
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
        route = self._classify_route_with_llm(message)
        if route == "analytics":
            return "analytics", self._run_analytics(message)
        if route == "router":
            return "router", self._system_info()
        if route == "sales":
            return "sales", self._run_sales(message)

        router_agent = self._load_router_agent()
        if router_agent is not None:
            try:
                result = router_agent.invoke({"input": message})
                return "router", result["output"]
            except Exception:
                pass

        return self._fallback_route(message)

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
                output = result["output"]
                if output:
                    return output
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
            self._sales_agent = self._build_hosted_sales_llm_agent()
            return self._sales_agent

    def _load_analytics_agent(self):
        if self._analytics_agent is not None:
            return self._analytics_agent

        if not has_llm_credentials() or not _hosted_direct_ai_enabled():
            return None

        self._analytics_agent = self._build_hosted_analytics_llm_agent()
        return self._analytics_agent

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

    def _classify_route_with_llm(self, message: str) -> Optional[str]:
        if not has_llm_credentials() or not _hosted_direct_ai_enabled():
            return None

        prompt = (
            "You are the Router Agent for an ERP assistant.\n"
            "Classify the user's message into exactly one domain: sales, analytics, or router.\n"
            "Rules:\n"
            "- sales: customers, orders, leads, CRM, support tickets\n"
            "- analytics: revenue, KPIs, summaries, reports, trends, product performance, executive analysis\n"
            "- router: system status, health, configuration, app/runtime status\n"
            "Return only one word: sales, analytics, or router.\n\n"
            f"Message: {message}\n"
            "Label:"
        )
        try:
            raw = self._coerce_llm_output(get_llm().invoke(prompt)).strip().lower()
            for label in ("sales", "analytics", "router"):
                if label in raw:
                    return label
        except Exception:
            return None
        return None

    def _fallback_route(self, message: str) -> tuple[str, str]:
        lower_message = message.lower()
        customer_count_phrases = ("how many customers", "count customers", "number of customers", "total customers")
        analytics_keywords = (
            "revenue",
            "analytics",
            "report",
            "top product",
            "worst product",
            "aov",
            "average order value",
            "trend",
            "insight",
            "top customers by revenue",
            "sales trend",
            "performance",
            "analysis",
            "summary",
        )
        sales_keywords = (
            "customer",
            "customers",
            "lead",
            "leads",
            "order",
            "orders",
            "ticket",
            "support",
        )

        if any(phrase in lower_message for phrase in customer_count_phrases):
            return "sales", self.sales_tools.handle(message)
        if any(keyword in lower_message for keyword in ["system", "health", "status"]):
            return "router", self._system_info()
        if any(keyword in lower_message for keyword in analytics_keywords):
            return "analytics", self._analytics_fallback(message)
        if any(keyword in lower_message for keyword in sales_keywords):
            return "sales", self.sales_tools.handle(message)
        return "sales", self.sales_tools.handle(message)

    def _build_hosted_sales_llm_agent(self):
        if self._hosted_sales_llm_agent is not None:
            return self._hosted_sales_llm_agent

        def respond(message: str) -> str:
            context = self._build_sales_context()
            prompt = (
                "You are the hosted Sales Agent for an ERP demo.\n"
                "Use only the provided data snapshot. If the answer is not supported by the snapshot, say so briefly.\n"
                "Be concise and businesslike.\n\n"
                f"Sales snapshot:\n{context}\n\n"
                f"Question: {message}\n"
                "Answer:"
            )
            response = get_llm().invoke(prompt)
            return self._coerce_llm_output(response)

        self._hosted_sales_llm_agent = _HostedLLMAgent(respond)
        return self._hosted_sales_llm_agent

    def _build_hosted_analytics_llm_agent(self):
        if self._hosted_analytics_llm_agent is not None:
            return self._hosted_analytics_llm_agent

        def respond(message: str) -> str:
            context = self._build_analytics_context()
            prompt = (
                "You are the hosted Analytics Agent for an ERP demo.\n"
                "Use only the provided business snapshot.\n"
                "Answer in plain English. Mention concrete figures from the snapshot. "
                "If there is not enough information, say that clearly instead of inventing facts.\n\n"
                f"Business snapshot:\n{context}\n\n"
                f"Question: {message}\n"
                "Answer:"
            )
            response = get_llm().invoke(prompt)
            return self._coerce_llm_output(response)

        self._hosted_analytics_llm_agent = _HostedLLMAgent(respond)
        return self._hosted_analytics_llm_agent

    def _build_sales_context(self) -> str:
        customer_summary = self.sales_tools._customer_summary()
        recent_orders = self.sales_tools._list_recent_orders()
        recent_leads = self.sales_tools._list_leads()
        return f"{customer_summary}\n\n{recent_orders}\n\n{recent_leads}"

    def _build_analytics_context(self) -> str:
        sections = [
            self._analytics_fallback("revenue by month"),
            self._analytics_fallback("what is our total revenue"),
            self._analytics_fallback("top products by revenue"),
            self._analytics_fallback("worst 5 products by revenue"),
            self._analytics_fallback("top customers by revenue"),
            self._analytics_fallback("average order value by month"),
            self._analytics_fallback("sales trend"),
        ]
        return "\n\n".join(section for section in sections if section)

    def _coerce_llm_output(self, response) -> str:
        if hasattr(response, "content"):
            return str(response.content).strip()
        return str(response).strip()

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
        if any(term in lower_message for term in ["total analysis", "overall analysis", "overall performance", "executive summary"]):
            return self._analytics_summary()

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

        if "total revenue" in lower_message or ("revenue" in lower_message and "total" in lower_message):
            rows = self.sales_tools.sales_sql_read(
                """
                SELECT
                    ROUND(COALESCE(SUM(total), 0), 2) AS total_revenue,
                    COUNT(*) AS order_count
                FROM orders
                """
            )
            if rows and "error" not in rows[0]:
                row = rows[0]
                return (
                    "💰 **Total Revenue:**\n\n"
                    f"Our total revenue is ${row['total_revenue']:.2f} across {row['order_count']} orders."
                )

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

        if any(keyword in lower_message for keyword in ["worst", "lowest", "bottom"]) and "product" in lower_message:
            rows = self.sales_tools.sales_sql_read(
                """
                SELECT
                    p.name AS product_name,
                    ROUND(SUM(oi.quantity * oi.price), 2) AS revenue
                FROM order_items oi
                JOIN products p ON p.id = oi.product_id
                GROUP BY p.id, p.name
                ORDER BY revenue ASC
                LIMIT 5
                """
            )
            if rows and "error" not in rows[0]:
                lines = ["📉 **Worst Products by Revenue:**", ""]
                for index, row in enumerate(rows, start=1):
                    lines.append(f"{index}. {row['product_name']} - ${row['revenue']:.2f}")
                return "\n".join(lines)

        if "top customer" in lower_message or ("top" in lower_message and "customer" in lower_message):
            rows = self.sales_tools.sales_sql_read(
                """
                SELECT
                    c.name AS customer_name,
                    ROUND(COALESCE(SUM(o.total), 0), 2) AS revenue,
                    COUNT(o.id) AS orders
                FROM customers c
                LEFT JOIN orders o ON c.id = o.customer_id
                GROUP BY c.id, c.name
                ORDER BY revenue DESC
                LIMIT 5
                """
            )
            if rows and "error" not in rows[0]:
                lines = ["🏆 **Top Customers by Revenue:**", ""]
                for index, row in enumerate(rows, start=1):
                    lines.append(
                        f"{index}. {row['customer_name']} - ${row['revenue']:.2f} across {row['orders']} orders"
                    )
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

        if "sales trend" in lower_message or ("trend" in lower_message and "sales" in lower_message):
            rows = self.sales_tools.sales_sql_read(
                """
                SELECT
                    strftime('%Y-%m', created_at) AS period,
                    ROUND(SUM(total), 2) AS revenue
                FROM orders
                GROUP BY strftime('%Y-%m', created_at)
                ORDER BY period
                """
            )
            if rows and "error" not in rows[0]:
                start_period = rows[0]["period"]
                end_period = rows[-1]["period"]
                start_revenue = rows[0]["revenue"]
                end_revenue = rows[-1]["revenue"]
                direction = "upward" if end_revenue >= start_revenue else "downward"
                lines = ["📈 **Sales Trend Summary:**", ""]
                lines.append(f"Revenue moved from ${start_revenue:.2f} in {start_period} to ${end_revenue:.2f} in {end_period}.")
                lines.append(f"The overall trend is {direction}.")
                return "\n".join(lines)

        return (
            "Analytics fallback supports revenue by month, total revenue, top or worst products by revenue, "
            "top customers by revenue, average order value by month, and sales trends. "
            "Set GROQ_API_KEY to enable full analytics AI answers."
        )

    def _analytics_summary(self) -> str:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM customers")
            customer_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM orders")
            order_count = cursor.fetchone()[0]
            cursor.execute("SELECT ROUND(COALESCE(SUM(total), 0), 2), ROUND(COALESCE(AVG(total), 0), 2) FROM orders")
            total_revenue, average_order_value = cursor.fetchone()
            cursor.execute(
                """
                SELECT c.name, ROUND(COALESCE(SUM(o.total), 0), 2) AS revenue
                FROM customers c
                LEFT JOIN orders o ON c.id = o.customer_id
                GROUP BY c.id, c.name
                ORDER BY revenue DESC
                LIMIT 1
                """
            )
            top_customer = cursor.fetchone()

        top_customer_name = top_customer[0] if top_customer else "N/A"
        top_customer_revenue = top_customer[1] if top_customer else 0

        return (
            "📊 **Executive Summary:**\n\n"
            f"• Total Revenue: ${total_revenue:.2f}\n"
            f"• Total Orders: {order_count}\n"
            f"• Total Customers: {customer_count}\n"
            f"• Average Order Value: ${average_order_value:.2f}\n"
            f"• Top Customer by Revenue: {top_customer_name} (${top_customer_revenue:.2f})\n\n"
            "This is the high-level business snapshot for the current demo dataset."
        )


@lru_cache(maxsize=1)
def get_direct_service() -> DirectERPService:
    return DirectERPService()
