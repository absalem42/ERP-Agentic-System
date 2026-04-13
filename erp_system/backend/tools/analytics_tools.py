from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from backend.config.llm import coerce_text, get_llm, has_llm_credentials
from backend.db import fetch_all, fetch_one, get_db
from backend.memory.base_memory import AnalyticsReportMemory, RouterGlobalState
from backend.mcp.tool_registry import ToolRegistry
from backend.tools.common import detect_numeric_columns, extract_json_payload, format_rows, looks_like_json_payload


FORBIDDEN_SQL_PATTERNS = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma)\b", re.I)


class AnalyticsTools:
    def __init__(
        self,
        state_store: RouterGlobalState,
        report_memory: AnalyticsReportMemory,
        registry: ToolRegistry,
        db_path: str | None = None,
    ):
        self.state_store = state_store
        self.report_memory = report_memory
        self.registry = registry
        self.db_path = db_path
        self.registry.register_tool(
            name="text_to_sql_tool",
            handler=self.text_to_sql_tool,
            description="Translate analytics questions into guarded read-only SQL",
            input_schema={"question": "str"},
            module="analytics",
        )
        self.registry.register_tool(
            name="rag_definition_tool",
            handler=self.rag_definition_tool,
            description="Retrieve glossary and document context for analytics explanations",
            input_schema={"query": "str"},
            module="analytics",
        )
        self.registry.register_tool(
            name="analytics_reporting_tool",
            handler=self.analytics_reporting_tool,
            description="Convert tabular analytics results into a chart specification and narrative",
            input_schema={"question": "str", "rows": "list[dict]"},
            module="analytics",
        )
        self.registry.register_tool(
            name="save_report_tool",
            handler=self.save_report_tool,
            description="Persist a read-only SQL report in saved_reports",
            input_schema={"title": "str", "sql": "str"},
            module="analytics",
            read_only=False,
        )
        self.registry.register_tool(
            name="run_report_tool",
            handler=self.run_report_tool,
            description="Run a saved report and format the result",
            input_schema={"title": "str"},
            module="analytics",
        )

    def _log(self, tool_name: str, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        self.state_store.log_tool_call("analytics", tool_name, payload, result)
        return result

    def is_read_only_sql(self, sql: str) -> bool:
        normalized = self.normalize_sql(sql).lower()
        if not normalized:
            return False
        if FORBIDDEN_SQL_PATTERNS.search(normalized):
            return False
        return normalized.startswith("select") or normalized.startswith("with")

    def normalize_sql(self, sql: str | None) -> str:
        if not sql:
            return ""
        normalized = sql.strip()
        fenced = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", normalized, re.IGNORECASE | re.DOTALL)
        if fenced:
            normalized = fenced.group(1).strip()
        normalized = re.sub(r"^\s*sql\s+", "", normalized, flags=re.IGNORECASE)
        if not re.match(r"^\s*(select|with)\b", normalized, re.IGNORECASE):
            match = re.search(r"\b(select|with)\b.*", normalized, re.IGNORECASE | re.DOTALL)
            if match:
                normalized = match.group(0).strip()
        return normalized.strip().rstrip(";")

    def text_to_sql_tool(self, question: str) -> dict[str, Any]:
        lowered = question.lower().strip()
        if lowered.startswith("run sql "):
            sql = self.normalize_sql(question[8:].strip())
            if not self.is_read_only_sql(sql):
                result = {"message": "Analytics SQL must be read-only.", "sql": sql}
                return self._log("text_to_sql_tool", {"question": question}, result)
            rows = fetch_all(sql, db_path=self.db_path)
            result = {"message": f"SQL executed.\n\n{format_rows(rows)}", "sql": sql, "rows": rows}
            return self._log("text_to_sql_tool", {"question": question}, result)

        query_spec = self._heuristic_query(question)
        if query_spec is None and has_llm_credentials():
            llm_sql = self._llm_sql(question)
            if llm_sql:
                query_spec = {"sql": llm_sql, "params": (), "meta": {}}

        if query_spec is None:
            result = {"message": "No SQL mapping available for that analytics question.", "rows": [], "sql": None}
            return self._log("text_to_sql_tool", {"question": question}, result)
        sql = self.normalize_sql(query_spec["sql"])
        if not self.is_read_only_sql(sql):
            result = {"message": "Analytics SQL must be read-only.", "sql": sql}
            return self._log("text_to_sql_tool", {"question": question}, result)

        rows = fetch_all(sql, query_spec.get("params", ()), db_path=self.db_path)
        result = {
            "message": f"SQL executed.\n\n{format_rows(rows)}",
            "sql": sql,
            "rows": rows,
            "meta": query_spec.get("meta", {}),
        }
        return self._log("text_to_sql_tool", {"question": question}, result)

    def rag_definition_tool(self, query: str) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT term, definition, module FROM glossary
                WHERE lower(term) LIKE ? OR lower(definition) LIKE ?
                ORDER BY term
                """,
                (f"%{query.lower()}%", f"%{query.lower()}%"),
            )
            glossary = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """
                SELECT module, path, tags FROM documents
                WHERE lower(module) = 'analytics' OR lower(tags) LIKE ? OR lower(path) LIKE ?
                """,
                (f"%{query.lower()}%", f"%{query.lower()}%"),
            )
            docs = [dict(row) for row in cursor.fetchall()]
        result = {
            "message": f"Found {len(glossary)} glossary entries and {len(docs)} related documents.",
            "glossary": glossary,
            "documents": docs,
        }
        return self._log("rag_definition_tool", {"query": query}, result)

    def analytics_reporting_tool(
        self,
        question: str,
        rows: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        chart_spec = self._build_chart_spec(question, rows)
        narrative = self._build_narrative(question, rows, context=context, meta=meta)
        result = {"message": narrative, "chart_spec": chart_spec, "rows": rows}
        return self._log("analytics_reporting_tool", {"question": question, "row_count": len(rows)}, result)

    def save_report_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_read_only_sql(payload["sql"]):
            result = {"message": "Saved reports must use read-only SQL."}
            return self._log("save_report_tool", payload, result)
        message = self.report_memory.save_report(payload["title"], payload["sql"])
        result = {"message": message, "title": payload["title"]}
        return self._log("save_report_tool", payload, result)

    def run_report_tool(self, title: str) -> dict[str, Any]:
        report = self.report_memory.get_saved_report(title)
        if report is None:
            result = {"message": f"Saved report '{title}' not found."}
            return self._log("run_report_tool", {"title": title}, result)
        rows = fetch_all(report["sql"], db_path=self.db_path)
        report_result = self.analytics_reporting_tool(report["title"], rows)
        result = {
            "message": f"{report['title']}\n\n{format_rows(rows)}\n\n{report_result['message']}",
            "rows": rows,
            "chart_spec": report_result["chart_spec"],
            "title": report["title"],
        }
        return self._log("run_report_tool", {"title": title}, result)

    def handle(self, message: str) -> dict[str, Any]:
        lowered = message.lower().strip()
        if lowered.startswith("save report") and looks_like_json_payload(message):
            return self.save_report_tool(extract_json_payload(message))
        if lowered.startswith("run report"):
            title = message[len("run report") :].strip()
            return self.run_report_tool(title)

        sql_result = self.text_to_sql_tool(message)
        if "read-only" in sql_result["message"].lower():
            return sql_result

        explanation = self.rag_definition_tool(message)
        reporting = self.analytics_reporting_tool(
            message,
            sql_result.get("rows", []),
            context=explanation,
            meta=sql_result.get("meta"),
        )
        response = (
            f"{sql_result['message']}\n\n"
            f"{reporting['message']}\n\n"
            f"Context: {explanation['message']}"
        )
        return {
            "message": response,
            "rows": sql_result.get("rows", []),
            "chart_spec": reporting["chart_spec"],
            "sql": sql_result.get("sql"),
        }

    def _heuristic_query(self, question: str) -> dict[str, Any] | None:
        lowered = question.lower()
        if "total revenue" in lowered:
            return {
                "sql": "SELECT ROUND(COALESCE(SUM(total), 0), 2) AS total_revenue FROM orders",
                "params": (),
                "meta": {"analysis_type": "total_revenue"},
            }
        if "this month" in lowered and "revenue" in lowered and "trend" in lowered:
            current_month = datetime.now(timezone.utc).strftime("%Y-%m")
            latest_available = fetch_one(
                "SELECT strftime('%Y-%m', MAX(created_at)) AS latest_period FROM orders",
                db_path=self.db_path,
            )
            return {
                "sql": (
                    "SELECT strftime('%Y-%m-%d', created_at) AS period, ROUND(SUM(total), 2) AS revenue "
                    "FROM orders WHERE strftime('%Y-%m', created_at) = ? "
                    "GROUP BY strftime('%Y-%m-%d', created_at) ORDER BY period"
                ),
                "params": (current_month,),
                "meta": {
                    "analysis_type": "current_month_revenue_trend",
                    "requested_period": current_month,
                    "latest_available_period": (latest_available or {}).get("latest_period"),
                },
            }
        if (
            "revenue by month" in lowered
            or "monthly revenue" in lowered
            or ("revenue" in lowered and "trend" in lowered)
        ):
            return {
                "sql": (
                    "SELECT strftime('%Y-%m', created_at) AS period, ROUND(SUM(total), 2) AS revenue "
                    "FROM orders GROUP BY strftime('%Y-%m', created_at) ORDER BY period"
                ),
                "params": (),
                "meta": {"analysis_type": "monthly_revenue_trend"},
            }
        if (
            "top products" in lowered
            or "product revenue" in lowered
            or "worst products" in lowered
            or ("products" in lowered and "revenue" in lowered)
        ):
            order_direction = "ASC" if "worst" in lowered else "DESC"
            return {
                "sql": (
                    "SELECT p.name AS product_name, ROUND(SUM(oi.quantity * oi.price), 2) AS revenue "
                    "FROM orders o "
                    "JOIN order_items oi ON oi.order_id = o.id "
                    "JOIN products p ON p.id = oi.product_id "
                    "WHERE lower(COALESCE(o.status, '')) != 'cancelled' "
                    f"GROUP BY p.id, p.name ORDER BY revenue {order_direction} LIMIT 5"
                ),
                "params": (),
                "meta": {"analysis_type": "product_revenue_ranking", "ranking_direction": order_direction},
            }
        if "top customers" in lowered:
            return {
                "sql": (
                    "SELECT c.name AS customer_name, ROUND(SUM(o.total), 2) AS revenue "
                    "FROM orders o JOIN customers c ON c.id = o.customer_id "
                    "GROUP BY c.id, c.name ORDER BY revenue DESC LIMIT 5"
                ),
                "params": (),
                "meta": {"analysis_type": "customer_revenue_ranking"},
            }
        if "average order value" in lowered or "aov" in lowered:
            return {
                "sql": (
                    "SELECT strftime('%Y-%m', created_at) AS period, ROUND(AVG(total), 2) AS average_order_value "
                    "FROM orders GROUP BY strftime('%Y-%m', created_at) ORDER BY period"
                ),
                "params": (),
                "meta": {"analysis_type": "average_order_value"},
            }
        if "open tickets" in lowered:
            return {
                "sql": "SELECT status, COUNT(*) AS ticket_count FROM tickets GROUP BY status ORDER BY status",
                "params": (),
                "meta": {"analysis_type": "ticket_status_counts"},
            }
        return None

    def _llm_sql(self, question: str) -> str | None:
        prompt = (
            "Generate a single read-only SQLite SELECT query for this analytics question. "
            "Use only SELECT or WITH. Return only SQL.\n"
            f"Question: {question}"
        )
        response = coerce_text(get_llm().invoke(prompt))
        return self.normalize_sql(response)

    def _build_chart_spec(self, question: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        headers = list(rows[0].keys())
        numeric_columns = detect_numeric_columns(rows)
        if not numeric_columns:
            return None
        x_column = next((header for header in headers if header not in numeric_columns), headers[0])
        y_column = numeric_columns[0]
        chart_type = "line" if "trend" in question.lower() else "bar"
        return {
            "type": chart_type,
            "title": question.title(),
            "data": rows,
            "x": x_column,
            "y": y_column,
        }

    def _build_narrative(
        self,
        question: str,
        rows: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        meta = meta or {}
        if meta.get("analysis_type") == "current_month_revenue_trend" and not rows:
            requested_period = meta.get("requested_period") or "the requested month"
            latest_period = meta.get("latest_available_period")
            if latest_period:
                return (
                    f"No order revenue data is available for {requested_period}. "
                    f"Latest available revenue data is from {latest_period} in the sample database."
                )
            return f"No order revenue data is available for {requested_period}."
        if not rows:
            return f"No data available to answer: {question}"
        if meta.get("analysis_type") == "product_revenue_ranking":
            top_product = rows[0]
            total_revenue = sum(float(row.get("revenue", 0) or 0) for row in rows)
            glossary_terms = len((context or {}).get("glossary", []))
            document_count = len((context or {}).get("documents", []))
            return (
                f"Based on fulfilled order line revenue, {top_product['product_name']} leads with "
                f"{float(top_product['revenue']):.2f}. The top {len(rows)} products contribute "
                f"{total_revenue:.2f} in revenue. Why: this ranking is computed from orders, order_items, "
                f"and products, and it is supported by {glossary_terms} glossary entries and {document_count} "
                "analytics documents."
            )
        if len(rows) == 1:
            pairs = [f"{key} = {value}" for key, value in rows[0].items()]
            return f"The query produced one row: {', '.join(pairs)}."
        numeric_columns = detect_numeric_columns(rows)
        if numeric_columns:
            key = numeric_columns[0]
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            if meta.get("analysis_type") == "monthly_revenue_trend":
                return (
                    f"Monthly order revenue covers {len(rows)} periods. Revenue ranges from "
                    f"{min(values):.2f} to {max(values):.2f} across the available sample history."
                )
            return f"The dataset contains {len(rows)} rows. The {key} values range from {min(values):.2f} to {max(values):.2f}."
        return f"The dataset contains {len(rows)} rows."
