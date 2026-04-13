from __future__ import annotations

from typing import Any

from backend.config.llm import coerce_text, get_llm, has_llm_credentials
from backend.db import get_db
from backend.memory.base_memory import RouterGlobalState
from backend.mcp.tool_registry import ToolRegistry
from backend.tools.common import extract_json_payload, format_rows, looks_like_json_payload, parse_llm_json_object


class InventoryTools:
    HIGH_RISK_PO_AMOUNT = 5000.0

    def __init__(self, state_store: RouterGlobalState, registry: ToolRegistry, db_path: str | None = None):
        self.state_store = state_store
        self.registry = registry
        self.db_path = db_path
        self.registry.register_tool(
            name="inventory_query_tool",
            handler=self.inventory_query_tool,
            description="Inspect stock, supplier, and procurement state",
            input_schema={"message": "str"},
            module="inventory",
        )
        self.registry.register_tool(
            name="adjust_stock_tool",
            handler=self.adjust_stock_tool,
            description="Adjust stock and create a stock movement",
            input_schema={"product_id": "int", "change_qty": "int", "reason": "str"},
            module="inventory",
            read_only=False,
        )
        self.registry.register_tool(
            name="forecast_tool",
            handler=self.forecast_tool,
            description="Forecast demand using recent stock movements",
            input_schema={"product_id": "int"},
            module="inventory",
        )
        self.registry.register_tool(
            name="create_purchase_order_tool",
            handler=self.create_purchase_order_tool,
            description="Create a purchase order for a product",
            input_schema={"product_id": "int", "quantity": "int"},
            module="inventory",
            read_only=False,
            requires_approval=True,
        )
        self.registry.register_tool(
            name="receive_purchase_order_tool",
            handler=self.receive_purchase_order_tool,
            description="Record a purchase order receipt and update stock",
            input_schema={"po_id": "int", "product_id": "int", "received_qty": "int"},
            module="inventory",
            read_only=False,
        )
        self.registry.register_tool(
            name="inventory_rag_tool",
            handler=self.inventory_rag_tool,
            description="Retrieve inventory glossary and supplier document context",
            input_schema={"query": "str"},
            module="inventory",
        )

    def _log(self, tool_name: str, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        self.state_store.log_tool_call("inventory", tool_name, payload, result)
        return result

    def inventory_query_tool(self, message: str) -> dict[str, Any]:
        lowered = message.lower()
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            if "below their reorder point" in lowered or "below reorder point" in lowered or "reorder point" in lowered:
                cursor.execute(
                    """
                    SELECT p.id AS product_id, p.sku, p.name, s.qty_on_hand, s.reorder_point,
                           (s.reorder_point - s.qty_on_hand) AS shortage
                    FROM stock s
                    JOIN products p ON p.id = s.product_id
                    WHERE s.qty_on_hand < s.reorder_point
                    ORDER BY shortage DESC, p.id
                    """
                )
                rows = [dict(row) for row in cursor.fetchall()]
                title = "Products Below Reorder Point"
            elif "low stock" in lowered or "reorder next" in lowered or "stock risk" in lowered:
                cursor.execute(
                    """
                    SELECT p.id AS product_id, p.name, s.qty_on_hand, s.reorder_point,
                           MAX(COALESCE(sp.default_cost, 0)) AS estimated_unit_cost
                    FROM stock s
                    JOIN products p ON p.id = s.product_id
                    LEFT JOIN supplier_products sp ON sp.product_id = p.id
                    GROUP BY p.id, p.name, s.qty_on_hand, s.reorder_point
                    ORDER BY (s.qty_on_hand - s.reorder_point) ASC, p.id
                    """
                )
                rows = [dict(row) for row in cursor.fetchall()]
                title = "Stock Risk"
            elif "supplier" in lowered:
                cursor.execute(
                    """
                    SELECT s.id, s.name, s.email, sp.product_id, sp.lead_time_days, sp.default_cost
                    FROM suppliers s JOIN supplier_products sp ON sp.supplier_id = s.id
                    ORDER BY s.id, sp.product_id
                    """
                )
                rows = [dict(row) for row in cursor.fetchall()]
                title = "Suppliers"
            else:
                cursor.execute(
                    """
                    SELECT p.id AS product_id, p.name, s.qty_on_hand, s.reorder_point
                    FROM stock s JOIN products p ON p.id = s.product_id
                    ORDER BY p.id
                    """
                )
                rows = [dict(row) for row in cursor.fetchall()]
                title = "Stock"
        if title == "Products Below Reorder Point":
            if rows:
                message_text = f"{title}\n\n{format_rows(rows)}"
            else:
                message_text = "No products are currently below their reorder point."
        else:
            message_text = f"{title}\n\n{format_rows(rows)}"
        result = {"message": message_text, "rows": rows}
        return self._log("inventory_query_tool", {"message": message}, result)

    def adjust_stock_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT qty_on_hand FROM stock WHERE product_id = ?", (payload["product_id"],))
            row = cursor.fetchone()
            if row is None:
                result = {"message": f"Stock record for product {payload['product_id']} not found."}
                return self._log("adjust_stock_tool", payload, result)
            new_qty = int(row[0]) + int(payload["change_qty"])
            cursor.execute("UPDATE stock SET qty_on_hand = ? WHERE product_id = ?", (new_qty, payload["product_id"]))
            cursor.execute(
                """
                INSERT INTO stock_movements (product_id, change_qty, reason, ref_id, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (payload["product_id"], payload["change_qty"], payload.get("reason"), payload.get("ref_id")),
            )
            conn.commit()
        result = {"message": f"Stock updated for product {payload['product_id']}. New quantity: {new_qty}.", "qty_on_hand": new_qty}
        return self._log("adjust_stock_tool", payload, result)

    def forecast_tool(self, product_id: int) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ABS(change_qty) AS demand
                FROM stock_movements
                WHERE product_id = ? AND change_qty < 0
                ORDER BY id DESC LIMIT 6
                """,
                (product_id,),
            )
            demand_samples = [float(row[0]) for row in cursor.fetchall()]
        avg_demand = round(sum(demand_samples) / len(demand_samples), 2) if demand_samples else 0.0
        recommended_qty = max(1, int(round(avg_demand * 2))) if avg_demand else 1
        result = {
            "message": f"Forecast demand for product {product_id} is {avg_demand:.2f} units per period.",
            "average_demand": avg_demand,
            "recommended_quantity": recommended_qty,
        }
        return self._log("forecast_tool", {"product_id": product_id}, result)

    def create_purchase_order_tool(self, payload: dict[str, Any], requested_by: str = "system", approved: bool = False) -> dict[str, Any]:
        supplier = self._pick_supplier(payload["product_id"])
        if supplier is None:
            result = {"message": "No supplier found for requested product."}
            return self._log("create_purchase_order_tool", payload, result)

        estimated_cost = float(supplier["default_cost"]) * int(payload["quantity"])
        if estimated_cost >= self.HIGH_RISK_PO_AMOUNT and not approved:
            approval = self.state_store.create_approval(
                "inventory",
                {"action": "create_purchase_order", "payload": payload},
                requested_by,
            )
            result = {"message": "Purchase order requires approval before creation.", "approval_required": approval}
            return self._log("create_purchase_order_tool", payload, result)

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO purchase_orders (supplier_id, status, created_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (supplier["supplier_id"], payload.get("status", "draft")),
            )
            po_id = int(cursor.lastrowid)
            cursor.execute(
                """
                INSERT INTO po_items (po_id, product_id, quantity, unit_cost)
                VALUES (?, ?, ?, ?)
                """,
                (po_id, payload["product_id"], payload["quantity"], supplier["default_cost"]),
            )
            conn.commit()
        result = {
            "message": f"Purchase order created with id {po_id} for supplier {supplier['supplier_name']}.",
            "po_id": po_id,
            "supplier_id": supplier["supplier_id"],
            "estimated_cost": estimated_cost,
        }
        return self._log("create_purchase_order_tool", payload, result)

    def receive_purchase_order_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO po_receipts (po_id, product_id, received_qty, received_at)
                VALUES (?, ?, ?, ?)
                """,
                (payload["po_id"], payload["product_id"], payload["received_qty"], payload.get("received_at")),
            )
            cursor.execute("SELECT qty_on_hand FROM stock WHERE product_id = ?", (payload["product_id"],))
            current_qty = int(cursor.fetchone()[0])
            new_qty = current_qty + int(payload["received_qty"])
            cursor.execute("UPDATE stock SET qty_on_hand = ? WHERE product_id = ?", (new_qty, payload["product_id"]))
            cursor.execute(
                """
                INSERT INTO stock_movements (product_id, change_qty, reason, ref_id, created_at)
                VALUES (?, ?, 'purchase_receipt', ?, CURRENT_TIMESTAMP)
                """,
                (payload["product_id"], payload["received_qty"], payload["po_id"]),
            )
            cursor.execute("UPDATE purchase_orders SET status = 'received' WHERE id = ?", (payload["po_id"],))
            conn.commit()
        result = {"message": f"Receipt recorded for purchase order {payload['po_id']}.", "qty_on_hand": new_qty}
        return self._log("receive_purchase_order_tool", payload, result)

    def inventory_rag_tool(self, query: str) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT module, path, tags FROM documents
                WHERE module = 'inventory' AND (lower(tags) LIKE ? OR lower(path) LIKE ?)
                """,
                (f"%{query.lower()}%", f"%{query.lower()}%"),
            )
            docs = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """
                SELECT term, definition, module FROM glossary
                WHERE module = 'inventory' OR lower(term) LIKE ?
                """,
                (f"%{query.lower()}%",),
            )
            glossary = [dict(row) for row in cursor.fetchall()]
        result = {"message": f"Inventory knowledge returned {len(docs)} documents and {len(glossary)} glossary rows.", "documents": docs, "glossary": glossary}
        return self._log("inventory_rag_tool", {"query": query}, result)

    def handle(self, message: str, requested_by: str = "system") -> dict[str, Any]:
        lowered = message.lower().strip()
        if lowered.startswith("adjust stock") and looks_like_json_payload(message):
            return self.adjust_stock_tool(extract_json_payload(message))
        if lowered.startswith("create purchase order") and looks_like_json_payload(message):
            return self.create_purchase_order_tool(extract_json_payload(message), requested_by=requested_by)
        if lowered.startswith("receive purchase order") and looks_like_json_payload(message):
            return self.receive_purchase_order_tool(extract_json_payload(message))
        if "forecast" in lowered and looks_like_json_payload(message):
            payload = extract_json_payload(message)
            return self.forecast_tool(int(payload["product_id"]))
        if (
            "below their reorder point" in lowered
            or "below reorder point" in lowered
            or "which products are below" in lowered
            or "low stock" in lowered
            or "reorder next" in lowered
            or "stock risk" in lowered
        ):
            return self.inventory_query_tool(message)
        planned = self._llm_plan(message)
        if planned:
            planned_result = self._dispatch_planned_action(planned, message, requested_by)
            if planned_result is not None:
                return planned_result
        if "contract" in lowered or "supplier terms" in lowered:
            return self.inventory_rag_tool(message)
        return self.inventory_query_tool(message)

    def execute_approved_payload(self, approval_payload: dict[str, Any], requested_by: str = "system") -> dict[str, Any]:
        action = approval_payload.get("action")
        payload = approval_payload.get("payload") if "payload" in approval_payload else approval_payload
        if action == "create_purchase_order" or {"product_id", "quantity"} <= set(payload):
            return self.create_purchase_order_tool(payload, requested_by=requested_by, approved=True)
        return {"message": "No executable inventory action found for approval."}

    def _llm_plan(self, message: str) -> dict[str, Any] | None:
        if not has_llm_credentials():
            return None

        prompt = (
            "You are the inventory planning brain for an ERP agent.\n"
            "Choose exactly one action and return only JSON.\n"
            "Allowed actions:\n"
            "- adjust_stock: payload needs product_id, change_qty, reason, ref_id optional\n"
            "- create_purchase_order: payload needs product_id, quantity\n"
            "- receive_purchase_order: payload needs po_id, product_id, received_qty, received_at optional\n"
            "- forecast: payload needs product_id\n"
            "- inventory_rag: payload needs query\n"
            "- inventory_query: payload may be empty\n"
            "If the user asks to reorder, restock, or buy units from a supplier, choose create_purchase_order.\n"
            "Return only JSON.\n"
            f"User message: {message}"
        )
        planned = parse_llm_json_object(coerce_text(get_llm().invoke(prompt)))
        if planned:
            self.state_store.log_tool_call("inventory", "inventory_llm_plan", {"message": message}, planned)
        return planned

    def _dispatch_planned_action(
        self,
        planned: dict[str, Any],
        message: str,
        requested_by: str,
    ) -> dict[str, Any] | None:
        action = str(planned.get("action", "")).strip().lower()
        payload = planned.get("payload") or {}
        if action == "adjust_stock" and {"product_id", "change_qty"} <= set(payload):
            return self.adjust_stock_tool(payload)
        if action == "create_purchase_order" and {"product_id", "quantity"} <= set(payload):
            return self.create_purchase_order_tool(payload, requested_by=requested_by)
        if action == "receive_purchase_order" and {"po_id", "product_id", "received_qty"} <= set(payload):
            return self.receive_purchase_order_tool(payload)
        if action == "forecast" and "product_id" in payload:
            return self.forecast_tool(int(payload["product_id"]))
        if action == "inventory_rag":
            query = payload.get("query") or message
            return self.inventory_rag_tool(query)
        if action == "inventory_query":
            return self.inventory_query_tool(message)
        return None

    def _pick_supplier(self, product_id: int) -> dict[str, Any] | None:
        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT sp.supplier_id, s.name AS supplier_name, sp.lead_time_days, sp.default_cost
                FROM supplier_products sp
                JOIN suppliers s ON s.id = sp.supplier_id
                WHERE sp.product_id = ?
                ORDER BY sp.default_cost ASC, sp.lead_time_days ASC
                LIMIT 1
                """,
                (product_id,),
            )
            row = cursor.fetchone()
        return dict(row) if row else None
