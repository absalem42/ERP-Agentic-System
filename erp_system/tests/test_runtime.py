import sqlite3
import sys
import types
from pathlib import Path

import pytest


def create_sample_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.executescript(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            created_at TEXT
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            total REAL NOT NULL,
            status TEXT,
            created_at TEXT
        );

        CREATE TABLE leads (
            id INTEGER PRIMARY KEY,
            customer_name TEXT,
            contact_email TEXT,
            message TEXT,
            score REAL,
            status TEXT,
            created_at TEXT
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL
        );

        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL
        );
        """
    )
    cursor.executemany(
        "INSERT INTO customers (id, name, email, created_at) VALUES (?, ?, ?, ?)",
        [
            (1, "Acme Corp", "contact@acme.example", "2024-01-10 10:00:00"),
            (2, "Globex LLC", "sales@globex.example", "2024-02-15 12:30:00"),
        ],
    )
    cursor.executemany(
        "INSERT INTO orders (id, customer_id, total, status, created_at) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, 948.49, "paid", "2024-04-05 14:00:00"),
            (2, 2, 249.50, "pending", "2024-04-10 13:20:00"),
        ],
    )
    cursor.executemany(
        "INSERT INTO leads (id, customer_name, contact_email, message, score, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                1,
                "Wayne Enterprises",
                "bruce@wayne.example",
                "Interested in bulk order",
                0.9,
                "new",
                "2024-04-02 08:30:00",
            ),
        ],
    )
    cursor.executemany(
        "INSERT INTO products (id, name, price) VALUES (?, ?, ?)",
        [
            (1, "Widget Pro", 199.99),
            (2, "Service A", 499.00),
        ],
    )
    cursor.executemany(
        "INSERT INTO order_items (id, order_id, product_id, quantity, price) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, 1, 2, 199.99),
            (2, 1, 2, 1, 499.00),
            (3, 2, 1, 1, 199.99),
        ],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def runtime_paths(tmp_path):
    sample_db = tmp_path / "sample" / "erp_sample.db"
    runtime_db = tmp_path / "runtime" / "erp_public_demo.db"
    create_sample_db(sample_db)
    return sample_db, runtime_db


def test_prepare_runtime_db_copies_demo_database(runtime_paths, monkeypatch):
    from backend import runtime

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("DB_PATH", raising=False)

    resolved = runtime.prepare_runtime_db(sample_db=sample_db, runtime_db=runtime_db)

    assert resolved == runtime_db
    assert runtime_db.exists()
    assert runtime_db.read_bytes() == sample_db.read_bytes()
    assert runtime_db.as_posix() == runtime.os.environ["DB_PATH"].replace("\\", "/")


def test_direct_service_routes_customer_queries_without_api(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat("show customers", "router")

    assert result["agent_used"] == "sales"
    assert "Acme Corp" in result["response"]
    assert result["execution_time"] >= 0


def test_direct_service_analytics_fallback_supports_revenue_queries(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat("revenue by month", "analytics")

    assert result["agent_used"] == "analytics"
    assert "Revenue" in result["response"]
    assert "2024-04" in result["response"]


def test_direct_service_routes_customer_count_questions_to_count_response(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat("how many customers do we have", "router")

    assert result["agent_used"] == "sales"
    assert "There are 2 customers in the database." in result["response"]


def test_direct_mode_respects_explicit_disable_flag(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ERP_ENABLE_DIRECT_AI", "false")

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)

    assert service._load_router_agent() is None
    assert service._load_sales_agent() is None
    assert service._load_analytics_agent() is None


def test_hosted_direct_mode_enables_ai_when_groq_key_is_present(monkeypatch):
    from backend import runtime

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("ERP_ENABLE_DIRECT_AI", raising=False)

    assert runtime._hosted_direct_ai_enabled() is True


def test_direct_service_analytics_supports_total_revenue_question(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat("what is our total revenue", "analytics")

    assert result["agent_used"] == "analytics"
    assert "Total Revenue" in result["response"]
    assert "$1197.99" in result["response"]


def test_direct_service_analytics_supports_worst_products_question(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat("worst 5 products by revenue", "analytics")

    assert result["agent_used"] == "analytics"
    assert "Worst Products by Revenue" in result["response"]
    assert "Service A" in result["response"]


def test_direct_service_prefers_hosted_analytics_agent_when_available(runtime_paths, monkeypatch):
    from backend.runtime import DirectERPService

    sample_db, runtime_db = runtime_paths
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("ERP_ENABLE_DIRECT_AI", raising=False)

    class FakeAnalyticsAgent:
        def invoke(self, payload):
            return {"output": f"AI analytics answer for: {payload['input']}"}

    fake_module = types.SimpleNamespace(create_analytics_agent=lambda: FakeAnalyticsAgent())
    monkeypatch.setitem(sys.modules, "backend.agents.AnalyticsAgent", fake_module)

    service = DirectERPService(sample_db=sample_db, runtime_db=runtime_db)
    result = service.chat("give me a plain english executive analysis of overall performance", "analytics")

    assert result["agent_used"] == "analytics"
    assert result["response"] == "AI analytics answer for: give me a plain english executive analysis of overall performance"
