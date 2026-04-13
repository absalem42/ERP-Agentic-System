import sqlite3
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    role TEXT DEFAULT 'user',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    module TEXT,
    payload_json TEXT,
    status TEXT DEFAULT 'pending',
    requested_by TEXT,
    decided_by TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    decided_at DATETIME
);

CREATE TABLE tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT,
    tool_name TEXT,
    input_json TEXT,
    output_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    agent_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER,
    sender TEXT,
    content TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    role TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE customer_kv (
    customer_id INTEGER,
    key TEXT,
    value TEXT,
    PRIMARY KEY (customer_id, key)
);

CREATE TABLE leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT,
    contact_email TEXT,
    message TEXT,
    score REAL,
    status TEXT DEFAULT 'new',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    description TEXT
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    total REAL NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL
);

CREATE TABLE tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    subject TEXT,
    body TEXT,
    status TEXT DEFAULT 'open',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    invoice_number TEXT,
    issue_date DATE,
    due_date DATE,
    total_amount REAL,
    status TEXT DEFAULT 'unpaid',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE invoice_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id INTEGER NOT NULL,
    description TEXT,
    quantity INTEGER,
    unit_price REAL
);

CREATE TABLE invoice_orders (
    invoice_id INTEGER,
    order_id INTEGER,
    PRIMARY KEY (invoice_id, order_id)
);

CREATE TABLE payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    amount REAL,
    method TEXT,
    received_at DATETIME
);

CREATE TABLE payment_allocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payment_id INTEGER,
    invoice_id INTEGER,
    amount REAL
);

CREATE TABLE chart_of_accounts (
    account TEXT PRIMARY KEY,
    description TEXT
);

CREATE TABLE ledger_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_date DATE NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ledger_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id INTEGER NOT NULL,
    account TEXT NOT NULL,
    debit REAL DEFAULT 0,
    credit REAL DEFAULT 0
);

CREATE TABLE stock (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    qty_on_hand INTEGER NOT NULL DEFAULT 0,
    reorder_point INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE stock_movements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    change_qty INTEGER NOT NULL,
    reason TEXT,
    ref_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE suppliers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT
);

CREATE TABLE supplier_products (
    supplier_id INTEGER,
    product_id INTEGER,
    lead_time_days INTEGER,
    default_cost REAL,
    PRIMARY KEY (supplier_id, product_id)
);

CREATE TABLE purchase_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    supplier_id INTEGER NOT NULL,
    status TEXT DEFAULT 'draft',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE po_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    po_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_cost REAL NOT NULL
);

CREATE TABLE po_receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    po_id INTEGER,
    product_id INTEGER,
    received_qty INTEGER,
    received_at DATETIME
);

CREATE TABLE saved_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    sql TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE glossary (
    term TEXT PRIMARY KEY,
    definition TEXT,
    module TEXT
);

CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    module TEXT,
    path TEXT,
    tags TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE model_registry (
    name TEXT,
    version TEXT,
    path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (name, version)
);

CREATE TABLE ml_features_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT,
    entity_id INTEGER,
    feature_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def create_test_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.executescript(SCHEMA_SQL)

    cursor.executemany(
        "INSERT INTO users (id, name, email, role, created_at) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "Alice Example", "alice@example.com", "admin", "2025-01-01 08:00:00"),
            (2, "Bob Example", "bob@example.com", "user", "2025-01-01 08:05:00"),
        ],
    )
    cursor.executemany(
        "INSERT INTO customers (id, name, email, phone, created_at) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "Acme Corp", "contact@acme.example", "+971500000001", "2025-01-02 09:00:00"),
            (2, "Globex LLC", "ops@globex.example", "+971500000002", "2025-01-03 10:00:00"),
        ],
    )
    cursor.executemany(
        "INSERT INTO products (id, sku, name, price, description) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "WIDGET-PRO", "Widget Pro", 100.0, "Flagship industrial widget"),
            (2, "SERVICE-A", "Service A", 150.0, "Implementation service"),
        ],
    )
    cursor.executemany(
        "INSERT INTO orders (id, customer_id, total, status, created_at) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, 300.0, "paid", "2025-01-05 11:00:00"),
            (2, 2, 150.0, "pending", "2025-02-10 12:00:00"),
        ],
    )
    cursor.executemany(
        "INSERT INTO order_items (id, order_id, product_id, quantity, price) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, 1, 2, 100.0),
            (2, 1, 2, 1, 100.0),
            (3, 2, 2, 1, 150.0),
        ],
    )
    cursor.executemany(
        "INSERT INTO leads (id, customer_name, contact_email, message, score, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "Wayne Enterprises", "bruce@wayne.example", "Interested in a demo and urgent pricing", None, "new", "2025-01-04 08:30:00"),
        ],
    )
    cursor.executemany(
        "INSERT INTO tickets (id, customer_id, subject, body, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "Delivery question", "Where is the shipment?", "open", "2025-01-06 09:30:00"),
        ],
    )
    cursor.executemany(
        "INSERT INTO chart_of_accounts (account, description) VALUES (?, ?)",
        [
            ("Cash", "Cash on hand"),
            ("Accounts Receivable", "Receivables from customers"),
            ("Revenue", "Sales revenue"),
        ],
    )
    cursor.executemany(
        "INSERT INTO stock (id, product_id, qty_on_hand, reorder_point) VALUES (?, ?, ?, ?)",
        [
            (1, 1, 12, 10),
            (2, 2, 2, 5),
        ],
    )
    cursor.executemany(
        "INSERT INTO stock_movements (id, product_id, change_qty, reason, ref_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, 1, -3, "sale", 1, "2025-01-05 11:00:00"),
            (2, 1, -2, "sale", 2, "2025-01-15 11:00:00"),
            (3, 2, -4, "sale", 2, "2025-02-10 12:00:00"),
        ],
    )
    cursor.executemany(
        "INSERT INTO suppliers (id, name, email, phone) VALUES (?, ?, ?, ?)",
        [
            (1, "Supply Hub", "sales@supplyhub.example", "+971400000001"),
            (2, "Northwind Parts", "ops@northwind.example", "+971400000002"),
        ],
    )
    cursor.executemany(
        "INSERT INTO supplier_products (supplier_id, product_id, lead_time_days, default_cost) VALUES (?, ?, ?, ?)",
        [
            (1, 1, 5, 60.0),
            (2, 1, 7, 58.0),
            (1, 2, 4, 90.0),
        ],
    )
    cursor.executemany(
        "INSERT INTO glossary (term, definition, module) VALUES (?, ?, ?)",
        [
            ("Revenue", "Income generated from orders and invoices.", "analytics"),
            ("Lead Score", "A probability-style score for conversion likelihood.", "sales"),
            ("EOQ", "Economic order quantity for replenishment planning.", "inventory"),
        ],
    )
    cursor.executemany(
        "INSERT INTO documents (id, module, path, tags, created_at) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "sales", "docs/sales/playbook.md", "playbook,crm", "2025-01-01 09:00:00"),
            (2, "finance", "docs/finance/refund_policy.pdf", "policy,refund", "2025-01-01 09:05:00"),
            (3, "inventory", "docs/inventory/supplier_terms.pdf", "contract,supplier", "2025-01-01 09:10:00"),
            (4, "analytics", "docs/analytics/definitions.md", "glossary,metrics", "2025-01-01 09:15:00"),
        ],
    )
    cursor.executemany(
        "INSERT INTO model_registry (name, version, path, created_at) VALUES (?, ?, ?, ?)",
        [
            ("intent_classifier", "v1", "models/intent_classifier.pkl", "2025-01-01 10:00:00"),
            ("finance_anomaly", "v1", "models/finance_anomaly.pkl", "2025-01-01 10:05:00"),
            ("inventory_forecast", "v1", "models/inventory_forecast.pkl", "2025-01-01 10:10:00"),
            ("lead_score", "v1", "models/lead_score.pkl", "2025-01-01 10:15:00"),
        ],
    )

    conn.commit()
    conn.close()


@pytest.fixture
def runtime_paths(tmp_path):
    sample_db = tmp_path / "sample" / "erp_sample.db"
    runtime_db = tmp_path / "runtime" / "erp_runtime.db"
    create_test_db(sample_db)
    return sample_db, runtime_db


@pytest.fixture(autouse=True)
def skip_repo_dotenv(monkeypatch):
    monkeypatch.setenv("ERP_SKIP_DOTENV", "1")
