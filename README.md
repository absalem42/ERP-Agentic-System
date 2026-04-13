# ERP Agentic System

Helios Dynamics ERP is a local-first, agent-driven ERP prototype built around FastAPI, Streamlit, SQLite, and an in-process MCP-style tool registry. The current implementation ships one canonical app under `erp_system/` and covers all 5 rubric agents: Router, Sales, Finance, Inventory, and Analytics.

## Architecture

### Runtime
- `erp_system/backend/runtime.py`
  - shared runtime used by both FastAPI and direct Streamlit mode
  - copies a writable runtime database from `databases/erp_sample.db`
  - owns the router, sales, finance, inventory, and analytics agents

### Agents
- `router`
  - classifies prompts, tracks conversation state, enforces approval gating, and exposes tool registry/system status
- `sales`
  - customers, leads, orders, tickets, lead scoring, and customer entity memory
- `finance`
  - customer invoice posting, vendor/AP bill posting, payment allocation, journal posting, trial balance reads, policy lookup, and anomaly-aware approvals
- `inventory`
  - stock queries, reorder-threshold checks, stock movements, purchase orders, receipts, and simple demand forecasting
- `analytics`
  - read-only text-to-SQL, glossary/document context, saved reports, and chart specification output

### Data and memory
- SQLite schema is driven by `erp_system/databases/erp.db` and documented in `erp_system/databases/db.md`
- conversation and orchestration state uses:
  - `approvals`
  - `tool_calls`
  - `conversations`
  - `messages`
  - `users`
- domain tables follow the documented Router / Sales / Finance / Inventory / Analytics grouping

### MCP-style tools
- tools are registered in-process through `backend/mcp/mcp_adapter.py`
- every tool exposes:
  - `name`
  - `module`
  - `description`
  - `input_schema`
  - `read_only`
  - `requires_approval`
- every tool call is logged to `tool_calls`

## API

Main endpoints:
- `POST /chat`
- `GET /health`
- `GET /agents`
- `GET /approvals`
- `POST /approvals/{id}/approve`
- `POST /approvals/{id}/reject`
- `GET /audit/tool-calls`
- `GET /saved-reports`

## Streamlit UI

`erp_system/frontend/streamlit_app.py` supports two modes:
- direct mode
  - no `API_URL`
  - Streamlit imports the shared runtime directly
- split mode
  - `API_URL` set
  - Streamlit calls FastAPI

The UI exposes:
- chat
- approvals
- audit trail
- saved reports
- health
- release/provider marker for deploy verification

Verified demo prompts shown in the UI:
- router
  - `Show me this month revenue trend`
- sales
  - `Create a new lead for Al Noor Trading, email sales@alnoor.com, interested in 500 units`
- finance
  - `Post an invoice for customer 1 linked to order 1 for 15000 AED due on 2025-03-15`
- inventory
  - `Reorder 20 units of product 2 from the best supplier`
- analytics
  - `What are the top 5 products by revenue and why?`
  - `Run report Products Below ROP`

## Local Run

### FastAPI
```bash
cd erp_system
python -m uvicorn backend.api:app --reload --port 8000
```

### Streamlit direct mode
```bash
cd erp_system
streamlit run frontend/streamlit_app.py
```

### Docker
```bash
cd erp_system
docker compose up --build
```

## Environment

Copy `erp_system/.env.example` to `erp_system/.env` for shared defaults.
Put machine-specific secrets in `erp_system/.env.local`.

Primary hosted provider path:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_TARGET_URI`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`

This build accepts either:
- a base Azure endpoint plus deployment name
- or the full Azure chat completions target URI directly

If Azure credentials are absent, the app still runs with deterministic fallback logic.

## Verification

From `erp_system/`:
```bash
python -m pytest -q
pytest -q
```

Current verification target covers:
- runtime DB setup
- router persistence and audit logs
- approval gating
- sales writes and memory
- finance posting, vendor/AP flows, and balancing
- inventory procurement and reorder-threshold flows
- analytics read-only enforcement and saved reports
- API smoke endpoints
- Streamlit hosted env bootstrap

Additional traceability notes live in `erp_system/docs/requirements_traceability.md`.

## Presentation Notes

- The sample SQLite database is historical and fixed for demo purposes.
- Time-relative analytics questions use the real current calendar window first.
- If the sample database has no rows for the requested current period, the app explains that directly and reports the latest available sample period instead of silently widening the scope to all history.
- The current finance schema supports customer billing workflows through `invoices.customer_id`.
- Vendor/AP workflows are supported through additive runtime schema tables (`vendors`, `vendor_bills`, related payment/allocation tables) without replacing the customer billing model.
