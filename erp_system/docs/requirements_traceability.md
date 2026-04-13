# ERP Requirements Traceability

This matrix links the Helios Dynamics assignment, the rubric, and the current implementation.

## Agent Coverage

| Area | Requirement | Current Coverage |
| --- | --- | --- |
| Router | Route to domain agents, track state, approvals, tool registry | Implemented via `backend/agents/simple_router_agent.py`, `classifier_tool`, `registry_tool`, conversations/messages state, and approvals |
| Sales | Leads, orders, tickets, contextual CRM replies | Implemented with SQL R/W, lead scoring, `customer_kv`, and sales RAG responses |
| Finance | Invoices, ledger updates, anomaly checks, approvals | Implemented for customer AR and vendor/AP bills, customer/vendor payments, journals, trial balance, anomaly gating, and approval execution |
| Inventory | Stock queries, reorder checks, forecasting, PO creation | Implemented with stock reads/writes, reorder-threshold queries, forecasting, purchase orders, and receipts |
| Analytics | NL→SQL, explainable insights, chart/dataframe output | Implemented with guarded read-only SQL, glossary/doc context, saved reports, ranking/trend queries, and chart specs |

## Governance And Observability

| Area | Requirement | Current Coverage |
| --- | --- | --- |
| Approvals | Risky finance/inventory actions must request approval | High-value customer invoices, vendor bills, and purchase orders create approval rows and execute safely after approval |
| Audit Trail | Log tool calls and actions | All registered tools log to `tool_calls`; UI debug mode surfaces logs |
| Memory | Conversation and global state in SQLite | `conversations`, `messages`, `approvals`, and `customer_kv` are persisted |

## Frontend And Deployment

| Area | Requirement | Current Coverage |
| --- | --- | --- |
| Streamlit UI | Chat, approvals, observability, reports | Implemented in `frontend/streamlit_app.py` with normal/debug modes |
| FastAPI | Backend API endpoints | Implemented in `backend/api.py` |
| Saved Reports | Reusable report execution | Existing `saved_reports` are visible and runnable from UI/API |
| Public Demo | Deployed proof of behavior | Streamlit Cloud deploy uses `codex/deploy-public-demo`; local acceptance target remains `127.0.0.1:8501` |

## Known Limits

- The sample database is historical, so current-month trend questions may correctly return a no-data explanation.
- Supplier email sending is still represented as a workflow stub rather than a live outbound integration.
- The UI is intentionally simple; the focus is functional completeness and traceability rather than visual complexity.
