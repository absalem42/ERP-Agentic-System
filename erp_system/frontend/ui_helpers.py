from __future__ import annotations

from typing import Any


def build_navigation_items(pending_approval_count: int) -> list[dict[str, str]]:
    return [
        {"key": "chat", "label": "Chat", "badge": ""},
        {"key": "approvals", "label": "Approvals", "badge": str(pending_approval_count) if pending_approval_count else ""},
        {"key": "audit", "label": "Audit", "badge": ""},
        {"key": "reports", "label": "Reports", "badge": ""},
        {"key": "health", "label": "Health", "badge": ""},
    ]


def assistant_title(agent_used: str | None) -> str:
    if not agent_used:
        return "System"
    normalized = str(agent_used).strip().replace("_", " ")
    return normalized.title()


def build_status_tiles(health: dict[str, Any], pending_approval_count: int) -> list[dict[str, str]]:
    provider = provider_badge_text(health)
    return [
        {"label": "Router Ready", "value": "Ready"},
        {"label": "Pending Approvals", "value": str(pending_approval_count)},
        {"label": "Provider", "value": provider},
    ]


def sample_prompts_for_agent(agent: str) -> list[str]:
    prompts = {
        "router": [
            "Which products need replenishment soon, and what should I do next?",
            "Show me what happened across sales, finance, and inventory today.",
            "Create an invoice for the latest paid order and tell me if approval is needed.",
        ],
        "sales": [
            "Add a lead for New Horizon. Their email is sales@newhorizon.example and they want an urgent demo.",
            "Show my top customers and explain who looks ready for follow-up.",
            "Create a support ticket for Acme Corp asking for an invoice copy.",
        ],
        "finance": [
            "Post an invoice for customer 1 with two units at 100 each and link it to order 1.",
            "Record a bank transfer payment of 200 for invoice 1.",
            "Explain our finance policy for refunds and high-risk approvals.",
        ],
        "inventory": [
            "Reorder 10 units of product 2 from the best supplier.",
            "Show current stock risk and recommend what to buy next.",
            "Receive purchase order 1 for product 2 with 10 units today.",
        ],
        "analytics": [
            "What is the monthly revenue trend?",
            "Show top customers by revenue and explain what stands out.",
            "Run the monthly revenue report and visualize it.",
        ],
    }
    return prompts.get(agent, prompts["router"])


def build_workspace_metrics(
    health: dict[str, Any],
    approvals: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    saved_reports: list[dict[str, Any]],
) -> list[dict[str, str]]:
    pending_approvals = sum(1 for approval in approvals if approval.get("status") == "pending")
    provider_mode = str(health.get("provider_mode", "fallback")).upper()
    customer_count = str(health.get("customer_count", 0))
    tool_call_count = str(len(tool_calls))
    report_count = str(len(saved_reports))

    return [
        {"label": "Provider", "value": provider_mode, "caption": "Azure-backed reasoning mode"},
        {"label": "Customers", "value": customer_count, "caption": "Customer records available"},
        {"label": "Pending Approvals", "value": str(pending_approvals), "caption": "Actions waiting for sign-off"},
        {"label": "Logged Tool Calls", "value": tool_call_count, "caption": "Audited workflow actions"},
        {"label": "Saved Reports", "value": report_count, "caption": "Reusable analytics views"},
    ]


def summarize_tool_calls(tool_calls: list[dict[str, Any]], limit: int = 5) -> list[str]:
    summaries: list[str] = []
    for item in tool_calls[-limit:]:
        summaries.append(
            f"{item.get('agent', 'system')} -> {item.get('tool_name', 'unknown')}"
        )
    return list(reversed(summaries))


def provider_badge_text(health: dict[str, Any]) -> str:
    provider_mode = str(health.get("provider_mode", "fallback")).lower()
    if provider_mode == "azure":
        return "Azure OpenAI live"
    return "Fallback mode"
