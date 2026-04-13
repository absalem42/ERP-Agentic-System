from frontend.ui_helpers import (
    assistant_title,
    build_navigation_items,
    build_status_tiles,
    build_workspace_metrics,
    provider_badge_text,
    sample_prompts_for_agent,
    summarize_tool_calls,
)


def test_sample_prompts_for_agent_returns_natural_language_examples():
    prompts = sample_prompts_for_agent("inventory")

    assert len(prompts) == 3
    assert any("supplier" in prompt.lower() or "reorder" in prompt.lower() for prompt in prompts)


def test_build_workspace_metrics_uses_health_and_operational_lists():
    metrics = build_workspace_metrics(
        {"provider_mode": "azure", "customer_count": 12},
        [{"status": "pending"}, {"status": "approved"}],
        [{"tool_name": "a"}, {"tool_name": "b"}, {"tool_name": "c"}],
        [{"title": "Revenue"}],
    )

    labels = {metric["label"]: metric["value"] for metric in metrics}

    assert labels["Provider"] == "AZURE"
    assert labels["Customers"] == "12"
    assert labels["Pending Approvals"] == "1"
    assert labels["Logged Tool Calls"] == "3"
    assert labels["Saved Reports"] == "1"


def test_summarize_tool_calls_returns_latest_calls_in_reverse_chronological_order():
    summaries = summarize_tool_calls(
        [
            {"agent": "sales", "tool_name": "sales_query_tool"},
            {"agent": "finance", "tool_name": "post_invoice_tool"},
            {"agent": "analytics", "tool_name": "run_report_tool"},
        ],
        limit=2,
    )

    assert summaries == [
        "analytics -> run_report_tool",
        "finance -> post_invoice_tool",
    ]


def test_provider_badge_text_reflects_live_provider():
    assert provider_badge_text({"provider_mode": "azure"}) == "Azure OpenAI live"
    assert provider_badge_text({"provider_mode": "fallback"}) == "Fallback mode"


def test_build_navigation_items_sets_approval_badge_only_when_needed():
    items = build_navigation_items(2)
    labels = {item["key"]: item["badge"] for item in items}

    assert labels["chat"] == ""
    assert labels["approvals"] == "2"
    assert labels["health"] == ""


def test_assistant_title_normalizes_agent_names():
    assert assistant_title("simple_router_agent") == "Simple Router Agent"
    assert assistant_title("finance") == "Finance"
    assert assistant_title(None) == "System"


def test_build_status_tiles_uses_provider_and_approval_count():
    tiles = build_status_tiles({"provider_mode": "azure"}, 3)
    labels = {tile["label"]: tile["value"] for tile in tiles}

    assert labels["Router Ready"] == "Ready"
    assert labels["Pending Approvals"] == "3"
    assert labels["Provider"] == "Azure OpenAI live"
