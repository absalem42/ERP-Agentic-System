from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.runtime_env import bootstrap_runtime_environment
from frontend.ui_helpers import assistant_title, build_panel_visibility, build_status_tiles

bootstrap_runtime_environment(getattr(st, "secrets", None))


API_URL = os.getenv("API_URL", "").strip()
DIRECT_MODE = not API_URL

if DIRECT_MODE:
    from backend.runtime import get_direct_service

    direct_service = get_direct_service()
else:
    direct_service = None


st.set_page_config(
    page_title="ERP Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .main-header {
                color: #1f77b4;
                text-align: center;
                margin-bottom: 1.6rem;
            }
            .user-message {
                background-color: #e3f2fd;
                color: #1565c0;
                padding: 0.8rem 1rem;
                border-radius: 10px;
                margin: 0.8rem 0;
                margin-left: 2rem;
                border-left: 4px solid #2196f3;
                font-weight: 500;
            }
            .assistant-message {
                background-color: #fff3e0;
                color: #2e7d32;
                padding: 0.8rem 1rem;
                border-radius: 10px;
                margin: 0.8rem 0;
                margin-right: 2rem;
                border-left: 4px solid #4caf50;
                font-weight: 500;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            }
            .section-card {
                background: #f8fafc;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 1rem;
                margin-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_health() -> dict | None:
    try:
        if DIRECT_MODE:
            return direct_service.get_health()
        response = requests.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def list_approvals() -> list[dict]:
    if DIRECT_MODE:
        return direct_service.list_approvals()
    response = requests.get(f"{API_URL}/approvals", timeout=20)
    response.raise_for_status()
    return response.json()["approvals"]


def list_tool_calls() -> list[dict]:
    if DIRECT_MODE:
        return direct_service.list_tool_calls()
    response = requests.get(f"{API_URL}/audit/tool-calls", timeout=20)
    response.raise_for_status()
    return response.json()["tool_calls"]


def list_saved_reports() -> list[dict]:
    if DIRECT_MODE:
        return direct_service.list_saved_reports()
    response = requests.get(f"{API_URL}/saved-reports", timeout=20)
    response.raise_for_status()
    return response.json()["saved_reports"]


def call_agent(message: str, agent_name: str, user_id: int, session_id: str) -> dict:
    if DIRECT_MODE:
        return direct_service.chat(message, agent_name, user_id=user_id, session_id=session_id)
    response = requests.post(
        f"{API_URL}/chat",
        json={"message": message, "agent": agent_name, "user_id": user_id, "session_id": session_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def set_approval(approval_id: int, action: str) -> None:
    if DIRECT_MODE:
        if action == "approve":
            direct_service.approve_approval(approval_id, decided_by="streamlit")
        else:
            direct_service.reject_approval(approval_id, decided_by="streamlit")
        return
    response = requests.post(f"{API_URL}/approvals/{approval_id}/{action}", timeout=20)
    response.raise_for_status()


def render_chart(chart_spec: dict | None) -> None:
    if not chart_spec:
        return
    data = chart_spec.get("data") or []
    if not data:
        st.json(chart_spec)
        return
    frame = pd.DataFrame(data)
    x_column = chart_spec.get("x")
    y_column = chart_spec.get("y")
    st.caption(chart_spec.get("title", "Chart"))
    if x_column in frame.columns and y_column in frame.columns:
        chart_frame = frame[[x_column, y_column]].set_index(x_column)
        if chart_spec.get("type") == "line":
            st.line_chart(chart_frame)
        else:
            st.bar_chart(chart_frame)
    else:
        st.json(chart_spec)


inject_styles()

agent_mapping = {
    "Router Agent": "router",
    "Sales Agent": "sales",
    "Finance Agent": "finance",
    "Inventory Agent": "inventory",
    "Analytics Agent": "analytics",
}

agent_info = {
    "Router Agent": "🤖 Smart routing and system management",
    "Sales Agent": "🛍️ Customer and sales operations",
    "Finance Agent": "💼 Invoice, payment, and policy workflows",
    "Inventory Agent": "📦 Stock and procurement operations",
    "Analytics Agent": "📊 Data analysis and reporting",
}

verified_prompt_examples = {
    "Router Agent": "Show me this month revenue trend",
    "Sales Agent": "Create a new lead for Al Noor Trading, email sales@alnoor.com, interested in 500 units",
    "Finance Agent": "Post an invoice for customer 1 linked to order 1 for 15000 AED due on 2025-03-15",
    "Inventory Agent": "Reorder 20 units of product 2 from the best supplier",
    "Analytics Agent": "What are the top 5 products by revenue and why?",
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "Router Agent"
if "last_agent_used" not in st.session_state:
    st.session_state.last_agent_used = "router"
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-main"
if "user_id" not in st.session_state:
    st.session_state.user_id = 1
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

health_data = get_health()
approvals = list_approvals()
panel_visibility = build_panel_visibility(bool(st.session_state.debug_mode))

st.markdown('<h1 class="main-header">🚀 ERP Chat Assistant - Live Development!</h1>', unsafe_allow_html=True)

if not health_data:
    st.error("Agents not available. Please check the backend or runtime configuration.")
    st.stop()

debug_col, _ = st.columns([1, 5])
with debug_col:
    st.session_state.debug_mode = st.toggle("Debug mode", value=bool(st.session_state.debug_mode))
panel_visibility = build_panel_visibility(bool(st.session_state.debug_mode))

with st.sidebar:
    st.title("🎯 Select Agent")
    agent_choice = st.selectbox(
        "Choose an agent:",
        list(agent_mapping.keys()),
        index=list(agent_mapping.keys()).index(st.session_state.selected_agent),
    )
    if agent_choice != st.session_state.selected_agent:
        st.session_state.selected_agent = agent_choice
        st.session_state.messages = []
        st.session_state.last_agent_used = agent_mapping[agent_choice]

    if panel_visibility["show_identity_controls"]:
        st.session_state.user_id = int(
            st.number_input("User ID", min_value=1, value=int(st.session_state.user_id), step=1)
        )
        st.session_state.session_id = st.text_input("Session ID", value=st.session_state.session_id)

    if st.button("Clear Chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.last_agent_used = agent_mapping[st.session_state.selected_agent]
        st.rerun()

st.subheader(agent_info[st.session_state.selected_agent])
st.caption(f"Verified demo prompt: {verified_prompt_examples[st.session_state.selected_agent]}")

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f'<div class="user-message">👤 You: {message["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        rendered_agent = message.get("agent_label") or assistant_title(message.get("agent_used"))
        st.markdown(
            f'<div class="assistant-message">🤖 {rendered_agent}: {message["content"]}</div>',
            unsafe_allow_html=True,
        )
        render_chart(message.get("chart_spec"))
        if message.get("rows"):
            with st.expander("Structured Result"):
                st.dataframe(pd.DataFrame(message["rows"]), width="stretch")
        if panel_visibility["show_tool_calls"] and message.get("tool_calls"):
            with st.expander("Tool Calls"):
                st.json(message["tool_calls"])
        if panel_visibility["show_approval_details"] and message.get("approval_required"):
            with st.expander("Approval Details"):
                st.json(message["approval_required"])

user_input = st.text_input(
    "Ask a question:",
    key="chat_input",
    placeholder=verified_prompt_examples[st.session_state.selected_agent],
)

if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner(f"Getting response from {st.session_state.selected_agent}..."):
        result = call_agent(
            user_input,
            agent_mapping[st.session_state.selected_agent],
            int(st.session_state.user_id),
            st.session_state.session_id,
        )
    agent_used = str(result.get("agent_used", "router")).strip().lower() or "router"
    st.session_state.last_agent_used = agent_used
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("response", "No response received"),
            "agent_used": agent_used,
            "agent_label": assistant_title(agent_used),
            "chart_spec": result.get("chart_spec"),
            "rows": result.get("rows"),
            "tool_calls": result.get("tool_calls", []),
            "approval_required": result.get("approval_required"),
        }
    )
    st.rerun()

status_tiles = build_status_tiles(
    health_data,
    sum(1 for approval in approvals if approval.get("status") == "pending"),
    current_agent_label=str(st.session_state.last_agent_used),
)
status_columns = st.columns(len(status_tiles))
router_status = health_data.get("agents", {}).get("router", "unavailable")
selected_backend_agent = str(st.session_state.last_agent_used)
selected_status = health_data.get("agents", {}).get(selected_backend_agent, "unavailable")
provider_mode = str(health_data.get("provider_mode", "fallback")).lower()

for column, tile in zip(status_columns, status_tiles):
    with column:
        if tile["label"] == "Router Ready":
            if router_status == "available":
                st.success("✅ Router Ready")
            else:
                st.error("❌ Router Failed")
        elif tile["label"] == "Active Agent":
            if selected_status == "available":
                st.success(f"✅ Active Agent: {tile['value']}")
            else:
                st.error(f"❌ Active Agent Failed: {tile['value']}")
        elif tile["label"] == "Provider":
            if provider_mode == "azure":
                st.success("✅ Azure Connected")
            else:
                st.warning("⚠️ Fallback Mode")
        else:
            st.info(f"{tile['label']}: {tile['value']}")

if panel_visibility["show_approvals"]:
    with st.expander("Approvals Queue"):
        if not approvals:
            st.info("No approvals found.")
        for approval in approvals:
            st.markdown(
                f"**Approval #{approval['id']}** · `{approval['module']}` · `{approval['status']}`"
            )
            if panel_visibility["show_approval_details"]:
                st.json(approval["payload_json"])
            if approval["status"] == "pending":
                cols = st.columns(2)
                if cols[0].button("Approve", key=f"approve-{approval['id']}", width="stretch"):
                    set_approval(approval["id"], "approve")
                    st.rerun()
                if cols[1].button("Reject", key=f"reject-{approval['id']}", width="stretch"):
                    set_approval(approval["id"], "reject")
                    st.rerun()

if panel_visibility["show_audit_trail"]:
    with st.expander("Audit Trail"):
        tool_calls = list_tool_calls()
        if tool_calls:
            st.dataframe(pd.DataFrame(tool_calls), width="stretch")
        else:
            st.info("No tool calls logged yet.")

if panel_visibility["show_saved_reports"]:
    with st.expander("Saved Reports"):
        reports = list_saved_reports()
        if reports:
            for report in reports:
                st.markdown(f"**{report['title']}**")
                st.code(report["sql"])
        else:
            st.info("No saved reports yet.")

if panel_visibility["show_health"]:
    with st.expander("Health"):
        st.json(health_data)

if panel_visibility["show_footer_runtime"]:
    st.markdown("---")
    st.info(
        f"💬 Chat with {assistant_title(str(st.session_state.last_agent_used))} • {len(st.session_state.messages)} messages • "
        f"{'Direct runtime' if DIRECT_MODE else f'API: {API_URL}'}"
    )
