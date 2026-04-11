import os
import sys
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.runtime_env import bootstrap_runtime_environment

bootstrap_runtime_environment(getattr(st, "secrets", {}))

from backend.runtime import get_direct_service

API_URL = os.getenv("API_URL")
RUNTIME_MODE = os.getenv("ERP_RUNTIME_MODE", "api" if API_URL else "direct")
DIRECT_SERVICE = None

if RUNTIME_MODE == "direct":
    DIRECT_SERVICE = get_direct_service()


def get_health_status():
    if RUNTIME_MODE == "api":
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            return False, None
        except Exception:
            return False, None

    return True, DIRECT_SERVICE.get_health()


def call_agent(message: str, agent_type: str) -> str:
    agent_mapping = {
        "Router Agent": "router",
        "Sales Agent": "sales",
        "Analytics Agent": "analytics",
    }
    agent_name = agent_mapping.get(agent_type, "router")

    if RUNTIME_MODE == "api":
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"message": message, "agent": agent_name},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response received")
            return f"Error: API returned status {response.status_code}"
        except requests.exceptions.Timeout:
            return "Error: Request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to backend API."
        except Exception as exc:
            return f"Error: {exc}"

    data = DIRECT_SERVICE.chat(message, agent_name)
    return data["response"]


AGENTS_AVAILABLE, health_data = get_health_status()

st.set_page_config(
    page_title="ERP Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-message {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.8rem 0 0.8rem 2rem;
        border-left: 4px solid #2196f3;
        font-weight: 500;
    }
    .assistant-message {
        background-color: #fff3e0;
        color: #2e7d32;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.8rem 2rem 0.8rem 0;
        border-left: 4px solid #4caf50;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-header">🚀 ERP Chat Assistant - Public Demo</h1>', unsafe_allow_html=True)

if not AGENTS_AVAILABLE:
    st.error("Agents not available. Please check the backend configuration.")
    if health_data:
        st.json(health_data)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "Router Agent"

st.sidebar.title("🎯 Select Agent")
agent_choice = st.sidebar.selectbox(
    "Choose an agent:",
    ["Router Agent", "Sales Agent", "Analytics Agent"],
    key="agent_selector",
)

if agent_choice != st.session_state.selected_agent:
    st.session_state.selected_agent = agent_choice
    st.session_state.messages = []

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

mode_label = "Direct Hosted Mode" if RUNTIME_MODE == "direct" else "API Mode"
st.sidebar.caption(f"Mode: {mode_label}")

if RUNTIME_MODE == "direct":
    llm_mode = health_data.get("llm_mode", "fallback") if health_data else "fallback"
    if llm_mode == "groq-hosted":
        st.sidebar.success("Live AI mode: Groq-backed hosted responses are enabled.")
    else:
        st.sidebar.warning("Fallback mode: hosted AI is disabled or no GROQ_API_KEY was loaded.")

    st.sidebar.info(
        "This public demo runs Streamlit directly against the ERP runtime layer. "
        "SQLite data is demo-grade and may reset between restarts."
    )

agent_info = {
    "Router Agent": "🤖 Smart routing and system management",
    "Sales Agent": "🛍️ Customer and sales operations",
    "Analytics Agent": "📊 Data analysis and reporting",
}

st.subheader(agent_info[agent_choice])

if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 You: {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🤖 {agent_choice}: {message["content"]}</div>', unsafe_allow_html=True)

user_input = st.text_input(
    "Ask a question:",
    key="chat_input",
    placeholder=f"Ask {agent_choice} something...",
)

if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner(f"Getting response from {agent_choice}..."):
        response = call_agent(user_input, agent_choice)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

if health_data:
    col1, col2, col3 = st.columns(3)
    with col1:
        if health_data.get("agents", {}).get("router") == "available":
            st.success("✅ Router Ready")
        else:
            st.error("❌ Router Failed")
    with col2:
        if health_data.get("agents", {}).get("sales") == "available":
            st.success("✅ Sales Ready")
        else:
            st.error("❌ Sales Failed")
    with col3:
        if health_data.get("agents", {}).get("analytics") == "available":
            st.success("✅ Analytics Ready")
        else:
            st.error("❌ Analytics Failed")

st.markdown("---")
footer_target = API_URL if RUNTIME_MODE == "api" else health_data.get("database_path", "direct-runtime")
st.info(f"💬 Chat with {agent_choice} • {len(st.session_state.messages)} messages • Runtime: {footer_target}")
