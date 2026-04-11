import os
from typing import Any, ClassVar, List, Optional

try:
    from langchain_core.language_models.llms import LLM
except Exception:
    try:
        from langchain.llms.base import LLM
    except Exception:
        class LLM:
            """Minimal fallback base class for environments without LangChain LLM base exports."""

            def invoke(self, prompt: str, **kwargs: Any):
                return self._call(prompt, **kwargs)

            def bind(self, **kwargs):
                return self

try:
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
except Exception:
    CallbackManagerForLLMRun = Any

try:
    from langchain_groq import ChatGroq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_ollama import OllamaLLM

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "20"))


class MockLLM(LLM):
    """Mock LLM for testing or no-provider fallback."""

    _call_count: ClassVar[int] = 0

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        MockLLM._call_count += 1
        prompt_lower = prompt.lower()

        if "thought:" in prompt_lower and "action:" in prompt_lower:
            question = ""
            if "question:" in prompt_lower:
                question_part = prompt_lower.split("question:")[-1]
                question = question_part.split("thought:")[0].strip() if "thought:" in question_part else question_part.strip()

            available_tools = []
            if "get_customers" in prompt_lower:
                available_tools.append("get_customers")
            if "get_orders" in prompt_lower:
                available_tools.append("get_orders")
            if "get_leads" in prompt_lower:
                available_tools.append("get_leads")
            if "execute_with_sales_agent" in prompt_lower:
                available_tools.append("execute_with_sales_agent")
            if "get_system_info" in prompt_lower:
                available_tools.append("get_system_info")
            if "classify_and_route" in prompt_lower:
                available_tools.append("classify_and_route")
            if "get_customer_summary" in prompt_lower:
                available_tools.append("get_customer_summary")
            if "search_customers" in prompt_lower:
                available_tools.append("search_customers")

            if "observation:" in prompt_lower and MockLLM._call_count > 1:
                return """Thought: I now have the final answer
Final Answer: Here is the information you requested based on the previous action."""

            if "execute_with_sales_agent" in available_tools:
                if "system" in question or "info" in question:
                    return """I should get system information.

Action: get_system_info
Action Input: """
                return f"""I need to route this request to the Sales Agent.

Action: execute_with_sales_agent
Action Input: {question}"""

            if "get_customers" in available_tools:
                if "summary" in question:
                    return """I should get customer summary statistics.

Action: get_customer_summary
Action Input: """
                if "search" in question or "find" in question:
                    search_term = question.replace("search", "").replace("find", "").replace("customer", "").strip()
                    return f"""I need to search for specific customers.

Action: search_customers
Action Input: {search_term}"""
                if "lead" in question:
                    return """I should get the list of leads.

Action: get_leads
Action Input: """
                if "order" in question:
                    return """I should get the list of orders.

Action: get_orders
Action Input: """
                return """I should get the list of customers.

Action: get_customers
Action Input: """

            if "customer" in question:
                return """I need to help with customer information.

Action: get_customers
Action Input: """

            return """I need to help with this request.

Final Answer: I can help you with various tasks. Please specify what you need assistance with."""

        return "Mock response: No external LLM provider is configured."

    def bind(self, **kwargs):
        MockLLM._call_count = 0
        return self


class DirectGroqLLM(LLM):
    """HTTP-backed Groq client for environments without langchain_groq."""

    api_key: str
    model: str = DEFAULT_GROQ_MODEL
    timeout: float = DEFAULT_GROQ_TIMEOUT_SECONDS
    temperature: float = 0.1

    @property
    def _llm_type(self) -> str:
        return "groq-http"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


def get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def has_llm_credentials() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def get_llm():
    """Get the configured LLM instance."""
    if has_llm_credentials():
        model = get_groq_model()
        if GROQ_AVAILABLE:
            try:
                print(f"Using Groq model: {model}")
                return ChatGroq(
                    model=model,
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0.1,
                    timeout=DEFAULT_GROQ_TIMEOUT_SECONDS,
                    max_retries=1,
                )
            except Exception as exc:
                print(f"Groq configuration error: {exc}")

        try:
            print(f"Using Groq model: {model}")
            return DirectGroqLLM(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model,
                timeout=DEFAULT_GROQ_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            print(f"Groq HTTP fallback error: {exc}")

    if OLLAMA_AVAILABLE:
        try:
            import requests

            for base_url in ("http://localhost:11434", "http://host.docker.internal:11434", "http://172.17.0.1:11434"):
                try:
                    response = requests.get(f"{base_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        print(f"Connected to Ollama at {base_url}")
                        return OllamaLLM(model="llama3.1:8b", base_url=base_url, temperature=0.1)
                except requests.RequestException:
                    continue
            print("Ollama not reachable from any configured URL")
        except Exception as exc:
            print(f"Ollama configuration error: {exc}")

    print("Using MockLLM for testing")
    return MockLLM()
