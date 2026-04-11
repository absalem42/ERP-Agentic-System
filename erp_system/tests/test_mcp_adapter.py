import io
import sys


def test_register_tool_handles_cp1252_stdout(monkeypatch):
    from backend.mcp.mcp_adapter import MCPAdapter

    stdout_buffer = io.BytesIO()
    stdout = io.TextIOWrapper(stdout_buffer, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stdout)

    adapter = MCPAdapter()
    adapter.register_tool("sample", lambda: "ok", "sample tool")
    stdout.flush()

    assert "sample" in stdout_buffer.getvalue().decode("cp1252")
