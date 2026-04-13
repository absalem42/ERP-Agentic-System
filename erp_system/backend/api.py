from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.runtime import get_direct_service


app = FastAPI(
    title="Helios Dynamics ERP API",
    description="Agent-driven ERP system API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    agent: str = "router"
    user_id: int | str = 1
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    tool_calls: list[dict[str, Any]]
    approval_required: dict[str, Any] | None = None
    chart_spec: dict[str, Any] | None = None
    rows: list[dict[str, Any]] | None = None
    execution_time: float


@app.get("/")
async def root():
    return {"message": "Helios Dynamics ERP API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
async def health():
    return get_direct_service().get_health()


@app.get("/agents")
async def list_agents():
    return {"agents": get_direct_service().list_agents()}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = get_direct_service().chat(
        request.message,
        request.agent,
        user_id=request.user_id,
        session_id=request.session_id,
    )
    return ChatResponse(**result)


@app.get("/approvals")
async def list_approvals(status: str | None = None):
    return {"approvals": get_direct_service().list_approvals(status=status)}


@app.post("/approvals/{approval_id}/approve")
async def approve_approval(approval_id: int, decided_by: str = "system"):
    approval = get_direct_service().approve_approval(approval_id, decided_by=decided_by)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return approval


@app.post("/approvals/{approval_id}/reject")
async def reject_approval(approval_id: int, decided_by: str = "system"):
    approval = get_direct_service().reject_approval(approval_id, decided_by=decided_by)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return approval


@app.get("/audit/tool-calls")
async def audit_tool_calls(limit: int = 50):
    return {"tool_calls": get_direct_service().list_tool_calls(limit=limit)}


@app.get("/saved-reports")
async def saved_reports():
    return {"saved_reports": get_direct_service().list_saved_reports()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
