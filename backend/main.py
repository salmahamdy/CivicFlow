from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import os

from agent import run_until_approval, resume_after_approval, get_session_state

app = FastAPI(title="CivicFlow", description="AI-powered business registration assistant")

# Serve frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))


# ─── Request / Response Models ───────────────────────────────────────────────

class StartRequest(BaseModel):
    request: str

class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.post("/api/start")
def start_registration(body: StartRequest):
    """Start the registration pipeline. Returns session_id and intermediate steps."""
    session_id = str(uuid.uuid4())[:8]
    result = run_until_approval(session_id, body.request)
    return {"session_id": session_id, **result}


@app.post("/api/approve")
def approve_submission(body: ApprovalRequest):
    """Resume the pipeline after human approval or rejection."""
    state = get_session_state(body.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    result = resume_after_approval(body.session_id, body.approved)
    return result


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    """Get current session state."""
    state = get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state
