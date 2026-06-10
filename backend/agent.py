from typing import TypedDict, List, Annotated
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
import json
import os

from knowledge_search import search_regulations, format_results
#─── LangSmith ───────────────────────────────────────────────────────────────
import os

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "civicflow")
# ─── Model ────────────────────────────────────────────────────────────────────

MODEL_NAME = "phi3.5:3.8b-mini-instruct-q4_K_M"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
    base_url="http://localhost:11434"
)


# ─── State ────────────────────────────────────────────────────────────────────
MAX_RETRIES = 2
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    plan: str
    research_findings: str
    filled_form: dict
    validation_status: str
    validation_errors: List[str]
    retry_count: int
    next_step: str

# ─── Nodes ────────────────────────────────────────────────────────────────────

def orchestrator_node(state: AgentState):
    if not state.get("plan"):
        user_request = state["messages"][-1].content
        prompt = (
            "Create a simple 3-step plan for registering this business:\n"
            "1. Research Regulations\n"
            "2. Fill Application\n"
            "3. Validate & Submit\n"
            "Output only the plan, no extra text."
        )
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_request)
        ])
        return {
            "plan": response.content,
            "next_step": "researcher",
            "messages": [AIMessage(content=f"Plan created:\n{response.content}")]
        }
    return {"next_step": state["next_step"]}


def researcher_node(state: AgentState):
    user_request = state["messages"][0].content
    results = search_regulations(user_request)
    findings = format_results(results)

    summary_prompt = (
        "You are a government regulations researcher. Based on the regulations below, "
        "write a brief summary of what permits, documents, and fees the applicant needs.\n\n"
        f"Regulations found:\n{findings}\n\n"
        "Summarize in 3-4 sentences. Be specific about fees and documents."
    )
    response = llm.invoke([
        SystemMessage(content=summary_prompt),
        HumanMessage(content=user_request)
    ])

    return {
        "research_findings": findings,
        "next_step": "filler",
        "messages": [AIMessage(content=f"Research complete:\n{response.content}")]
    }
def filler_node(state: AgentState):
    user_request = state["messages"][0].content
    findings = state["research_findings"]
    retry_count = state.get("retry_count", 0)
    validation_errors = state.get("validation_errors", [])

    # Increment only on retry (errors exist from a previous validation)
    if validation_errors:
        retry_count += 1

    prompt = (
        "You are a form-filling assistant. Extract the applicant's information "
        "from their request and the regulations below, then output ONLY valid JSON "
        "with no extra text, no markdown, no backticks.\n\n"
        f"Regulations:\n{findings}\n\n"
        "Required JSON format:\n"
        '{"applicant_name": "...", "business_type": "...", '
        '"location": "...", "fees_acknowledged": true/false, '
        '"documents_attached": ["doc1", "doc2"]}\n\n'
        "Rules:\n"
        "- Extract the applicant name from the request. If not provided, use 'Not specified'.\n"
        "- Business type must match the regulations found.\n"
        "- Documents list must include ALL documents from the regulations.\n"
        "- Set fees_acknowledged to true only if the request doesn't object to fees."
    )

    # On retry, inject validation errors so the LLM can correct
    if retry_count > 0:
        errors = state.get("validation_errors", [])
        error_str = "\n".join(f"  - {e}" for e in errors)
        prompt += (
            f"\n\nYour previous attempt had these errors:\n{error_str}\n"
            "Fix ALL of them this time."
        )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_request)
    ])

    # Parse the JSON from LLM response
    try:
        raw = response.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        form_data = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        form_data = {
            "applicant_name": "Not specified",
            "business_type": "Unknown",
            "location": "Not specified",
            "fees_acknowledged": False,
            "documents_attached": []
        }

    return {
        "filled_form": form_data,
        "next_step": "validator",
        "retry_count": retry_count,
        "messages": [AIMessage(content="Application form drafted.")]
    }
def validator_node(state: AgentState):
    form = state["filled_form"]
    errors = []

    # Check required fields exist and aren't empty
    required_fields = ["applicant_name", "business_type", "location"]
    for field in required_fields:
        value = form.get(field, "")
        if not value or value.strip().lower() in ("", "not specified", "unknown", "n/a"):
            errors.append(f"Missing or invalid field: {field}")

    # Check documents list
    docs = form.get("documents_attached", [])
    if not isinstance(docs, list) or len(docs) == 0:
        errors.append("Documents list is empty or invalid")

    # Check fees acknowledgment
    if not form.get("fees_acknowledged"):
        errors.append("Fees not acknowledged")

    # Cross-check against regulations if available
    findings = state.get("research_findings", "")
    if findings and isinstance(docs, list):
        findings_lower = findings.lower()
        if "health" in findings_lower and not any("health" in d.lower() for d in docs):
            errors.append("Missing required document: health-related certificate")
        if "fire" in findings_lower and not any("fire" in d.lower() for d in docs):
            errors.append("Missing required document: fire safety clearance")

    status = "FAIL" if errors else "PASS"

    return {
        "validation_status": status,
        "validation_errors": errors,
        "next_step": "human_approval",
        "messages": [AIMessage(
            content=f"Validation: {status}" + (f"\nIssues: {', '.join(errors)}" if errors else "")
        )]
    }

def human_approval_node(state: AgentState):
    return {
        "messages": [AIMessage(content="Awaiting human approval...")]
    }


def submission_node(state: AgentState):
    result = submit_application(state["filled_form"])
    return {
        "messages": [AIMessage(content=result)],
        "next_step": "end"
    }

def route_after_validation(state: AgentState) -> str:
    if state["validation_status"] == "PASS":
        return "human_approval"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "human_approval"
    return "filler"
# ─── Graph ────────────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("filler", filler_node)
workflow.add_node("validator", validator_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("submission", submission_node)

workflow.set_entry_point("orchestrator")
workflow.add_edge("orchestrator", "researcher")
workflow.add_edge("researcher", "filler")
workflow.add_edge("filler", "validator")
workflow.add_conditional_edges("validator", route_after_validation, {
    "human_approval": "human_approval",
    "filler": "filler"
})
workflow.add_edge("human_approval", "submission")
workflow.add_edge("submission", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])


# ─── Session helpers ──────────────────────────────────────────────────────────

def _config(session_id: str):
    return {"configurable": {"thread_id": session_id}}


def run_until_approval(session_id: str, user_request: str) -> dict:
    """Run graph until it hits the human_approval interrupt."""
    initial_state = {
        "messages": [HumanMessage(content=user_request)],
        "plan": "",
        "research_findings": "",
        "filled_form": {},
        "validation_status": "",
        "validation_errors": [],
        "retry_count": 0,
        "next_step": ""
        }

    steps = []
    final_state = {}

    for event in graph.stream(initial_state, _config(session_id), stream_mode="values"):
        final_state = event
        if event.get("messages"):
            msg = event["messages"][-1]
            if hasattr(msg, "content") and msg.content:
                steps.append(msg.content)

    return {
        "steps": steps,
        "form": final_state.get("filled_form", {}),
        "validation": final_state.get("validation_status", ""),
        "plan": final_state.get("plan", ""),
        "regulations": final_state.get("research_findings", ""),
        "status": "awaiting_approval"
    }


def resume_after_approval(session_id: str, approved: bool) -> dict:
    """Resume the graph after human makes a decision."""
    if not approved:
        return {"status": "cancelled", "message": "Submission cancelled by user."}

    messages = []
    for event in graph.stream(None, _config(session_id), stream_mode="values"):
        if event.get("messages"):
            msg = event["messages"][-1]
            if hasattr(msg, "content") and msg.content:
                messages.append(msg.content)

    final_message = messages[-1] if messages else "Process complete."
    return {
        "status": "submitted",
        "message": final_message
    }


def get_session_state(session_id: str) -> dict | None:
    """Return the current checkpoint state for a session."""
    state = graph.get_state(_config(session_id))
    if not state or not state.values:
        return None
    return {
        "form": state.values.get("filled_form", {}),
        "validation": state.values.get("validation_status", ""),
        "plan": state.values.get("plan", ""),
        "regulations": state.values.get("research_findings", ""),
    }
