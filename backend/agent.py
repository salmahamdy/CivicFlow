from typing import TypedDict, List, Annotated
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
import json
import os

from knowledge_search import search_regulations, format_results

# ─── LangSmith (only activates if an API key is configured) ──────────────────

if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "civicflow")
else:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

# ─── Model ────────────────────────────────────────────────────────────────────

MODEL_NAME = os.environ.get("OLLAMA_MODEL", "phi3.5:3.8b-mini-instruct-q4_K_M")

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
    base_url="http://localhost:11434"
)

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_RETRIES = 2

# Business types that require enhanced review before form filling
# (heavily regulated under Egyptian law)
SENSITIVE_TYPES = {"daycare", "restaurant", "pharmacy", "import_export"}

EXTRA_CHECKS = {
    "daycare": [
        "Criminal record certificates (فيش وتشبيه) for ALL staff",
        "Ministry of Social Solidarity director qualification check",
        "Staff-to-child ratio compliance verification",
    ],
    "restaurant": [
        "National Food Safety Authority (NFSA) inspection readiness",
        "Civil Defense fire safety compliance",
        "Health certificates validity for all food handlers",
    ],
    "pharmacy": [
        "Licensed pharmacist ownership verification (Pharmacy Practice Law)",
        "Minimum-distance compliance from existing pharmacies",
        "Egyptian Drug Authority (EDA) controlled-substances compliance",
    ],
    "import_export": [
        "Importers Registry capital requirements verification",
        "Egyptian-ownership percentage compliance (Law 121/1982)",
        "Nafeza customs platform registration",
    ],
}

# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    plan: str
    research_findings: str
    filled_form: dict
    validation_status: str
    validation_errors: List[str]
    retry_count: int
    matched_business_type: str
    next_step: str

# ─── Mock Government Services ────────────────────────────────────────────────

def submit_application(form_data: dict) -> str:
    ref_id = abs(hash(str(form_data))) % 10000
    return f"Application submitted successfully. Reference ID: #CFL-{ref_id:04d}"

# ─── Nodes ────────────────────────────────────────────────────────────────────

def orchestrator_node(state: AgentState):
    """Classify the business type (deterministic KB lookup) and generate
    an informed plan (LLM). The classification drives downstream routing."""
    user_request = state["messages"][0].content

    # 1. Classify: deterministic knowledge-base lookup
    results = search_regulations(user_request)
    matched_type = results[0]["business_type"] if results else "general"
    findings = format_results(results)

    # 2. Plan: LLM call, informed by the classification
    review_note = (
        "This business type requires ENHANCED REVIEW with additional checks."
        if matched_type in SENSITIVE_TYPES
        else "This business type follows the standard registration path."
    )
    prompt = (
        "You are a registration workflow planner. Create a brief 3-step plan "
        "for registering this business.\n\n"
        f"Detected business type: {matched_type}\n"
        f"{review_note}\n\n"
        "Steps to cover: research regulations, fill the application form, "
        "validate and submit for approval.\n"
        "Output only the numbered plan, no extra text."
    )
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_request)
    ])

    return {
        "plan": response.content,
        "matched_business_type": matched_type,
        "research_findings": findings,
        "next_step": "researcher",
        "messages": [AIMessage(content=f"Plan created (business type: {matched_type}):\n{response.content}")]
    }


def researcher_node(state: AgentState):
    """Summarize the matched regulations for the user (LLM).
    The lookup itself already happened in the orchestrator."""
    user_request = state["messages"][0].content
    findings = state["research_findings"]

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
        "next_step": "filler",
        "messages": [AIMessage(content=f"Research complete:\n{response.content}")]
    }


def enhanced_review_node(state: AgentState):
    """Deterministic extra checks for sensitive business types.
    Appends additional requirements to the findings so the filler
    and validator can see them."""
    btype = state["matched_business_type"]
    checks = EXTRA_CHECKS.get(btype, [])
    checks_text = "\n".join(f"- {c}" for c in checks)

    return {
        "research_findings": state["research_findings"]
            + f"\n\nENHANCED REVIEW REQUIREMENTS ({btype}):\n{checks_text}",
        "messages": [AIMessage(
            content=f"Enhanced review required for {btype}:\n{checks_text}"
        )]
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
    # NOTE: "Not specified" is a VALID value (user simply didn't provide it)
    required_fields = ["applicant_name", "business_type", "location"]
    for field in required_fields:
        value = form.get(field, "")
        if not value or str(value).strip().lower() in ("", "unknown", "n/a"):
            errors.append(f"Missing or invalid field: {field}")

    # Check documents list exists and has a reasonable number of items
    docs = form.get("documents_attached", [])
    if not isinstance(docs, list) or len(docs) == 0:
        errors.append("Documents list is empty or invalid")
    elif len(docs) < 2:
        errors.append("Too few documents listed — check regulations for full requirements")

    # Check fees acknowledgment
    if not form.get("fees_acknowledged"):
        errors.append("Fees not acknowledged")

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

# ─── Routing ──────────────────────────────────────────────────────────────────

def route_after_research(state: AgentState) -> str:
    """Sensitive business types go through enhanced review first.
    Decision is based on the deterministic KB classification, not LLM output."""
    if state.get("matched_business_type") in SENSITIVE_TYPES:
        return "enhanced_review"
    return "filler"


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
workflow.add_node("enhanced_review", enhanced_review_node)
workflow.add_node("filler", filler_node)
workflow.add_node("validator", validator_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("submission", submission_node)

workflow.set_entry_point("orchestrator")
workflow.add_edge("orchestrator", "researcher")
workflow.add_conditional_edges("researcher", route_after_research, {
    "enhanced_review": "enhanced_review",
    "filler": "filler"
})
workflow.add_edge("enhanced_review", "filler")
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
        "matched_business_type": "",
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
        "business_type": final_state.get("matched_business_type", ""),
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
        "business_type": state.values.get("matched_business_type", ""),
    }