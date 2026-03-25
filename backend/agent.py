from typing import TypedDict, List, Annotated
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator

# ─── Model ────────────────────────────────────────────────────────────────────

MODEL_NAME = "phi3.5:3.8b-mini-instruct-q4_K_M"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
    base_url="http://localhost:11434"
)

# ─── Mock Government Services ─────────────────────────────────────────────────

def search_regulations(query: str) -> str:
    if "food" in query.lower():
        return (
            "Regulation 101: Food trucks require Health Permit ($200) "
            "and Zone License ($100). Documents needed: "
            "ID, Vehicle Registration, Health Certificate."
        )
    return "Regulation 102: General Business License requires ID and Tax ID. Fee: $50."


def submit_application(form_data: dict) -> str:
    ref_id = abs(hash(str(form_data))) % 10000
    return f"SUCCESS: Application submitted. Reference ID: #NTI-{ref_id:04d}"


# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    plan: str
    research_findings: str
    filled_form: dict
    validation_status: str
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
    findings = search_regulations(user_request)
    return {
        "research_findings": findings,
        "next_step": "filler",
        "messages": [AIMessage(content=f"Regulations found:\n{findings}")]
    }


def filler_node(state: AgentState):
    findings = state["research_findings"]
    form_data = {
        "applicant_name": "John Doe",
        "business_type": "Food Truck" if "Food" in findings else "General Business",
        "fees_acknowledged": True,
        "documents_attached": ["ID", "Vehicle Registration"]
    }
    return {
        "filled_form": form_data,
        "next_step": "validator",
        "messages": [AIMessage(content="Application form drafted.")]
    }


def validator_node(state: AgentState):
    form = state["filled_form"]
    is_valid = bool(form.get("fees_acknowledged"))
    status = "PASS" if is_valid else "FAIL"
    return {
        "validation_status": status,
        "next_step": "human_approval",
        "messages": [AIMessage(content=f"Validation: {status}")]
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
workflow.add_edge("validator", "human_approval")
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
