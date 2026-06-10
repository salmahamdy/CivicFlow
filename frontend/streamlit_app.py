import streamlit as st
from backend.agent import run_until_approval, resume_after_approval, get_session_state
import uuid

st.set_page_config(page_title="CivicFlow", page_icon="⚙", layout="centered")

# ── Session state init ───────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.pipeline_result = None
    st.session_state.final_result = None
    st.session_state.stage = "input"  # input → review → done

# ── Header ───────────────────────────────────────────────────
st.markdown("# ⚙ CivicFlow")
st.caption("AI-Powered Business Registration — Human in the Loop")
st.divider()

# ── Stage: Input ─────────────────────────────────────────────
if st.session_state.stage == "input":
    user_request = st.text_area(
        "Describe your business",
        placeholder="e.g. I want to register a food truck in downtown Portland",
        height=80
    )

    if st.button("Start Registration →", type="primary", disabled=not user_request):
        session_id = str(uuid.uuid4())[:8]
        st.session_state.session_id = session_id

        with st.spinner("Running AI pipeline..."):
            result = run_until_approval(session_id, user_request)

        st.session_state.pipeline_result = result
        st.session_state.stage = "review"
        st.rerun()

# ── Stage: Review ────────────────────────────────────────────
elif st.session_state.stage == "review":
    result = st.session_state.pipeline_result

    # Show pipeline steps
    steps = result.get("steps", [])
    step_labels = ["📋 Plan", "📜 Research", "📄 Form Drafted", "✅ Validation", "⏸ Awaiting Approval"]

    for i, step_text in enumerate(steps):
        label = step_labels[i] if i < len(step_labels) else f"Step {i+1}"
        with st.expander(label, expanded=(i >= len(steps) - 2)):
            st.text(step_text)

    st.divider()

    # Show form for review
    st.subheader("👤 Your Approval Required")

    form = result.get("form", {})
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Applicant:** {form.get('applicant_name', '—')}")
        st.markdown(f"**Business Type:** {form.get('business_type', '—')}")
        st.markdown(f"**Location:** {form.get('location', '—')}")

    with col2:
        st.markdown(f"**Validation:** {result.get('validation', '—')}")
        st.markdown(f"**Fees Acknowledged:** {'Yes' if form.get('fees_acknowledged') else 'No'}")
        docs = form.get("documents_attached", [])
        st.markdown(f"**Documents:** {', '.join(docs) if docs else '—'}")

    st.caption(f"Session ID: {st.session_state.session_id}")

    # Approval buttons
    col_approve, col_reject = st.columns(2)

    with col_approve:
        if st.button("✓ Approve & Submit", type="primary", use_container_width=True):
            with st.spinner("Submitting application..."):
                final = resume_after_approval(st.session_state.session_id, True)
            st.session_state.final_result = final
            st.session_state.stage = "done"
            st.rerun()

    with col_reject:
        if st.button("✗ Cancel", use_container_width=True):
            final = resume_after_approval(st.session_state.session_id, False)
            st.session_state.final_result = final
            st.session_state.stage = "done"
            st.rerun()

# ── Stage: Done ──────────────────────────────────────────────
elif st.session_state.stage == "done":
    final = st.session_state.final_result

    if final.get("status") == "submitted":
        st.success(final.get("message", "Submitted!"))
    else:
        st.error(final.get("message", "Cancelled."))

    if st.button("↩ Start Over"):
        st.session_state.session_id = None
        st.session_state.pipeline_result = None
        st.session_state.final_result = None
        st.session_state.stage = "input"
        st.rerun()
