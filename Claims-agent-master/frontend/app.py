import json
import urllib.request
from urllib.error import HTTPError, URLError

import streamlit as st

st.set_page_config(
    page_title="Claims + Policy Assistant",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE_URL = "http://localhost:8000"


def post_json(url: str, payload: dict, timeout: int = 20) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def safe_post(url: str, payload: dict) -> dict | None:
    try:
        return post_json(url, payload)
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            body = str(exc)
        st.error(f"HTTP {exc.code}: {body}")
    except URLError as exc:
        st.error(f"Connection error: {exc.reason}")
    except Exception as exc:
        st.error(f"Request failed: {exc}")
    return None


def initialize_session_state() -> None:
    defaults = {
        "member_id": "",
        "claim_id": "",
        "estimated_cost": 0.0,
        "claim_response": None,
        "transaction_status": None,
        "error_message": None,
        "policy_thread_id": None,
        "chat_messages": [],
        "pending_supervisor_action": False,
        "override_notes": "",
        "show_chat": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def append_chat_message(role: str, content: str) -> None:
    st.session_state.chat_messages.append({"role": role, "content": content})


def get_policy_response_text(state: dict) -> str:
    if not isinstance(state, dict):
        return ""
    for key in ("assistant", "response", "answer", "content", "reply"):
        if key in state and state[key]:
            return str(state[key])
    return ""


def handle_policy_query(user_query: str) -> None:
    append_chat_message("user", user_query)
    payload = {
        "user_query": user_query,
        "thread_id": st.session_state.policy_thread_id,
    }
    resp = safe_post(f"{API_BASE_URL}/policy/ask", payload)
    if not resp:
        append_chat_message("assistant", "The policy backend is unavailable.")
        return

    st.session_state.policy_thread_id = resp.get("thread_id", st.session_state.policy_thread_id)
    st.session_state.pending_supervisor_action = bool(resp.get("pending_supervisor_action", False))

    # state = resp.get("state", {})
    # if isinstance(state, dict) and isinstance(state.get("messages"), list):
    #     for item in state.get("messages", []):
    #         append_chat_message("assistant", str(item))
    # else:
    #     assistant_text = get_policy_response_text(state)
    #     if assistant_text:
    #         append_chat_message("assistant", assistant_text)
    #     else:
    #         append_chat_message("assistant", "The policy assistant returned an empty reply.")
    if "bot_response" in resp:
        assistant_text = resp["bot_response"]
        append_chat_message("assistant", assistant_text)
    else:
        # Fallback to your older dictionary structure just in case
        state = resp.get("state", {})
        if isinstance(state, dict) and isinstance(state.get("messages"), list):
            for item in state.get("messages", []):
                append_chat_message("assistant", str(item))
        else:
            assistant_text = get_policy_response_text(state)
            if assistant_text:
                append_chat_message("assistant", assistant_text)
            else:
                append_chat_message("assistant", "The policy assistant returned an empty reply.")


def handle_policy_override(notes: str) -> None:
    if not notes.strip():
        st.error("Please enter override notes before submitting.")
        return

    payload = {
        "thread_id": st.session_state.policy_thread_id,
        "supervisor_notes": notes.strip(),
    }
    resp = safe_post(f"{API_BASE_URL}/policy/override", payload)
    if not resp:
        return

    st.session_state.pending_supervisor_action = False
    st.session_state.override_notes = ""
    # state = resp.get("state", {})
    # assistant_text = get_policy_response_text(state)
    # if assistant_text:
    #     append_chat_message("assistant", assistant_text)
    # else:
    #     append_chat_message("assistant", "Supervisor override accepted and policy assistant resumed.")
    # st.success("Supervisor notes submitted.")
    if "bot_response" in resp:
        append_chat_message("assistant", resp["bot_response"])
    else:
        state = resp.get("state", {})
        assistant_text = get_policy_response_text(state)
        if assistant_text:
            append_chat_message("assistant", assistant_text)
        else:
            append_chat_message("assistant", "Supervisor override accepted and policy assistant resumed.")
            
    st.success("Supervisor notes submitted.")


def render_chat_history() -> None:
    st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
    if not st.session_state.chat_messages:
        st.info("Start a conversation by asking a policy question.")
    else:
        for message in st.session_state.chat_messages:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            if role == "user":
                st.markdown(
                    f"<div class='chat-message-user'><strong>You</strong><br>{content}</div>",
                    unsafe_allow_html=True,
                )
            elif role == "assistant":
                st.markdown(
                    f"<div class='chat-message-assistant'><strong>Policy Assistant</strong><br>{content}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-message-assistant'><strong>{role.title()}</strong><br>{content}</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)


def format_claim_status(status: str | None) -> str:
    mapping = {
        "underwriting_pending": "Underwriting review in progress",
        "underwriting_cleared": "Underwriting cleared",
        "approved": "Approved for settlement",
        "rejected": "Rejected by human review",
        "settled": "Settlement completed",
        "member_verified": "Member verified",
        "member_verification_failed": "Member verification failed",
        "initiated": "Claim initiated",
    }
    if not status:
        return "Pending"
    return mapping.get(status.lower(), status.replace("_", " ").capitalize())


def get_transaction_message(status_text: str, rejected_action: bool = False) -> str:
    if rejected_action:
        return "Claim declined — no settlement will be processed."
    if status_text == "approved":
        return "Claim approved — settlement will proceed."
    if status_text == "rejected":
        return "Claim declined — no settlement will be processed."
    if status_text == "settled":
        return "Claim settlement completed successfully."
    return f"Claim status updated to {format_claim_status(status_text)}."


def claims_panel() -> None:
    st.header("Claims Submission Interface")
    with st.form("claims_form"):
        cols = st.columns(3)
        member_id = cols[0].text_input("Member ID", value=st.session_state.member_id)
        claim_id = cols[1].text_input("Claim ID", value=st.session_state.claim_id)
        estimated_cost = cols[2].number_input(
            "Estimated Cost",
            min_value=0.0,
            value=float(st.session_state.estimated_cost or 0.0),
            step=100.0,
            format="%.2f",
        )

        submitted = st.form_submit_button("Submit Claim")

        if submitted:
            st.session_state.member_id = member_id
            st.session_state.claim_id = claim_id
            st.session_state.estimated_cost = estimated_cost
            st.session_state.error_message = None
            st.session_state.transaction_status = None

            payload = {
                "member_id": member_id,
                "claim_id": claim_id,
                "estimated_cost": estimated_cost,
            }
            resp = safe_post(f"{API_BASE_URL}/submit", payload)
            if resp:
                state = resp.get("state", {})
                st.session_state.claim_response = {
                    "thread_id": resp.get("thread_id"),
                    "status": state.get("status"),
                    "underwriting_score": state.get("underwriting_score"),
                    "pending_human_review": bool(resp.get("interrupted", False)),
                    "claim_id": state.get("claim_id", claim_id),
                    "member_id": state.get("member_id", member_id),
                    "estimated_cost": state.get("estimated_cost", estimated_cost),
                }

    st.subheader("Latest Claim Status")
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    if not st.session_state.claim_response:
        st.info("No claim has been submitted yet. Use the form above to send a claim to the backend.")
        return

    cr = st.session_state.claim_response
    status_cols = st.columns(4)
    status_cols[0].metric("Claim ID", cr.get("claim_id", "-"))
    status_cols[1].metric("Member ID", cr.get("member_id", "-"))
    status_cols[2].metric("Estimated Cost", f"${cr.get('estimated_cost', 0):,.2f}")
    status_cols[3].metric("Status", format_claim_status(cr.get("status", "Pending")))

    st.write("---")
    st.write(f"**Underwriting score:** {cr.get('underwriting_score', 'N/A')}  ")
    st.write(f"**Thread:** {cr.get('thread_id', '-')}")

    if cr.get("pending_human_review"):
        st.warning(
            f"⚠️ Manual review is required because underwriting score {cr.get('underwriting_score', 'N/A')} triggered a breakpoint."
        )
        approval_cols = st.columns(2)
        approved = approval_cols[0].button("Approve Settlement", key="approve_settlement")
        rejected = approval_cols[1].button("Reject Settlement", key="reject_settlement")

        if approved or rejected:
            payload = {"thread_id": cr.get("thread_id"), "approved": approved}
            resp = safe_post(f"{API_BASE_URL}/approve", payload)
            if resp:
                final_state = resp.get("state", {})
                status_text = final_state.get("status", "Unknown")
                rejected_action = rejected and not approved
                if rejected_action:
                    status_text = "rejected"
                st.session_state.transaction_status = get_transaction_message(status_text, rejected_action=rejected_action)
                if rejected_action:
                    st.error(st.session_state.transaction_status)
                else:
                    st.success(st.session_state.transaction_status)

                st.session_state.claim_response = {
                    "thread_id": cr.get("thread_id"),
                    "status": status_text,
                    "underwriting_score": final_state.get("underwriting_score", cr.get("underwriting_score")),
                    "pending_human_review": False,
                    "claim_id": final_state.get("claim_id", cr.get("claim_id")),
                    "member_id": final_state.get("member_id", cr.get("member_id")),
                    "estimated_cost": final_state.get("estimated_cost", cr.get("estimated_cost")),
                }
    else:
        status_display = cr.get("status", "Unknown")
        if status_display == "rejected":
            st.error(f"Claim result: {format_claim_status(status_display)}.")
        elif status_display == "approved" or status_display == "settled":
            st.success(f"Claim result: {format_claim_status(status_display)}.")
        else:
            st.info(f"Claim result: {format_claim_status(status_display)}.")

    if st.session_state.transaction_status and not cr.get("pending_human_review"):
        st.info(st.session_state.transaction_status)


def assistant_drawer() -> None:
    if not st.session_state.show_chat:
        return

    with st.sidebar:
        st.title("🤖 Policy Intel Assistant")
        st.caption("How may I assist you with policy queries?")
        render_chat_history()

        if st.session_state.pending_supervisor_action:
            st.warning("Supervisor review is required to continue this policy thread.")
            st.text_area(
                "Supervisor override notes",
                value=st.session_state.override_notes,
                key="override_notes",
                height=120,
                help="Describe the guidance or decision for the policy assistant.",
            )
            if st.button("Submit Supervisor Notes", key="submit_override"):
                handle_policy_override(st.session_state.override_notes)

        query = st.chat_input("Ask your policy question...")
        if query:
            handle_policy_query(query)


def toggle_chat_drawer() -> None:
    st.session_state.show_chat = not st.session_state.show_chat


def main() -> None:
    initialize_session_state()

    sidebar_transform = "translateX(0)" if st.session_state.show_chat else "translateX(100%)"
    st.markdown(
        f"""
        <style>
        section[data-testid="stSidebar"] {{
            position: fixed !important;
            top: 0;
            right: 0;
            left: auto !important;
            width: 25vw !important;
            max-width: 420px !important;
            height: 100vh !important;
            transform: {sidebar_transform} !important;
            transition: transform 0.3s ease !important;
            background: #ffffff !important;
            z-index: 999;
            box-shadow: -14px 0 32px rgba(0, 0, 0, 0.14);
        }}

        .chat-history {{
            max-height: 500px;
            overflow-y: auto;
            padding-right: 8px;
        }}

        .chat-message-user {{
            background: #fff4e1;
            border-radius: 18px 18px 6px 18px;
            padding: 14px 16px;
            margin: 10px 0;
            max-width: 85%;
            margin-left: auto;
            text-align: right;
            font-size: 0.95rem;
        }}

        .chat-message-assistant {{
            background: #e8f1ff;
            border-radius: 18px 18px 18px 6px;
            padding: 14px 16px;
            margin: 10px 0;
            max-width: 85%;
            text-align: left;
            font-size: 0.95rem;
        }}

        button[title="toggle-policy-chat"] {{
            position: fixed !important;
            right: 24px !important;
            bottom: 24px !important;
            z-index: 1100 !important;
            min-width: 190px !important;
            border-radius: 999px !important;
            background: #ff8c00 !important;
            color: #ffffff !important;
            font-size: 1rem !important;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.18) !important;
            border: none !important;
            padding: 14px 20px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Insurance Claims + Policy Assistant")
    st.write(
        "Submit claims, review manual settlement breakpoints, and open the policy chat assistant for real-time guidance."
    )

    claims_panel()
    assistant_drawer()

    st.button(
        "💬 Policy Help",
        key="toggle_chat_button",
        help="toggle-policy-chat",
        on_click=toggle_chat_drawer,
    )


if __name__ == "__main__":
    main()
