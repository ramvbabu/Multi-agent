from __future__ import annotations

from typing import Any, TypedDict
from typing_extensions import Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

import sqlite3

from backend.tools.claims_tools import assess_underwriting_risk, verify_member_api


def _append_messages(
    existing: list[BaseMessage] | None,
    incoming: list[BaseMessage] | None,
) -> list[BaseMessage]:
    if existing is None:
        existing = []
    if incoming is None:
        return existing
    return [*existing, *incoming]


class ClaimsState(TypedDict):
    claim_id: str
    member_id: str
    estimated_cost: float
    member_verified: bool
    underwriting_score: int
    human_approved: bool
    status: str
    messages: Annotated[list[BaseMessage], _append_messages]


def verify_member_node(state: ClaimsState, config: RunnableConfig) -> dict[str, Any]:
    tool_result = verify_member_api.invoke({"member_id": state["member_id"]})
    verified = bool(tool_result.get("status") == "Active" and "error" not in tool_result)
    message = HumanMessage(
        content=(
            f"Member {state['member_id']} verification "
            f"{'passed' if verified else 'failed'}: {tool_result.get('status', 'unknown')}"
        )
    )

    return {
        "member_verified": verified,
        "status": "member_verified" if verified else "member_verification_failed",
        "messages": [message],
    }


def underwriting_node(state: ClaimsState, config: RunnableConfig) -> dict[str, Any]:
    if not state["member_verified"]:
        return {
            "underwriting_score": 0,
            "status": "underwriting_skipped",
            "messages": [
                HumanMessage(content="Underwriting skipped because member verification failed.")
            ],
        }

    tool_result = assess_underwriting_risk.invoke(
        {"claim_amount": state["estimated_cost"], "member_id": state["member_id"]}
    )
    risk_level = tool_result.get("risk_level", "unknown")
    score = 85 if tool_result.get("requires_human_fallback") else 45
    tool_reason = tool_result.get("notes", "Underwriting completed without additional notes.")

    return {
        "underwriting_score": score,
        "status": "underwriting_pending" if score > 70 else "underwriting_cleared",
        "messages": [HumanMessage(content=f"Underwriting risk: {risk_level}. {tool_reason}")],
    }


def human_review_node(state: ClaimsState, config: RunnableConfig) -> dict[str, Any]:
    approved = state["human_approved"]
    status = "approved" if approved else "rejected"
    message = HumanMessage(
        content=(
            "Human review approved the claim." if approved else "Human review rejected the claim."
        )
    )
    return {"status": status, "messages": [message]}


def settlement_processing_node(state: ClaimsState, config: RunnableConfig) -> dict[str, Any]:
    return {
        "status": "settled",
        "messages": [HumanMessage(content="Settlement processing completed successfully.")],
    }


def route_after_member_check(state: ClaimsState) -> str:
    return "underwriting" if state["member_verified"] else END


def route_after_underwriting(state: ClaimsState) -> str:
    return "human_review" if state["underwriting_score"] > 70 else "settlement_processing"


def route_after_human_decision(state: ClaimsState) -> str:
    return "settlement_processing" if state["human_approved"] else END


builder = StateGraph(ClaimsState)

builder.add_node("verify_member", verify_member_node)
builder.add_node("underwriting", underwriting_node)
builder.add_node("human_review", human_review_node)
builder.add_node("settlement_processing", settlement_processing_node)

builder.set_entry_point("verify_member")
builder.add_conditional_edges("verify_member", route_after_member_check)
builder.add_conditional_edges("underwriting", route_after_underwriting)
builder.add_conditional_edges("human_review", route_after_human_decision)
#builder.add_edge("human_review", route_after_human_decision)
builder.set_finish_point("settlement_processing")

connection = sqlite3.connect("claims_system.db", check_same_thread=False)
memory = SqliteSaver(connection)
app_graph = builder.compile(checkpointer=memory, interrupt_before=["human_review"])


if __name__ == "__main__":
    sample_state: ClaimsState = {
        "claim_id": "CLM-22000",
        "member_id": "M999",
        "estimated_cost": 22000.0,
        "member_verified": False,
        "underwriting_score": 0,
        "human_approved": False,
        "status": "pending",
        "messages": [],
    }

    config = {"configurable": {"thread_id": sample_state["claim_id"]}}

    print("--- starting workflow stream until breakpoint ---")
    for step in app_graph.stream(sample_state, config=config):
        print(step)

    print("--- manual manager review simulated ---")
    config = app_graph.update_state(config, {"human_approved": True})

    print("--- resuming workflow after approval ---")
    for step in app_graph.stream(sample_state, config=config):
        print(step)

    print(f"Final status: {sample_state['status']}")
