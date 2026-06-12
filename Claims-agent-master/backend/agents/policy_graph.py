from __future__ import annotations

from typing import Any, TypedDict
from typing_extensions import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
import sqlite3

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from IPython.display import Image, display

from backend.tools.policy_tools import query_policy_rag


class PolicyState(TypedDict):
    user_query: str
    retrieved_context: str
    bot_response: str
    requires_human_override: bool
    supervisor_notes: str
    status: str
    messages: Annotated[list[BaseMessage], lambda e, i: e or []]


def retrieve_policy_node(state: PolicyState, config: RunnableConfig) -> dict[str, Any]:
    tool_result = query_policy_rag.invoke({"query": state["user_query"]})
    context = tool_result.get("context", "") if isinstance(tool_result, dict) else ""
    requires_human = bool(tool_result.get("requires_human_override", False)) if isinstance(tool_result, dict) else False

    msg = HumanMessage(content=("Retrieved policy context for query."))

    return {
        "retrieved_context": context,
        "requires_human_override": requires_human,
        "status": "context_retrieved",
        "messages": [msg],
    }


def generate_answer_node(state: PolicyState, config: RunnableConfig) -> dict[str, Any]:
    api_key = None
    try:
        api_key = ChatOpenAI.get_default_api_key()
    except Exception:
        pass

    prompt = (
        "You are a Policy Assistant. Use the retrieved policy context to answer the user's question.\n\n"
        f"Context:\n{state.get('retrieved_context','')}\n\n"
        f"User question:\n{state.get('user_query','')}\n\n"
        "Provide a concise, accurate answer and note when the policy is ambiguous."
    )

    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    response = chat.invoke(prompt)
    
    # 👑 THE CRITICAL FIX: Extract the raw text content string property
    # Do NOT use str(response). Use response.content!
    clean_text = response.content

    # Wrap it accurately as an AI Message output component
    msg = AIMessage(content=clean_text)

    print(f"\n🤖 [NODE SUCCESS]: Generated Response Text: '{clean_text[:60]}...'")

    return {
        "bot_response": clean_text,
        "status": "answered",
        "messages": [msg],
    }


def human_review_node(state: PolicyState, config: RunnableConfig) -> dict[str, Any]:
    note = state.get("supervisor_notes", "")
    msg = HumanMessage(content=(f"Supervisor notes: {note}"))
    return {"status": "human_review", "messages": [msg]}


def route_policy_flow(state: PolicyState) -> str:
    return "human_review" if state.get("requires_human_override") else "generate_answer"


builder = StateGraph(PolicyState)

builder.add_node("retrieve_policy", retrieve_policy_node)
builder.add_node("generate_answer", generate_answer_node)
builder.add_node("human_review", human_review_node)

builder.set_entry_point("retrieve_policy")
builder.add_conditional_edges("retrieve_policy", route_policy_flow)
builder.set_finish_point("generate_answer")



connection = sqlite3.connect("policy_graph.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(connection)
policy_graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
display(Image(policy_graph.get_graph().draw_mermaid_png()))

if __name__ == "__main__":
    sample: PolicyState = {
        "user_query": "Does our policy cover roadside assistance for flat tires?",
        "retrieved_context": "",
        "bot_response": "",
        "requires_human_override": False,
        "supervisor_notes": "",
        "status": "pending",
        "messages": [],
    }

    config = {"configurable": {"thread_id": "policy-test-001"}}

    print("--- start policy_graph stream until breakpoint ---")
    for step in policy_graph.stream(sample, config=config):
        print(step)
