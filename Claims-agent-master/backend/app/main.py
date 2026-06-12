import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import uuid
import traceback

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

repo_root = Path(__file__).resolve().parents[2]
root_env_path = repo_root / ".env"
backend_env_path = repo_root / "backend" / ".env"
print(f"Looking for .env files at: {root_env_path} and {backend_env_path}")
if load_dotenv is not None:
    if root_env_path.exists():
        load_dotenv(root_env_path)
    elif backend_env_path.exists():
        load_dotenv(backend_env_path)

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "your_actual_langsmith_api_key")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "claim-bot-service")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://apac.api.smith.langchain.com")

os.environ.setdefault("LANGCHAIN_TRACING_V2", LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY)
os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)
os.environ.setdefault("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT)

from langsmith import traceable

from backend.agents.graph import app_graph
from backend.agents.policy import policy_graph

app = FastAPI(title="Claims State Graph API")#main application


@app.get("/health")
def health_check():
    return {"status": "ok"}


class ClaimSubmission(BaseModel):
    member_id: str
    claim_id: str
    estimated_cost: float


class Approval(BaseModel):
    thread_id: str
    approved: bool


class PolicyQuery(BaseModel):
    user_query: str
    thread_id: str | None = None


class PolicyOverride(BaseModel):
    thread_id: str
    supervisor_notes: str


def _serialize_messages(messages: Any) -> Any:
    if messages is None:
        return []
    serialized = []
    for m in messages:
        # BaseMessage-like objects have `content` attribute
        if hasattr(m, "content"):
            serialized.append(getattr(m, "content"))
        else:
            serialized.append(str(m))
    return serialized


def _serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not state:
        return out
    for k, v in state.items():
        if k == "messages":
            out[k] = _serialize_messages(v)
        else:
            out[k] = v
    return out


@app.post("/submit")
async def submit(claim: ClaimSubmission):
    thread_id = f"thread_{claim.member_id}_{claim.claim_id}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "claim_id": claim.claim_id,
        "member_id": claim.member_id,
        "estimated_cost": claim.estimated_cost,
        "member_verified": False,
        "underwriting_score": 0,
        "human_approved": False,
        "status": "Initiated",
        "messages": [],
    }

    current_state = dict(initial_state)
    interrupted = False
    pending_supervisor_action = False
    values = None

    try:
        snapshot = app_graph.get_state(config)
        values = getattr(snapshot, "values", None)
        stored_cost = values.get("estimated_cost") if isinstance(values, dict) else None
        next_node = getattr(snapshot, "next", None)

        if values is not None and stored_cost == claim.estimated_cost:
            pending_supervisor_action = next_node is not None
            return {
                "thread_id": thread_id,
                "interrupted": False,
                "pending_supervisor_action": pending_supervisor_action,
                "state": _serialize_state(values),
            }

        if values is not None and stored_cost != claim.estimated_cost:
            config = app_graph.update_state(
                config,
                {"estimated_cost": claim.estimated_cost, "status": "Initiated"},
                as_node="retrieve_policy",
            )
            for step in app_graph.stream(None, config=config):
                if "__interrupt__" in step:
                    interrupted = True
                    break
                for node_result in step.values():
                    if isinstance(node_result, dict):
                        current_state.update(node_result)

            try:
                snapshot = app_graph.get_state(config)
                values = getattr(snapshot, "values", current_state)
            except Exception:
                values = current_state

    except Exception:
        values = None

    if values is None:
        for step in app_graph.stream(initial_state, config=config):
            if "__interrupt__" in step:
                interrupted = True
                break
            for node_result in step.values():
                if isinstance(node_result, dict):
                    current_state.update(node_result)

        try:
            snapshot = app_graph.get_state(config)
            values = getattr(snapshot, "values", current_state)
        except Exception:
            values = current_state

    return {
        "thread_id": thread_id,
        "interrupted": interrupted,
        "pending_supervisor_action": pending_supervisor_action,
        "state": _serialize_state(values),
    }


@app.post("/approve")
async def approve(payload: Approval):
    config = {"configurable": {"thread_id": payload.thread_id}}

    # Apply manager decision to the graph state
    try:
        config = app_graph.update_state(config, {"human_approved": payload.approved})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update state: {e}")

    current_state: Dict[str, Any] = {}
    # Resume execution until completion
    for step in app_graph.stream(None, config=config):
        if "__interrupt__" in step:
            # should not hit another interrupt in this simple graph, but skip if it does
            continue
        for node_result in step.values():
            if isinstance(node_result, dict):
                current_state.update(node_result)

    try:
        snapshot = app_graph.get_state(config)
        values = getattr(snapshot, "values", current_state)
    except Exception:
        values = current_state

    return {"thread_id": payload.thread_id, "state": _serialize_state(values)}


@traceable(run_type="chain")
@app.post("/policy/ask")
async def policy_ask(query: PolicyQuery):
    # 1. Deterministic/Natural IDs are best, but if using UUID, compute it first
    thread_id = query.thread_id or f"policy_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        print(f"\n📥 [API Log]: Incoming query on thread {thread_id}: '{query.user_query}'")
        
        # Pull existing state snapshot to check for an active freeze
        historical_snapshot = policy_graph.get_state(config)
        
        # If no active history, trigger a fresh graph stream run
        if not historical_snapshot.next:
            print("🆕 Launching fresh policy graph stream lifecycle...")
            # We assign the stream to a variable so we can watch execution
            for chunk in policy_graph.stream({"user_query": query.user_query}, config=config, stream_mode="updates"):
                print(f"   ↳ [Node Execution]: {list(chunk.keys())}")
        else:
            print(f"🔄 Rehydrated existing frozen state session from node: {historical_snapshot.next}")

        # 2. Get the absolute latest, fully-merged snapshot state from the DB checkpointer
        final_snapshot = policy_graph.get_state(config)
        state_values = final_snapshot.values or {}
        
        # 3. Check if the graph is currently halted at the human_review breakpoint
        is_pending = "human_review" in final_snapshot.next
        
        # 4. Extract the actual text response safely
        bot_response = state_values.get("bot_response", "Processing your request...")
        if is_pending and not state_values.get("supervisor_notes"):
            bot_response = "⚠️ This scenario requires an official claims policy validation. Flagging for supervisor review..."

        # 5. Return a fully-formed payload that your Streamlit UI expects
        return {
            "thread_id": thread_id,
            "pending_supervisor_action": is_pending,
            "bot_response": bot_response,
            "status": state_values.get("status", "Initiated")
        }

    except Exception as e:
        # The crash firewall: Breaks background silences instantly
        print("\n❌ " + "!"*30 + " POLICY GRAPH API CRASH " + "!"*30)
        traceback.print_exc()
        print("❌ " + "!"*84 + "\n")
        raise HTTPException(status_code=500, detail=f"Backend Agent Execution Failure: {str(e)}")

@traceable(run_type="chain")
@app.post("/policy/override")
async def policy_override(payload: PolicyOverride):
    config = {"configurable": {"thread_id": payload.thread_id}}

    @traceable(
        name="policy_graph_update_state",
        run_type="chain",
        metadata={"thread_id": payload.thread_id, "supervisor_notes": payload.supervisor_notes},
    )
    def update_policy_state() -> Dict[str, Any]:
        return policy_graph.update_state(config, {"human_review": payload.supervisor_notes})

    try:
        config = update_policy_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update policy state: {e}")

    @traceable(name="policy_graph_resume", run_type="chain", metadata={"thread_id": payload.thread_id})
    def resume_policy_stream() -> Dict[str, Any]:
        current_state: Dict[str, Any] = {}
        for step in policy_graph.stream(None, config=config):
            if "__interrupt__" in step:
                continue
            for node_result in step.values():
                if isinstance(node_result, dict):
                    current_state.update(node_result)
        return current_state

    current_state = resume_policy_stream()

    try:
        snapshot = policy_graph.get_state(config)
        values = getattr(snapshot, "values", current_state)
    except Exception:
        values = current_state

    return {"thread_id": payload.thread_id, "state": _serialize_state(values)}


@app.get("/policy/pending")
async def policy_pending():
    pending_threads = []
    try:
        entries = policy_graph.list({})
    except Exception:
        entries = []

    for entry in entries:
        try:
            # entry may be a config dict or contain configurable
            cfg = entry if isinstance(entry, dict) else {}
            snapshot = policy_graph.get_state(cfg)
            values = getattr(snapshot, "values", {})
            if isinstance(values, dict) and "human_review" in values:
                tid = cfg.get("configurable", {}).get("thread_id") or values.get("thread_id")
                if tid:
                    pending_threads.append(tid)
        except Exception:
            continue

    return {"pending": pending_threads}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
