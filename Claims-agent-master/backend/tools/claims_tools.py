import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import tool

DB_PATH = Path(__file__).resolve().parents[2] / "claims_system.db"


def _print_banner(message: str, char: str = "=", width: int = 80) -> None:
    print("\n" + char * width)
    print(message)
    print(char * width)


def _connect_db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@tool
def verify_member_api(member_id: str) -> Dict[str, Any]:
    """Fetch a member profile from the local claims_system.db members table."""
    _print_banner("START TOOL: verify_member_api", "#")
    print(f"INPUT: member_id={member_id!r}")

    conn = _connect_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT member_id, name, policy_number, status, policy_tier FROM members WHERE member_id = ?",
        (member_id,),
    )
    row = cursor.fetchone()

    if row:
        result = {
            "member_id": row[0],
            "name": row[1],
            "policy_number": row[2],
            "status": row[3],
            "policy_tier": row[4],
        }
        print(f"DB RETURNED: {result}")
    else:
        result = {
            "member_id": member_id,
            "error": "Member not found",
            "status": "unknown",
        }
        print(f"DB RETURNED: Member not found for member_id={member_id!r}")

    conn.close()
    print("EXIT TOOL: verify_member_api\n")
    return result


@tool
def assess_underwriting_risk(
    claim_amount: float,
    member_id: Optional[str] = None,
    policy_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate the underwriting risk of a claim amount and optionally enrich with member profile context."""
    _print_banner("START TOOL: assess_underwriting_risk", "#")
    print(f"INPUT: claim_amount={claim_amount!r}, member_id={member_id!r}, policy_number={policy_number!r}")

    profile: Optional[Dict[str, Any]] = None
    conn = _connect_db()
    cursor = conn.cursor()

    if member_id:
        cursor.execute(
            "SELECT member_id, name, policy_number, status, policy_tier FROM members WHERE member_id = ?",
            (member_id,),
        )
        row = cursor.fetchone()
        if row:
            profile = {
                "member_id": row[0],
                "name": row[1],
                "policy_number": row[2],
                "status": row[3],
                "policy_tier": row[4],
            }
    elif policy_number:
        cursor.execute(
            "SELECT member_id, name, policy_number, status, policy_tier FROM members WHERE policy_number = ?",
            (policy_number,),
        )
        row = cursor.fetchone()
        if row:
            profile = {
                "member_id": row[0],
                "name": row[1],
                "policy_number": row[2],
                "status": row[3],
                "policy_tier": row[4],
            }

    if profile:
        print(f"DB RETURNED: {profile}")
    else:
        print("DB RETURNED: No matching member profile found.")

    risk_level = "low"
    requires_human_fallback = False
    messages = []

    if claim_amount >= 50000:
        risk_level = "high"
        requires_human_fallback = True
        messages.append("Claim amount exceeds high-risk threshold.")
    elif claim_amount >= 20000:
        risk_level = "medium"
        messages.append("Claim amount falls in the medium-risk range.")
    else:
        risk_level = "low"
        messages.append("Claim amount falls in the low-risk range.")

    if profile and profile.get("status") != "Active":
        requires_human_fallback = True
        messages.append("Member policy is not Active; manual underwriting review recommended.")

    if profile and profile.get("policy_tier") == "Gold" and claim_amount < 20000:
        messages.append("Gold tier member with a lower claim amount; standard underwriting applies.")

    if not profile:
        requires_human_fallback = True
        messages.append("No valid member profile found for this claim; route to human review.")

    result = {
        "claim_amount": claim_amount,
        "member_profile": profile,
        "risk_level": risk_level,
        "requires_human_fallback": requires_human_fallback,
        "notes": " ".join(messages),
    }

    print(f"RESULT: {result}")
    conn.close()
    print("EXIT TOOL: assess_underwriting_risk\n")
    return result


if __name__ == "__main__":
    print("🚀 STARTING DAY 3 INDEPENDENT COMPONENT TESTING...")
    
    # Test Case 1: Active Member, Low Cost
    verify_member_api.invoke({"member_id": "M001"})
    assess_underwriting_risk.invoke({"member_id": "M001", "claim_amount": 4200.00})
    
    # Test Case 2: High Cost triggering the exact Guardrail failure you wanted!
    assess_underwriting_risk.invoke({"member_id": "POL-1004", "claim_amount": 22000.00})