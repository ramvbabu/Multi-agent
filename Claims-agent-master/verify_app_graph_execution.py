import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import backend.app.claims_state_graph as cg

# Patch the imported tools to simulate a valid active member and a high-risk fallback path.
object.__setattr__(cg.verify_member_api, 'invoke', lambda payload: {
    'member_id': payload['member_id'],
    'name': 'MEM-10922',
    'policy_number': 'POL-10922',
    'status': 'Active',
    'policy_tier': 'Standard',
})
object.__setattr__(cg.assess_underwriting_risk, 'invoke', lambda payload: {
    'claim_amount': payload['claim_amount'],
    'member_profile': {'member_id': payload['member_id'], 'status': 'Active'},
    'risk_level': 'medium',
    'requires_human_fallback': True,
    'notes': 'Simulated fallback for a high-cost claim.',
})

config = {'configurable': {'thread_id': 'claim_session_101'}}

initial_state = {
    'claim_id': 'CLM-22000',
    'member_id': 'MEM-10922',
    'estimated_cost': 22000.0,
    'member_verified': False,
    'underwriting_score': 0,
    'human_approved': False,
    'status': 'Initiated',
    'messages': [],
}

current_state = dict(initial_state)

print('=== Phase 1: stream until human_review interrupt ===')
for step in cg.app_graph.stream(initial_state, config=config):
    print('STEP OUTPUT:', step)

    if '__interrupt__' in step:
        print('  Execution suspended before human_review')
        continue

    for node_result in step.values():
        if isinstance(node_result, dict):
            current_state.update(node_result)

    print(
        f"  status={current_state['status']}",
        f"underwriting_score={current_state['underwriting_score']}",
    )

snapshot = cg.app_graph.get_state(config)
print('\n=== Phase 1 State Snapshot ===')
print('next nodes:', snapshot.next)
print('snapshot status:', snapshot.values.get('status'))
print('snapshot underwriting_score:', snapshot.values.get('underwriting_score'))
print('suspended before human_review:', 'human_review' in snapshot.next)

print('\n=== Phase 2: apply manager approval and resume workflow ===')
config = cg.app_graph.update_state(config, {'human_approved': True})
current_state['human_approved'] = True
for step in cg.app_graph.stream(None, config=config):
    print('PHASE 2 STEP OUTPUT:', step)

    if '__interrupt__' in step:
        continue

    for node_result in step.values():
        if isinstance(node_result, dict):
            current_state.update(node_result)

    print(
        f"  status={current_state['status']}",
        f"underwriting_score={current_state['underwriting_score']}",
    )

print('\n=== Final claim state ===')
print(current_state)
