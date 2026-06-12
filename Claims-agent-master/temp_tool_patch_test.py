import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import backend.app.claims_state_graph as cg
print('before invoke type', type(cg.verify_member_api.invoke))
try:
    cg.verify_member_api.invoke = lambda payload: {'status':'Active'}
    print('assignment succeeded')
except Exception as e:
    print('assignment failed', type(e), e)
try:
    object.__setattr__(cg.verify_member_api, 'invoke', lambda payload: {'status':'Active'})
    print('object setattr succeeded')
except Exception as e:
    print('object setattr failed', type(e), e)
print('after invoke type', type(cg.verify_member_api.invoke))
