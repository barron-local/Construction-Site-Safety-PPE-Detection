import sys, time
sys.path.append('Model-Deployment')
import checklog
from kiosk_logic import KioskSession

checklog.init()
checklog.add_employees('1001,วรเดช พันธ์พืช')
recorder = checklog.CheckRecorder()
kiosk = KioskSession()

# 1. Reset
recorder.clear()
kiosk.reset()

# 2. Enter valid code
recorder.set_code('1001')
print('Current employee:', recorder.current())

# 3. Trigger check (like api_kiosk_trigger)
kiosk.trigger_check()

# 4. Simulate frames with full PPE
dummy_dets = [
    {'class': 'Person', 'confidence': 0.90, 'bbox': [100, 50, 500, 650]},
    {'class': 'Hardhat', 'confidence': 0.88, 'bbox': [220, 60, 380, 200]},
    {'class': 'Safety Vest', 'confidence': 0.85, 'bbox': [150, 220, 450, 550]}
]

# Run 15 frames over 4.5 seconds
t0 = time.time()
for f in range(15):
    t = t0 + (f * 0.35)
    emp = recorder.current()
    is_valid = checklog.employee_exists(emp['code']) if emp else True
    snap = kiosk.update(dummy_dets, (720, 1280), settings={'kiosk_check_seconds': 4.0}, now=t, is_valid_employee=is_valid)

print('Final state after full PPE & valid ID:', snap['state'])
print('Items:', snap['items'])
print('Missing:', snap['missing'])
print('Invalid employee:', snap.get('invalid_employee'))
assert snap['state'] == 'pass', f"Expected pass but got {snap['state']}"
print('TEST 1 PASSED: Employee 1001 with full PPE successfully PASSED!')

# TEST 2: Unregistered employee code (e.g. 9999)
recorder.clear()
kiosk.reset()
recorder.set_code('9999')
kiosk.trigger_check()
t0 = time.time()
for f in range(15):
    t = t0 + (f * 0.35)
    emp = recorder.current()
    is_valid = checklog.employee_exists(emp['code']) if emp else True
    snap = kiosk.update(dummy_dets, (720, 1280), settings={'kiosk_check_seconds': 4.0}, now=t, is_valid_employee=is_valid)

print('Final state with invalid code 9999:', snap['state'])
print('Invalid employee:', snap.get('invalid_employee'))
assert snap['state'] == 'fail', f"Expected fail but got {snap['state']}"
assert snap.get('invalid_employee') == True
print('TEST 2 PASSED: Unregistered employee correctly FAILED!')
