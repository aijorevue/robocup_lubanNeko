import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path('/home/cat/ros2_ws')
stage = root / '.task1-timeout-20261016'
source = root / 'ros2_test1/ros2_test1/target_vision.py'
candidate = stage / 'target_vision.py'
before = '424466987d9e5f96546575b7b4502371a2ec76627f4e432f432c78b5018f3970'
after = 'e9e512c684b21d2875ca0e323a3624492de495f3001d3e310635678776979338'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
state = subprocess.run(['systemctl', 'is-active', 'robocup-chassis-arm.service'], capture_output=True, text=True).stdout.strip()
assert state == 'inactive', state
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (OSError, PermissionError):
        continue
    assert not any(name in args for name in (
        'ros2_test1/lib/ros2_test1/target_vision',
        'task2_secondary_h7_test_app.py', 'task3_main_h7_test_app.py',
        'ball_task_test_app.py')), args
assert digest(source) == before, 'Board source changed since review'
assert digest(candidate) == after, 'Candidate hash mismatch'
compile(candidate.read_text(), str(candidate), 'exec')
backup = root / 'saved_versions/20261016_task1_total_timeout_30s_before'
backup.mkdir(exist_ok=False)
shutil.copy2(source, backup / source.name)
temporary = source.with_name('.target_vision.task1-timeout-next.py')
shutil.copy2(candidate, temporary)
os.chmod(temporary, source.stat().st_mode)
os.replace(temporary, source)
assert digest(source) == after
print(json.dumps({'source': str(source), 'backup': str(backup), 'sha256': digest(source), 'service': state}))
