import hashlib
import json
from pathlib import Path
import subprocess
from ros2_test1 import target_vision

root = Path('/home/cat/ros2_ws')
paths = [root / 'ros2_test1/ros2_test1/target_vision.py',
         root / 'build/ros2_test1/ros2_test1/target_vision.py',
         Path(target_vision.__file__)]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
expected = 'e9e512c684b21d2875ca0e323a3624492de495f3001d3e310635678776979338'
assert all(value == expected for value in hashes.values()), hashes
assert target_vision.DISC_CATCH_TOTAL_TIMEOUT_S == 30.0
assert target_vision.DISC_CATCH_TARGET_TIMEOUT_S == 4.0
state = subprocess.run(['systemctl', 'is-active', 'robocup-chassis-arm.service'], capture_output=True, text=True).stdout.strip()
assert state == 'inactive', state
result = {'runtime': target_vision.__file__, 'hashes': hashes, 'total_timeout_s': 30.0,
          'no_ball_timeout_s': 4.0, 'service': state}
text = json.dumps(result, indent=2)
(root / '.task1-timeout-20261016/verified.json').write_text(text)
print(text)
