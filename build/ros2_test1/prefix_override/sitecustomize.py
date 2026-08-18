import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/aijokaren/Desktop/rk3588s/install/ros2_test1'
