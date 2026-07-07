import os
import select
import termios

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class STM32BridgeNode(Node):
    def __init__(self):
        super().__init__('stm32_bridge_node')

        self.declare_parameter('port_name', '/dev/ttyS0')
        self.declare_parameter('baud_rate', 115200)

        self.port_name = self.get_parameter('port_name').value
        self.baud_rate = int(self.get_parameter('baud_rate').value)

        self.fd = os.open(self.port_name, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        self._configure_serial()

        self.rx_buffer = ''
        self.create_subscription(String, '/servo_cmd', self.cmd_callback, 10)
        self.create_timer(0.02, self.read_serial)

        self.get_logger().info(f'serial bridge started: {self.port_name} @ {self.baud_rate}')

    def _baud_to_termios(self, baud_rate: int):
        mapping = {
            9600: termios.B9600,
            19200: termios.B19200,
            38400: termios.B38400,
            57600: termios.B57600,
            115200: termios.B115200,
        }
        if baud_rate not in mapping:
            raise ValueError(f'unsupported baud rate: {baud_rate}')
        return mapping[baud_rate]

    def _configure_serial(self):
        attrs = termios.tcgetattr(self.fd)
        speed = self._baud_to_termios(self.baud_rate)

        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8
        attrs[3] = 0
        attrs[4] = speed
        attrs[5] = speed
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0

        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)

    def cmd_callback(self, msg: String):
        payload = msg.data.strip()
        if not payload:
            return

        frame = payload + '\n'
        os.write(self.fd, frame.encode('utf-8'))
        self.get_logger().info(f'tx: {payload}')

    def read_serial(self):
        readable, _, _ = select.select([self.fd], [], [], 0)
        if not readable:
            return

        data = os.read(self.fd, 256)
        if not data:
            return

        self.rx_buffer += data.decode('utf-8', errors='replace')

        while '\n' in self.rx_buffer:
            line, self.rx_buffer = self.rx_buffer.split('\n', 1)
            line = line.strip('\r')
            if line:
                self.get_logger().info(f'rx: {line}')


def main(args=None):
    rclpy.init(args=args)
    node = STM32BridgeNode()
    try:
        rclpy.spin(node)
    finally:
        os.close(node.fd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
