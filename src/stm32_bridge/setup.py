from setuptools import setup
import os
from glob import glob

package_name = 'stm32_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cat',
    maintainer_email='cat@lubancat',
    description='Serial bridge between STM32 and ROS2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'stm32_bridge_node = stm32_bridge.bridge_node:main',
        ],
    },
)
