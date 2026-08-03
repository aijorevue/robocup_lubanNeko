from setuptools import find_packages, setup
from glob import glob

package_name = 'ros2_test1'

CURRENT_CONSOLE_SCRIPTS = [
    'target_vision = ros2_test1.target_vision:main',
]

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['archive', 'archive.*', 'backups', 'backups.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.json')),
        ('share/' + package_name + '/urdf', glob('urdf/*.urdf')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/hook', [
            'hooks/ament_prefix_path.dsv',
            'hooks/ament_prefix_path.sh',
        ]),
    ],
    install_requires=['setuptools'],
    include_package_data=True,
    package_data={
        package_name: ['assets/*.png'],
    },
    zip_safe=True,
    maintainer='cat',
    maintainer_email='cat@lubancat',
    description='Selected colored target and QR detection with OpenCV',
    license='Apache-2.0',
    entry_points={
        'console_scripts': CURRENT_CONSOLE_SCRIPTS,
    },
)
