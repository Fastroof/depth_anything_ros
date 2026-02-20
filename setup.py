import os
from glob import glob
from setuptools import setup

package_name = 'depth_anything_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, f'{package_name}.backends'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@todo.todo',
    description='Depth Anything V2 Metric ROS 2 Wrapper',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_anything_node = depth_anything_ros.main_node:main'
        ],
    },
)