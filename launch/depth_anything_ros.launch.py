import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    depth_node = Node(
        package='depth_anything_ros',
        executable='depth_anything_node',
        name='depth_anything_node',
        output='screen',
        parameters=[os.path.join(get_package_share_directory('depth_anything_ros'), 
                'config', 'params.yaml')],
    )

    return LaunchDescription([
        depth_node
    ])