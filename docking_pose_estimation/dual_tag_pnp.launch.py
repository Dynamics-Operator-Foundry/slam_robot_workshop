from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # Start dual-tag PnP node
        Node(
            package='dual_tag_pnp',
            executable='dual_tag_pnp_node',
            name='dual_tag_pnp_node',
            output='screen'
        ),

        # Start RViz2
        ExecuteProcess(
            cmd=['rviz2'],
            output='screen'
        )
    ])
