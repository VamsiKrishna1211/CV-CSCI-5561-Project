# Example Launch File: ~/ros2_ws/src/stereo_slam_py/launch/slam.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='stereo_slam_py',
            executable='stereo_slam_node',
            name='stereo_slam_node',
            output='screen',
            parameters=[
                {'odom_frame': 'odom'},
                {'base_frame': 'camera_link'} # Match your robot's TF tree
            ]
            # Remap topics if necessary:
            # remappings=[
            #     ('/left/image_rect', '/my_camera/left/image_rect_color'),
            #     ('/right/image_rect', '/my_camera/right/image_rect'),
            #     ('/left/camera_info', '/my_camera/left/camera_info'),
            #     ('/right/camera_info', '/my_camera/right/camera_info'),
            #     ('/odom', '/visual_odom')
            # ]
        ),
        # Optionally launch RViz2 for visualization
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     arguments=['-d', '/path/to/your/slam_config.rviz'] # Load a config file
        # )
    ])
