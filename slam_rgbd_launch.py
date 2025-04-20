# 3D SLAM System Starts from ROS2
# RTAB-Map Launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # RTAB-Map RGB-D SLAM Node (Extracts keypoints (corners, edges) from the RGB image)
        Node(
            package='rtabmap_ros',
            executable='rtabmap',
            name='rtabmap',
            output='screen',

            # Parameters passed directly to the node (SLAM Node Parameters)
            parameters=[{
                # Set the robot's base frame
                'frame_id': 'base_footprint',
                
                # Enables RGB-D processing (activates Depth Fusion)
                # Whether to subscribe to RGB and depth images
                'subscribe_depth': True,
                'subscribe_rgb': True,

                # Sync RGB and Depth (required for Depth Fusion)
                # Enable approximate time synchronization
                'approx_sync': True,
                # How many image pairs to buffer before matching
                'queue_size': 30,

                # Use simulation time (important for Isaac Sim)
                'use_sim_time': True,
            }],

            # Topic remappings to match Isaac Sim or your RGB-D sensor
            remappings=[
                # Depth Fusion Inputs (Back-projection into 3D)
                # RGB image topic
                ('rgb/image', '/camera/color/image_raw'),
                # Depth image topic
                ('depth/image', '/camera/depth/image_raw'),
                # RGB camera intrinsics (for projection)
                ('rgb/camera_info', '/camera/color/camera_info'),

                # Pose Estimation Input (Odometry + Frame-to-Frame Matching)
                # Odometry topic from Isaac Sim
                ('odom', '/odom')
            ]
        )
    ])



"""
✅ Result:

A live 3D map is published on /map 
(tabmap_ros/rtabmap is running with RGB-D input, 
  it automatically publishes a map)

The robot is localized within that map
"""