#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from stereo_msgs.msg import DisparityImage
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

import cv2
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import transforms3d.quaternions as tf_quats # For odom quaternion -> matrix
import transforms3d.affines as tf_affines
import struct # For packing RGB

# Helper function to convert Open3D PointCloud to ROS2 PointCloud2
def o3d_to_ros2_pointcloud(o3d_cloud, frame_id, stamp):
    """Converts open3d.geometry.PointCloud to ROS2 sensor_msgs/PointCloud2."""
    points_np = np.asarray(o3d_cloud.points, dtype=np.float32)
    n_points = points_np.shape[0]

    # Check for colors
    has_colors = o3d_cloud.has_colors()
    if has_colors:
        colors_np = (np.asarray(o3d_cloud.colors) * 255).astype(np.uint8) # Convert 0-1 float to 0-255 uint8
        # Pack RGB into a single float32 (bytes: 0x00RRGGBB -> float)
        # Note: PointCloud2 field expects float32 for RGB visualization in RViz typically
        R = colors_np[:, 0].astype(np.uint32)
        G = colors_np[:, 1].astype(np.uint32)
        B = colors_np[:, 2].astype(np.uint32)
        rgb_packed_int = (R << 16) | (G << 8) | B
        rgb_packed_float = [struct.unpack('f', struct.pack('I', i))[0] for i in rgb_packed_int]
        rgb_packed_float_np = np.array(rgb_packed_float, dtype=np.float32)


    # Define fields
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 12 # Bytes per point (x, y, z)
    if has_colors:
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))
        point_step = 16 # Bytes per point (x, y, z, rgb)

    # Create data buffer
    # Create structured array first for easier assignment
    dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    if has_colors:
        dtype_list.append(('rgb', np.float32))

    structured_data = np.empty(n_points, dtype=dtype_list)
    structured_data['x'] = points_np[:, 0]
    structured_data['y'] = points_np[:, 1]
    structured_data['z'] = points_np[:, 2]
    if has_colors:
         structured_data['rgb'] = rgb_packed_float_np

    # Create PointCloud2 message
    msg = PointCloud2(
        header=Header(stamp=stamp, frame_id=frame_id),
        height=1,
        width=n_points,
        is_dense=False, # Usually False after filtering/downsampling
        is_bigendian=False,
        fields=fields,
        point_step=point_step,
        row_step=point_step * n_points,
        data=structured_data.tobytes()
    )
    return msg


class DisparityMappingNode(Node):
    def __init__(self):
        super().__init__('disparity_mapping_node')
        self.get_logger().info("Disparity Mapping Node Initializing...")

        # --- Parameters ---
        self.declare_parameter('disparity_topic', '/disparity')
        self.declare_parameter('color_topic', '/front_stereo_camera/left/image_rect_color')
        self.declare_parameter('info_topic', '/front_stereo_camera/left/camera_info')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('output_topic', '/map_points_from_disparity')
        self.declare_parameter('map_frame_id', 'odom') # Frame for the output map
        self.declare_parameter('voxel_size', 0.05) # Voxel size for downsampling map (meters)
        self.declare_parameter('map_publish_rate', 1.0) # Rate to publish map (Hz)
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('slop', 0.1) # Increased slop for 3 topics + odom

        disparity_topic = self.get_parameter('disparity_topic').value
        color_topic = self.get_parameter('color_topic').value
        info_topic = self.get_parameter('info_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.voxel_size = self.get_parameter('voxel_size').value
        map_publish_period = 1.0 / self.get_parameter('map_publish_rate').value
        queue_size = self.get_parameter('queue_size').value
        slop = self.get_parameter('slop').value

        # --- Internal State ---
        self.bridge = CvBridge()
        self.intrinsics_received = False
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.camera_frame_id = "camera_link" # Default, will be updated from info_msg

        # Open3D point cloud for accumulating the map
        self.map_o3d = o3d.geometry.PointCloud()
        self.map_needs_publishing = False

        # --- ROS Communications ---
        # Camera Info Subscriber (once)
        info_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, info_qos)

        # Synchronized Subscribers for Disparity, Color, Odom
        main_qos = QoSProfile(depth=queue_size, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.disp_sub = message_filters.Subscriber(self, DisparityImage, disparity_topic, qos_profile=main_qos)
        self.color_sub = message_filters.Subscriber(self, Image, color_topic, qos_profile=main_qos)
        self.odom_sub = message_filters.Subscriber(self, Odometry, odom_topic, qos_profile=main_qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.disp_sub, self.color_sub, self.odom_sub],
            queue_size=queue_size,
            slop=slop # Allow time difference
        )
        self.ts.registerCallback(self.data_callback)

        # PointCloud Publisher and Timer for map publishing
        self.map_pub = self.create_publisher(PointCloud2, self.output_topic, 1) # QoS 1 for map
        self.map_publish_timer = self.create_timer(map_publish_period, self.publish_map_callback)

        self.get_logger().info(f"Node initialized. Waiting for camera info on '{info_topic}'...")
        self.get_logger().info(f"Synchronizing: '{disparity_topic}', '{color_topic}', '{odom_topic}'")
        self.get_logger().info(f"Publishing map to '{self.output_topic}' in '{self.map_frame_id}' frame.")


    def info_callback(self, info_msg):
        """Store camera intrinsics and frame ID."""
        if not self.intrinsics_received:
            self.K = np.array(info_msg.k).reshape((3, 3))
            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]
            self.camera_frame_id = info_msg.header.frame_id
            self.intrinsics_received = True
            self.get_logger().info(f"Camera info received (fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, frame='{self.camera_frame_id}')")
            # Optional: Unsubscribe after receiving info
            # self.destroy_subscription(self.info_sub)
            # self.info_sub = None

    def data_callback(self, disp_msg, color_msg, odom_msg):
        """Process synchronized disparity, color, and odometry data."""
        if not self.intrinsics_received:
            self.get_logger().warn("Skipping data, camera intrinsics not yet received.", throttle_skip_first=False, throttle_duration_sec=5.0)
            return

        # --- Extract Pose ---
        # Get pose of camera frame in odom frame
        # Assumes odom_msg.child_frame_id is the camera or can be TF'd to it
        # For simplicity, we assume odom gives pose of camera_frame_id relative to map_frame_id
        if odom_msg.header.frame_id != self.map_frame_id:
             self.get_logger().warn_once(f"Odom frame '{odom_msg.header.frame_id}' doesn't match map frame '{self.map_frame_id}'. Assuming TF exists or pose is correct.")
        # TODO: Add TF lookup if odom_msg.child_frame_id != self.camera_frame_id

        q = odom_msg.pose.pose.orientation
        t = odom_msg.pose.pose.position
        # Convert quaternion (w, x, y, z) to rotation matrix (using transforms3d)
        try:
             R_odom_cam = tf_quats.quat2mat([q.w, q.x, q.y, q.z])
             t_odom_cam = np.array([t.x, t.y, t.z])
             T_odom_cam = tf_affines.compose(t_odom_cam, R_odom_cam, np.ones(3)) # Homogeneous transform
        except Exception as e:
             self.get_logger().error(f"Failed to convert odometry pose: {e}")
             return

        # --- Process Images ---
        try:
            disparity_image_raw = self.bridge.imgmsg_to_cv2(disp_msg.image, desired_encoding='passthrough')
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        # Check shapes
        if disparity_image_raw.shape[:2] != color_image.shape[:2]:
            self.get_logger().error("Disparity and color image shapes mismatch.")
            return

        # Ensure disparity is float32
        if disparity_image_raw.dtype != np.float32:
             # DisparityImage message standard is float32, but handle conversion just in case
             if disparity_image_raw.dtype == np.uint16 or disparity_image_raw.dtype == np.int16:
                   disparity_image = disparity_image_raw.astype(np.float32) / 16.0 # Example scaling if needed
                   self.get_logger().warn_once("Disparity image was not float32. Applied scaling /16.0. Adjust if necessary.")
             else:
                   self.get_logger().error(f"Unsupported disparity image type: {disparity_image_raw.dtype}")
                   return
        else:
             disparity_image = disparity_image_raw


        # --- Disparity to 3D ---
        f = disp_msg.f # Focal length from disparity message
        T = disp_msg.t # Baseline from disparity message

        if abs(f - self.fx) > 1e-3: # Use fx from camera info if significantly different
            # self.get_logger().warn_once(f"Focal length mismatch (DisparityMsg: {f}, CameraInfo: {self.fx}). Using CameraInfo.")
            f = self.fx

        if T <= 0:
             self.get_logger().error(f"Invalid baseline in DisparityImage message: {T}")
             return

        h, w = disparity_image.shape
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

        # Filter invalid disparities (using min/max from message and positive check)
        valid_disp_mask = (disparity_image > disp_msg.min_disparity) & \
                          (disparity_image < disp_msg.max_disparity) & \
                          (disparity_image > 1e-4) # Disparity must be positive

        # Calculate Depth Z = f * T / d
        Z = np.full_like(disparity_image, -1.0) # Initialize with invalid depth
        Z[valid_disp_mask] = (f * T) / disparity_image[valid_disp_mask]

        # Back-project to camera coordinates X, Y
        X = (u_coords - self.cx) * Z / self.fx
        Y = (v_coords - self.cy) * Z / self.fy

        # Final validity mask (include depth range check)
        valid_mask = valid_disp_mask & (Z > 0.1) & (Z < 20.0) # Example depth range 0.1m to 20m

        # Extract valid points and colors
        points_cam = np.vstack((X[valid_mask], Y[valid_mask], Z[valid_mask])).T.astype(np.float32)
        colors_cam = color_image[valid_mask].astype(np.float32) / 255.0 # Open3D expects 0-1 float colors

        if points_cam.shape[0] == 0:
             # self.get_logger().debug("No valid points generated from disparity image.")
             return # No points to add

        # --- Transform points to Map Frame ---
        # Convert points_cam to homogeneous coordinates (Nx4)
        points_cam_h = np.hstack((points_cam, np.ones((points_cam.shape[0], 1))))
        # Apply transformation: P_odom = T_odom_cam @ P_cam
        points_odom_h = (T_odom_cam @ points_cam_h.T).T
        # Convert back to non-homogeneous (Nx3)
        points_odom = points_odom_h[:, :3]

        # --- Add points to global map (Open3D) ---
        current_frame_o3d = o3d.geometry.PointCloud()
        current_frame_o3d.points = o3d.utility.Vector3dVector(points_odom)
        current_frame_o3d.colors = o3d.utility.Vector3dVector(colors_cam)

        self.map_o3d += current_frame_o3d # Efficiently add points
        self.map_needs_publishing = True
        # self.get_logger().debug(f"Added {points_cam.shape[0]} points to map.")


    def publish_map_callback(self):
        """Downsample the map and publish it if needed."""
        if not self.map_needs_publishing or len(self.map_o3d.points) == 0:
            return

        # --- Downsample Map ---
        map_downsampled = self.map_o3d.voxel_down_sample(self.voxel_size)
        # self.get_logger().info(f"Map downsampled from {len(self.map_o3d.points)} to {len(map_downsampled.points)} points (voxel size: {self.voxel_size:.3f})")
        self.map_o3d = map_downsampled # Update the stored map

        # --- Convert to ROS message ---
        stamp = self.get_clock().now().to_msg()
        map_msg = o3d_to_ros2_pointcloud(self.map_o3d, self.map_frame_id, stamp)

        # --- Publish ---
        self.map_pub.publish(map_msg)
        self.map_needs_publishing = False
        # self.get_logger().debug(f"Published map with {len(self.map_o3d.points)} points.")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = DisparityMappingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Disparity Mapping Node shutting down.")
    except Exception as e:
        if 'node' in locals() and isinstance(node, Node):
             node.get_logger().fatal(f"Unhandled exception: {e}")
        else:
             print(f"[FATAL] Unhandled exception during node initialization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals() and isinstance(node, Node) and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS Cleanup Complete.")


if __name__ == '__main__':
    main()
