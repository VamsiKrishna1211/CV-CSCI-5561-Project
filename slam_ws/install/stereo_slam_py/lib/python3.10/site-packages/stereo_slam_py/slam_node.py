#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Vector3
from std_msgs.msg import Header

import cv2
from cv_bridge import CvBridge
import numpy as np
# import open3d as o3d # Keep commented if not using visualization thread
import transforms3d.quaternions as tf_quats

from tf2_ros import TransformBroadcaster
# import threading # Keep commented if not using visualization thread


# --- Constants ---
FEATURE_DETECTOR = cv2.ORB_create(nfeatures=3000)
MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
MIN_STEREO_MATCHES = 10 # Adjusted back to a slightly more robust number
MIN_TEMPORAL_MATCHES = 15
MAX_MAP_POINTS = 10000
# VISUALIZATION_UPDATE_RATE = 5.0 # Commented out as visualization is disabled
# IMAGE_VIS_RATE = 10.0 # Commented out as visualization is disabled


# --- Helper Functions ---
def numpy_to_pointcloud2(points, parent_frame, stamp):
    """ Converts a Nx3 numpy array of points to a PointCloud2 message. """
    if points is None or points.shape[0] == 0:
        header = Header(stamp=stamp, frame_id=parent_frame)
        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
        return PointCloud2(header=header, height=1, width=0, fields=fields,
                           is_bigendian=False, point_step=12, row_step=0, is_dense=True)

    points = points.astype(np.float32)
    header = Header(stamp=stamp, frame_id=parent_frame)
    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]

    itemsize = np.dtype(np.float32).itemsize
    point_step = 3 * itemsize
    row_step = point_step * points.shape[0]

    return PointCloud2(
        header=header, height=1, width=points.shape[0], is_dense=True,
        is_bigendian=False, fields=fields, point_step=point_step,
        row_step=row_step, data=points.tobytes()
    )

# --- Main SLAM Node ---
class StereoSLAMNode(Node):
    def __init__(self):
        super().__init__('stereo_slam_node')
        self.get_logger().info("Initializing Stereo SLAM Node...")

        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value

        self.bridge = CvBridge()
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        self.left_img_sub = message_filters.Subscriber(self, Image, '/left/image_rect', qos_profile=qos_profile)
        self.right_img_sub = message_filters.Subscriber(self, Image, '/right/image_rect', qos_profile=qos_profile)
        self.left_info_sub = message_filters.Subscriber(self, CameraInfo, '/left/camera_info', qos_profile=qos_profile)
        self.right_info_sub = message_filters.Subscriber(self, CameraInfo, '/right/camera_info', qos_profile=qos_profile)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_img_sub, self.right_img_sub, self.left_info_sub, self.right_info_sub],
            queue_size=15, slop=0.05
        )
        self.ts.registerCallback(self.image_callback)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.map_pub = self.create_publisher(PointCloud2, '/map_points', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.initialized = False
        self.K_left = None
        self.K_right = None
        self.P_left = None
        self.P_right = None
        self.baseline = None
        self.current_pose_world = np.eye(4, dtype=np.float64)

        self.prev_frame = {'img_left': None, 'kp_left': None, 'des_left': None,
                           'points3d_cam': None, 'kp_indices': None}
        self.map_points_world = np.empty((0, 3), dtype=np.float64)

        # Direct visualization code removed/commented out previously remains removed.
        # Use RViz or rqt_image_view for visualization.

        self.get_logger().info("Stereo SLAM Node Initialized.")

    def get_camera_parameters(self, left_info_msg, right_info_msg):
        """Extracts camera intrinsic and projection matrices."""
        try:
            self.K_left = np.array(left_info_msg.k).reshape((3, 3))
            self.K_right = np.array(right_info_msg.k).reshape((3, 3))
            self.P_left = np.array(left_info_msg.p).reshape((3, 4))
            self.P_right = np.array(right_info_msg.p).reshape((3, 4))

            if self.P_right[0, 0] != 0:
                 self.baseline = -self.P_right[0, 3] / self.P_right[0, 0]
            else:
                 self.get_logger().error("Invalid P_right matrix (P[0,0] is zero), cannot calculate baseline.")
                 self.baseline = None
                 return False

            if self.baseline is None or self.baseline <= 1e-3: # Added small epsilon check
                self.get_logger().error(f"Invalid baseline calculated: {self.baseline}. Must be positive.")
                return False

            self.get_logger().info("Camera parameters received.")
            # Logging parameters can be verbose, uncomment if needed for debugging
            # self.get_logger().info(f" K_left:\n{self.K_left}")
            # self.get_logger().info(f" P_left:\n{self.P_left}")
            # self.get_logger().info(f" P_right:\n{self.P_right}")
            # self.get_logger().info(f" Calculated Baseline: {self.baseline:.4f}")
            return True
        except Exception as e:
            self.get_logger().error(f"Error processing camera info: {e}")
            return False


    def image_callback(self, left_img_msg, right_img_msg, left_info_msg, right_info_msg):
        """Main callback processing synchronized stereo images."""

        self.get_logger().debug("Processing new stereo image pair...")

        if rclpy.ok() is False: # Check if ROS context is still valid
             self.get_logger().warn("ROS context shut down, skipping callback.")
             return

        # --- Parameter Check & Initialization ---
        if not self.initialized:
            if self.K_left is None: # Try to get camera params only once
                 if not self.get_camera_parameters(left_info_msg, right_info_msg):
                      self.get_logger().warn("Waiting for valid camera parameters...")
                      return
            # If params are ready, initialize SLAM state on the first valid frame
            try:
                img_left_init = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='passthrough')
                # Convert to grayscale if needed (for ORB)
                if img_left_init.ndim == 3:
                    img_left_init = cv2.cvtColor(img_left_init, cv2.COLOR_BGR2GRAY if 'bgr' in left_img_msg.encoding else cv2.COLOR_RGB2GRAY) # Be explicit based on encoding if possible
                elif img_left_init.ndim != 2:
                    self.get_logger().error(f"Unsupported image dimensions for initialization: {img_left_init.ndim}")
                    return

                kp, des = FEATURE_DETECTOR.detectAndCompute(img_left_init, None)

                if des is None or len(kp) == 0:
                    self.get_logger().warn("No features detected in the first left image.")
                    return # Wait for a frame with features

                self.prev_frame['img_left'] = img_left_init
                self.prev_frame['kp_left'] = kp
                self.prev_frame['des_left'] = des
                self.current_pose_world = np.eye(4) # Start at origin
                self.initialized = True
                self.get_logger().info("SLAM Initialized.")
                return # Process starts from the second frame

            except Exception as e:
                self.get_logger().error(f"Error during initialization: {e}")
                self.initialized = False # Ensure reset if init fails
                return

        # --- Convert ROS Image messages to OpenCV images ---
        try:
            img_left = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='passthrough')
            img_right = self.bridge.imgmsg_to_cv2(right_img_msg, desired_encoding='passthrough')

            # Convert to grayscale if needed (for ORB)
            if img_left.ndim == 3:
                 img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY if 'bgr' in left_img_msg.encoding else cv2.COLOR_RGB2GRAY)
            elif img_left.ndim != 2:
                 self.get_logger().error(f"Unsupported left image dimensions: {img_left.ndim}")
                 return

            if img_right.ndim == 3:
                 img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY if 'bgr' in right_img_msg.encoding else cv2.COLOR_RGB2GRAY)
            elif img_right.ndim != 2:
                 self.get_logger().error(f"Unsupported right image dimensions: {img_right.ndim}")
                 return

            # DO NOT USE cv2.imshow or cv2.waitKey here! Use ROS tools for visualization.

        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        # --- Feature Detection ---
        try:
            kp_left_curr, des_left_curr = FEATURE_DETECTOR.detectAndCompute(img_left, None)
            kp_right_curr, des_right_curr = FEATURE_DETECTOR.detectAndCompute(img_right, None)
        except cv2.error as e:
            self.get_logger().error(f"Feature detection failed: {e}")
            return # Skip frame if detection fails

        if des_left_curr is None or len(kp_left_curr) < MIN_STEREO_MATCHES:
            self.get_logger().warn(f"Not enough features detected in left image ({len(kp_left_curr) if kp_left_curr is not None else 0}).")
            # Keep previous frame data, maybe reset pose if lost for too long?
            return
        if des_right_curr is None or len(kp_right_curr) == 0: # Need at least some features in right image
             self.get_logger().warn("No features detected in right image.")
             return

        # --- Stereo Matching (Current Frame Left <-> Right) ---
        try:
            stereo_matches = MATCHER.match(des_left_curr, des_right_curr)
            stereo_matches = sorted(stereo_matches, key=lambda x: x.distance)
        except cv2.error as e:
             self.get_logger().warn(f"Stereo matching failed: {e}. Skipping frame.")
             self._update_prev_frame(img_left, kp_left_curr, des_left_curr, None, None) # Update state but mark 3D points invalid
             return

        if len(stereo_matches) < MIN_STEREO_MATCHES:
            self.get_logger().warn(f"Not enough stereo matches ({len(stereo_matches)} < {MIN_STEREO_MATCHES}).")
            self._update_prev_frame(img_left, kp_left_curr, des_left_curr, None, None)
            return

        # --- Triangulation ---
        pts_left_stereo = np.float32([kp_left_curr[m.queryIdx].pt for m in stereo_matches]).reshape(-1, 1, 2)
        pts_right_stereo = np.float32([kp_right_curr[m.trainIdx].pt for m in stereo_matches]).reshape(-1, 1, 2)
        kp_indices_stereo = np.array([m.queryIdx for m in stereo_matches]) # Indices into kp_left_curr

        try:
            points4d_hom = cv2.triangulatePoints(self.P_left, self.P_right, pts_left_stereo, pts_right_stereo)
            w = points4d_hom[3]
            # Handle potential zero/negative 'w' more robustly
            valid_w_mask = w > 1e-6
            if not np.any(valid_w_mask): # No valid points
                self.get_logger().warn("Triangulation resulted in no valid 'w' values.")
                self._update_prev_frame(img_left, kp_left_curr, des_left_curr, None, None)
                return

            # Apply mask before division
            points4d_hom = points4d_hom[:, valid_w_mask]
            w = points4d_hom[3]
            kp_indices_stereo = kp_indices_stereo[valid_w_mask]

            points3d_cam_curr = (points4d_hom[:3] / w).T # Result is Nx3

            # Filter points by depth
            valid_depth_mask = (points3d_cam_curr[:, 2] > 0.1) & (points3d_cam_curr[:, 2] < 50.0)
            points3d_cam_curr = points3d_cam_curr[valid_depth_mask]
            kp_indices_stereo = kp_indices_stereo[valid_depth_mask]

        except cv2.error as e:
            self.get_logger().error(f"Triangulation failed: {e}")
            self._update_prev_frame(img_left, kp_left_curr, des_left_curr, None, None)
            return

        if len(points3d_cam_curr) < MIN_STEREO_MATCHES:
             self.get_logger().warn(f"Not enough valid 3D points after triangulation ({len(points3d_cam_curr)}).")
             self._update_prev_frame(img_left, kp_left_curr, des_left_curr, None, None)
             return

        # --- Temporal Matching & Pose Estimation ---
        pose_updated = False
        if self.prev_frame['des_left'] is not None and self.prev_frame['points3d_cam'] is not None and len(self.prev_frame['points3d_cam']) > 0:
            try:
                temporal_matches = MATCHER.match(self.prev_frame['des_left'], des_left_curr)
                temporal_matches = sorted(temporal_matches, key=lambda x: x.distance)
            except cv2.error as e:
                 self.get_logger().warn(f"Temporal matching failed: {e}. Assuming large motion/resetting.")
                 temporal_matches = []

            if len(temporal_matches) >= MIN_TEMPORAL_MATCHES:
                pts2d_curr = []
                pts3d_prev = []
                try: # Added try block for robust indexing
                    prev_kp_to_3d_map = {idx: pt for idx, pt in zip(self.prev_frame['kp_indices'], self.prev_frame['points3d_cam'])}
                    for m in temporal_matches:
                        prev_kp_idx = m.queryIdx
                        curr_kp_idx = m.trainIdx
                        if prev_kp_idx in prev_kp_to_3d_map: # Check if the previous keypoint has a valid 3D point
                             # Ensure indices are within bounds (though matcher should guarantee this if inputs non-empty)
                             if curr_kp_idx < len(kp_left_curr):
                                 pts3d_prev.append(prev_kp_to_3d_map[prev_kp_idx])
                                 pts2d_curr.append(kp_left_curr[curr_kp_idx].pt)
                             else:
                                  self.get_logger().warn(f"Current keypoint index {curr_kp_idx} out of bounds ({len(kp_left_curr)}).")
                except IndexError as e:
                     self.get_logger().error(f"Indexing error during temporal match processing: {e}")
                     # Continue without this match


                if len(pts2d_curr) >= MIN_TEMPORAL_MATCHES:
                    pts3d_prev_np = np.array(pts3d_prev, dtype=np.float64)
                    pts2d_curr_np = np.array(pts2d_curr, dtype=np.float64).reshape(-1, 1, 2)

                    try:
                        success, rvec, tvec, inliers = cv2.solvePnPRansac(
                            pts3d_prev_np, pts2d_curr_np, self.K_left, distCoeffs=None,
                            iterationsCount=100, reprojectionError=4.0, confidence=0.99,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )

                        if success and inliers is not None and len(inliers) >= 6:
                            R_delta_cv, _ = cv2.Rodrigues(rvec)
                            t_delta_cv = tvec.reshape(3, 1)
                            T_curr_prev = np.eye(4)
                            T_curr_prev[:3, :3] = R_delta_cv
                            T_curr_prev[:3, 3:] = t_delta_cv

                            try:
                                T_prev_curr = np.linalg.inv(T_curr_prev)
                                translation_norm = np.linalg.norm(T_prev_curr[:3, 3])

                                # --- Sanity check pose update ---
                                if translation_norm < 1.0: # Avoid huge jumps
                                    self.current_pose_world = self.current_pose_world @ T_prev_curr
                                    pose_updated = True
                                    # self.get_logger().debug(f"PnP successful with {len(inliers)} inliers. Pose updated.")
                                else:
                                    self.get_logger().warn(f"Large translation ({translation_norm:.2f}m) detected in PnP. Skipping pose update.")

                            except np.linalg.LinAlgError:
                                self.get_logger().warn("Singular matrix T_curr_prev inversion. Skipping pose update.")

                        else: # PnP failed or too few inliers
                             self.get_logger().warn(f"PnP failed or not enough inliers ({len(inliers) if inliers is not None else 'None'}). Keeping previous pose.")
                    except cv2.error as e:
                        self.get_logger().error(f"cv2.solvePnPRansac error: {e}. Keeping previous pose.")
                    except Exception as e: # Catch broader exceptions during PnP block
                         self.get_logger().error(f"Unexpected error during PnP processing: {e}")

                else: # Not enough valid 2D-3D correspondences
                     self.get_logger().warn(f"Not enough PnP correspondences ({len(pts2d_curr)} < {MIN_TEMPORAL_MATCHES}). Keeping previous pose.")
            else: # Not enough temporal matches
                self.get_logger().warn(f"Not enough temporal matches ({len(temporal_matches)} < {MIN_TEMPORAL_MATCHES}). Keeping previous pose.")
        # else: No previous data or no 3D points in previous frame
        #     self.get_logger().info("No previous 3D points available for temporal matching.")


        # --- Map Update ---
        if pose_updated or len(self.map_points_world) == 0: # Update map if pose changed or map is empty
             R_world_cam = self.current_pose_world[:3, :3]
             t_world_cam = self.current_pose_world[:3, 3]
             if points3d_cam_curr.ndim == 2 and points3d_cam_curr.shape[1] == 3:
                 new_points_world = (R_world_cam @ points3d_cam_curr.T).T + t_world_cam # Broadcasting
                 self.map_points_world = np.vstack((self.map_points_world, new_points_world))
                 if len(self.map_points_world) > MAX_MAP_POINTS:
                     self.map_points_world = self.map_points_world[-MAX_MAP_POINTS:]
             # else: Warning about shape already handled during triangulation


        # --- Publish Odometry and TF ---
        # Use the timestamp from the *input* message for consistency
        current_stamp = left_img_msg.header.stamp
        self.publish_odom_tf(self.current_pose_world, current_stamp)

        # --- Publish Map Points ---
        map_msg = numpy_to_pointcloud2(self.map_points_world, self.odom_frame, current_stamp)
        self.map_pub.publish(map_msg)

        # --- Update State for Next Iteration ---
        self._update_prev_frame(img_left, kp_left_curr, des_left_curr, points3d_cam_curr, kp_indices_stereo)


    def _update_prev_frame(self, img, kp, des, pts3d, kp_indices):
         """Helper to update the state for the next iteration."""
         self.prev_frame['img_left'] = img
         self.prev_frame['kp_left'] = kp
         self.prev_frame['des_left'] = des
         self.prev_frame['points3d_cam'] = pts3d
         self.prev_frame['kp_indices'] = kp_indices


    def publish_odom_tf(self, pose_matrix_world_cam, stamp):
        """Publishes odometry message and TF transform."""
        if not np.all(np.isfinite(pose_matrix_world_cam)):
             self.get_logger().error("Attempted to publish non-finite pose! Skipping.")
             return

        t = pose_matrix_world_cam[:3, 3]
        R_mat = pose_matrix_world_cam[:3, :3]

        try:
            # Ensure Rotation matrix is valid before converting
            if abs(np.linalg.det(R_mat) - 1.0) > 1e-3: # Check determinant is close to 1
                 self.get_logger().warn(f"Invalid rotation matrix determinant: {np.linalg.det(R_mat)}. Attempting normalization.")
                 # Normalize rows (simple method, SVD is more robust but slower)
                 for i in range(3):
                      norm = np.linalg.norm(R_mat[i, :])
                      if norm > 1e-6:
                           R_mat[i, :] /= norm
                      else: # Handle zero row case - unlikely but possible
                           self.get_logger().error("Zero row found in rotation matrix. Resetting to identity.")
                           R_mat = np.eye(3)
                           break
                 # Optionally re-orthogonalize using SVD if issues persist:
                 # U, _, Vt = np.linalg.svd(R_mat)
                 # R_mat = U @ Vt

            q_wxyz = tf_quats.mat2quat(R_mat)
        except Exception as e: # Catch LinAlgError and others during conversion
             self.get_logger().warn(f"Quaternion conversion failed: {e}. Using identity quaternion.")
             q_wxyz = np.array([1.0, 0.0, 0.0, 0.0])


        # --- Publish TF ---
        t_tf = TransformStamped()
        t_tf.header.stamp = stamp
        t_tf.header.frame_id = self.odom_frame
        t_tf.child_frame_id = self.base_frame
        t_tf.transform.translation.x = float(t[0]) # Ensure float type
        t_tf.transform.translation.y = float(t[1])
        t_tf.transform.translation.z = float(t[2])
        t_tf.transform.rotation.w = float(q_wxyz[0])
        t_tf.transform.rotation.x = float(q_wxyz[1])
        t_tf.transform.rotation.y = float(q_wxyz[2])
        t_tf.transform.rotation.z = float(q_wxyz[3])
        self.tf_broadcaster.sendTransform(t_tf)

        # --- Publish Odometry Message ---
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        odom_msg.pose.pose.position.x = float(t[0])
        odom_msg.pose.pose.position.y = float(t[1])
        odom_msg.pose.pose.position.z = float(t[2])
        odom_msg.pose.pose.orientation.w = float(q_wxyz[0])
        odom_msg.pose.pose.orientation.x = float(q_wxyz[1])
        odom_msg.pose.pose.orientation.y = float(q_wxyz[2])
        odom_msg.pose.pose.orientation.z = float(q_wxyz[3])

        # Simplified fixed covariance
        P_cov = np.diag([0.1, 0.1, 0.1, 0.17, 0.17, 0.17])**2
        odom_msg.pose.covariance = P_cov.flatten().tolist()
        T_cov = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])**2
        odom_msg.twist.covariance = T_cov.flatten().tolist()
        # Twist is zero as it's not estimated
        odom_msg.twist.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
        odom_msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)

        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StereoSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Stereo SLAM Node (KeyboardInterrupt).")
    except Exception as e: # Catch other exceptions during spin
        node.get_logger().error(f"Unhandled exception during spin: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
    finally:
        # No OpenCV windows to destroy if not used
        # cv2.destroyAllWindows()
        if node and rclpy.ok(): # Check if node still exists and context is valid
             node.destroy_node()
        if rclpy.ok(): # Check context before shutting down
             rclpy.shutdown()
        print("ROS Cleanup Complete.")


if __name__ == '__main__':
    main()
