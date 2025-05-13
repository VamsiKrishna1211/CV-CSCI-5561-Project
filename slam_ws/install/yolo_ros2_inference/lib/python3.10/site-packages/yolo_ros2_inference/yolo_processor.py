#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge # For ROS2 image conversion
import cv2 # OpenCV for potential color conversion if needed, though YOLO plot() output is usually BGR
import time
import numpy as np # YOLO plot() returns a NumPy array

from ultralytics import YOLO

# Ensure your model paths are correct.
# These names "yolo11m.pt" seem non-standard for recent Ultralytics models.
# Replace with your actual model filenames or paths.
DETECTION_MODEL_PATH = "yolov8m.pt"  # Example: "yolov8m.pt" or path/to/your/detection_model.pt
SEGMENTATION_MODEL_PATH = "/root/project/slam_ws/src/yolo_ros2_inference/yolo_ros2_inference/yolo11s-seg.pt" # Example: "yolov8m-seg.pt" or path/to/your/segmentation_model.pt

INPUT_IMAGE_TOPIC = "/front_stereo_camera/left/image_rect_color" # From your topic list
DETECTION_OUTPUT_TOPIC = "/ultralytics/detection/image"
SEGMENTATION_OUTPUT_TOPIC = "/ultralytics/segmentation/image"

class UltralyticsNode(Node):
    def __init__(self):
        super().__init__("ultralytics_ros2_node")
        self.get_logger().info(f"Ultralytics ROS2 Node starting...")
        self.get_logger().info(f"Attempting to load detection model from: {DETECTION_MODEL_PATH}")
        self.get_logger().info(f"Attempting to load segmentation model from: {SEGMENTATION_MODEL_PATH}")

        try:
            self.detection_model = YOLO(DETECTION_MODEL_PATH)
            self.segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)
            self.get_logger().info("YOLO models loaded successfully.")
        except Exception as e:
            self.get_logger().fatal(f"Failed to load YOLO models: {e}. Please check model paths and integrity.")
            # Consider raising an error or shutting down the node if models are critical
            # For now, we'll let it proceed, but inference will fail.
            # rclpy.shutdown() # Option to shutdown
            return


        self.bridge = CvBridge()
        # QoS profile: keep last 10, best effort for image streams
        qos_profile = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)


        # Publishers
        self.det_image_pub = self.create_publisher(Image, DETECTION_OUTPUT_TOPIC, qos_profile)
        self.seg_image_pub = self.create_publisher(Image, SEGMENTATION_OUTPUT_TOPIC, qos_profile)
        self.get_logger().info(f"Publishing detection results to: {DETECTION_OUTPUT_TOPIC}")
        self.get_logger().info(f"Publishing segmentation results to: {SEGMENTATION_OUTPUT_TOPIC}")

        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            INPUT_IMAGE_TOPIC,
            self.image_callback,
            qos_profile # QoS profile
        )
        self.get_logger().info(f"Subscribed to image topic: {INPUT_IMAGE_TOPIC}")

        # Brief pause similar to the original script, though not strictly necessary
        # time.sleep(1) # Can be removed if not needed


    def image_callback(self, msg: Image):
        """Callback function to process image and publish annotated images."""
        # self.get_logger().debug("Image received...") # Uncomment for verbose logging

        if not hasattr(self, 'detection_model') or not hasattr(self, 'segmentation_model'):
             self.get_logger().warn("Models not loaded, skipping inference.")
             return

        try:
            # Convert ROS Image message to OpenCV image (BGR format is common for CvBridge from color images)
            # The YOLO model expects RGB by default but can handle BGR.
            # The .plot() method returns a BGR NumPy array.
            cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # --- Detection ---
        if self.det_image_pub.get_subscription_count() > 0:
            try:
                # Inference expects image in HWC format. cv_image_bgr is already in this format.
                # Models can typically handle BGR or RGB input, check your specific model.
                # For Ultralytics, you can pass BGR directly.
                det_result = self.detection_model(cv_image_bgr, verbose=False) # verbose=False to reduce console output
                # .plot() returns a BGR NumPy array with annotations
                det_annotated_bgr = det_result[0].plot(show=False)

                # Convert annotated BGR image back to ROS Image message
                # Ensure the encoding matches the annotated image format ("bgr8")
                det_img_msg = self.bridge.cv2_to_imgmsg(det_annotated_bgr, encoding="bgr8")
                det_img_msg.header = msg.header # Preserve original timestamp and frame_id
                self.det_image_pub.publish(det_img_msg)
            except Exception as e:
                self.get_logger().error(f"Error during detection processing: {e}")

        # --- Segmentation ---
        if self.seg_image_pub.get_subscription_count() > 0:
            try:
                seg_result = self.segmentation_model(cv_image_bgr, verbose=False)
                # .plot() returns a BGR NumPy array with annotations
                seg_annotated_bgr = seg_result[0].plot(show=False)

                # Convert annotated BGR image back to ROS Image message
                seg_img_msg = self.bridge.cv2_to_imgmsg(seg_annotated_bgr, encoding="bgr8")
                seg_img_msg.header = msg.header # Preserve original timestamp and frame_id
                self.seg_image_pub.publish(seg_img_msg)
            except Exception as e:
                self.get_logger().error(f"Error during segmentation processing: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = None # Initialize node to None for robust error handling in finally block
    try:
        node = UltralyticsNode()
        if hasattr(node, 'detection_model') and hasattr(node, 'segmentation_model'): # Only spin if models loaded
             rclpy.spin(node)
        else:
             node.get_logger().fatal("Node initialization failed (models not loaded). Shutting down.")
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("Node shutting down (KeyboardInterrupt).")
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Unhandled exception in main: {e}")
        else:
            print(f"[FATAL] Unhandled exception during node creation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok(): # Check if RCLPY context is still valid
            rclpy.shutdown()
        print("ROS2 cleanup complete.")


if __name__ == '__main__':
    main()
