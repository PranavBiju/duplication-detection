import os
os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Camera intrinsics (adjust if needed)
fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5

# Landmark storage
landmarks = []

def is_duplicate(new_pos, existing_landmarks, threshold=0.2):
    for lm in existing_landmarks:
        dist = np.linalg.norm(np.array(lm['position']) - np.array(new_pos))
        if dist < threshold:
            return True
    return False

class LandmarkDetector(Node):
    def __init__(self):
        super().__init__('landmark_detector')
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        rgb_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')

        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.callback)

    def get_rover_position(self):
        try:
            t: TransformStamped = self.tf_buffer.lookup_transform(
                'odom', 'base_link', rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            z = t.transform.translation.z
            return np.array([x, y, z])
        except Exception as e:
            self.get_logger().warn(f"[TF] Could not get transform: {e}")
            return None

    def callback(self, rgb_msg, depth_msg):
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        results = model(rgb_image)

        rover_pos = self.get_rover_position()
        if rover_pos is None:
            return

        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            class_id = int(det.cls[0])
            label = model.names[class_id]

            # Center of bounding box
            u, v = (x1 + x2) // 2, (y1 + y2) // 2

            Z = depth_image[v, u] / 1000.0  # mm → m
            if Z == 0 or np.isnan(Z):
                continue

            # Local 3D coordinates from camera
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            local_pos = np.array([X, Y, Z])

            # Estimate global position by adding to base_link position (ignores rotation)
            global_pos = rover_pos + local_pos

            if not is_duplicate(global_pos, landmarks):
                landmarks.append({
                    "id": len(landmarks),
                    "label": label,
                    "position": global_pos.tolist(),
                    "time": datetime.now(),
                    "image": rgb_image[y1:y2, x1:x2]
                })
                print(f"[+] New object added: {label} at {global_pos}")

def main(args=None):
    rclpy.init(args=args)
    node = LandmarkDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()