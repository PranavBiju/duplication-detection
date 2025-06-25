import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import rclpy

import tf2_ros
import tf_transformations


#this depends on camera
fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5

model = YOLO("yolov8n.pt")
landmarks = []

rclpy.init()
node = rclpy.create_node('landmark_node')
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer, node)


def is_duplicate(new_pos, existing_landmarks, threshold=0.2):
    for lm in existing_landmarks:
        dist = np.linalg.norm(np.array(lm['position']) - np.array(new_pos))
        if dist < threshold:
            return True
    return False


def image_callback(rgb_image, depth_image):
    results = model(rgb_image)

    for det in results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        class_id = int(det.cls[0])
        conf = float(det.conf[0])
        label = model.names[class_id]

        u, v = (x1 + x2) // 2, (y1 + y2) // 2
        Z = depth_image[v, u] / 1000.0

        if Z == 0 or np.isnan(Z):
            continue

        local_pos = np.array([
            (u - cx) * Z / fx,
            (v - cy) * Z / fy,
            Z,
            1.0
        ])

        try:
            trans = tf_buffer.lookup_transform('map', 'camera_link', rclpy.time.Time())
            translation = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ]
            rotation = [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ]

            T = tf_transformations.quaternion_matrix(rotation)
            T[0:3, 3] = translation

            global_pos = np.dot(T, local_pos)[:3].tolist()

        except Exception as e:
            print(f"[!] TF transform failed: {e}")
            global_pos = local_pos[:3].tolist()  

        if not is_duplicate(global_pos, landmarks):
            landmarks.append({
                "id": len(landmarks),
                "label": label,
                "position": global_pos,
                "time": datetime.now(),
                "image": rgb_image[y1:y2, x1:x2]
            })
            print(f"[+] New object added: {label} at {global_pos}")
        else:
            print(f"[=] Duplicate object skipped: {label} at {global_pos}")



rgb = cv2.imread("test.jpg")
depth = np.full((rgb.shape[0], rgb.shape[1]), 1000, dtype=np.uint16)

print("First detection:")
image_callback(rgb, depth)

print("\nSecond detection (should be duplicate):")
image_callback(rgb, depth)

rclpy.shutdown()