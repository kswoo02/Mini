import cv2
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, Image

from ultralytics import YOLO


class YOLODetector(Node):

    def __init__(self):

        super().__init__('yolo_detector')

        # =================================
        # 1. YOLO 모델 로드
        # =================================

        MODEL_PATH = "/home/mu-03/Downloads/best.pt"

        self.model = YOLO(MODEL_PATH)

        self.get_logger().info(f"YOLO Model Loaded: {MODEL_PATH}")
        self.get_logger().info(f"Classes: {self.model.names}")

        # =================================
        # 2. 이미지 데이터 초기화
        # =================================

        self.frame = None
        self.depth_image = None

        # =================================
        # 3. RGB Camera Subscriber
        # =================================

        self.rgb_subscription = self.create_subscription(
            CompressedImage,
            '/robot3/oakd/rgb/image_raw/compressed',
            self.rgb_callback,
            qos_profile_sensor_data
        )

        # =================================
        # 4. Depth Camera Subscriber
        # =================================

        self.depth_subscription = self.create_subscription(
            Image,
            '/robot3/oakd/stereo/image_raw',
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info("YOLO + Depth Detector Started")
        self.get_logger().info("Waiting for RGB and Depth images...")

    # =================================
    # RGB Image Callback
    # =================================

    def rgb_callback(self, msg):

        try:

            np_arr = np.frombuffer(msg.data, np.uint8)

            self.frame = cv2.imdecode(
                np_arr,
                cv2.IMREAD_COLOR
            )

            if self.frame is None:
                self.get_logger().warning("RGB image decode failed")

        except Exception as e:

            self.get_logger().error(f"RGB Error: {e}")

    # =================================
    # Depth Image Callback
    # =================================

    def depth_callback(self, msg):

        try:

            # Depth image conversion
            depth = np.frombuffer(
                msg.data,
                dtype=np.uint8
            )

            depth_image = cv2.imdecode(
                depth,
                cv2.IMREAD_UNCHANGED
            ) if msg.encoding in ['jpeg', 'jpg'] else None

            # 일반 ROS Image 메시지 변환
            if depth_image is None:

                if msg.encoding == '16UC1':
                    depth_image = np.frombuffer(
                        msg.data,
                        dtype=np.uint16
                    ).reshape(msg.height, msg.width)

                elif msg.encoding == '32FC1':
                    depth_image = np.frombuffer(
                        msg.data,
                        dtype=np.float32
                    ).reshape(msg.height, msg.width)

                else:
                    self.get_logger().error(
                        f"Unsupported depth encoding: {msg.encoding}"
                    )
                    return

            # mm -> m
            if depth_image.dtype == np.uint16:

                self.depth_image = depth_image.astype(
                    np.float32
                ) / 1000.0

            else:

                # 32FC1은 일반적으로 meter 단위
                self.depth_image = depth_image.copy()

        except Exception as e:

            self.get_logger().error(f"Depth Error: {e}")

    # =================================
    # Bounding Box Center Distance
    # =================================

    def get_object_distance(self, center_x, center_y):

        if self.depth_image is None:
            return None

        depth = self.depth_image

        depth_h, depth_w = depth.shape

        rgb_h, rgb_w = self.frame.shape[:2]

        # RGB 좌표를 Depth 이미지 좌표로 변환
        depth_x = int(center_x * depth_w / rgb_w)
        depth_y = int(center_y * depth_h / rgb_h)

        depth_x = min(max(depth_x, 0), depth_w - 1)
        depth_y = min(max(depth_y, 0), depth_h - 1)

        # 중심점 주변 5x5 영역
        x1 = max(0, depth_x - 2)
        x2 = min(depth_w, depth_x + 3)

        y1 = max(0, depth_y - 2)
        y2 = min(depth_h, depth_y + 3)

        depth_region = depth[y1:y2, x1:x2]

        # 0 또는 무효값 제거
        valid_depth = depth_region[
            np.isfinite(depth_region) &
            (depth_region > 0) &
            (depth_region <= 10.0)
        ]

        if len(valid_depth) == 0:
            return None

        # 주변 5x5 픽셀의 중앙값 사용
        distance = np.median(valid_depth)

        return float(distance)

    # =================================
    # YOLO Detection
    # =================================

    def detect(self):

        if self.frame is None or self.depth_image is None:
            return

        # YOLO detection
        results = self.model.predict(
            self.frame,
            imgsz=640,
            conf=0.25,
            device='cpu',
            verbose=False
        )

        annotated_frame = self.frame.copy()

        # Detection 결과 반복
        for box in results[0].boxes:

            # Bounding Box 좌표
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # 객체 중심 좌표
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Class 및 Confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = self.model.names[class_id]

            # 중심점 거리 계산
            distance = self.get_object_distance(
                center_x,
                center_y
            )

            # 거리 표시
            if distance is not None:

                label = f"{class_name} {confidence:.2f} {distance:.2f}m"

                self.get_logger().info(
                    f"{class_name}: Distance = {distance:.2f} m"
                )

            else:

                label = f"{class_name} {confidence:.2f} No Depth"

            # Bounding Box 표시
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # 중심점 표시
            cv2.circle(
                annotated_frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Label 표시
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Display
        cv2.imshow(
            "TurtleBot4 YOLO + Depth Detection",
            annotated_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            raise KeyboardInterrupt


# =================================
# Main
# =================================

def main():

    rclpy.init()

    node = YOLODetector()

    try:

        while rclpy.ok():

            rclpy.spin_once(
                node,
                timeout_sec=0.01
            )

            node.detect()

    except KeyboardInterrupt:

        print("YOLO + Depth Detection Stopped")

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


# import cv2
# import rclpy
# import numpy as np

# from rclpy.node import Node
# from rclpy.qos import qos_profile_sensor_data

# from sensor_msgs.msg import CompressedImage, Image
# from cv_bridge import CvBridge

# from ultralytics import YOLO


# class YOLODetector(Node):

#     def __init__(self):

#         super().__init__('yolo_detector')

#         # YOLO model
#         MODEL_PATH = "/home/mu-03/Downloads/best.pt"
#         self.model = YOLO(MODEL_PATH)

#         self.get_logger().info(f"YOLO Model Loaded: {MODEL_PATH}")
#         self.get_logger().info(f"Classes: {self.model.names}")

#         # OpenCV bridge
#         self.bridge = CvBridge()

#         self.frame = None

#         # Camera topic
#         self.subscription = self.create_subscription(
#             CompressedImage,
#             '/robot3/oakd/rgb/image_raw/compressed',
#             self.image_callback,
#             qos_profile_sensor_data
#         )

#         self.get_logger().info("YOLO Detector Started")

#     def image_callback(self, msg):

#         try:
#             np_arr = np.frombuffer(msg.data, np.uint8)

#             self.frame = cv2.imdecode(
#                 np_arr,
#                 cv2.IMREAD_COLOR
#             )

#             if self.frame is None:
#                 self.get_logger().warning("Image decode failed")

#         except Exception as e:
#             self.get_logger().error(f"Image conversion error: {e}")

#     def detect(self):

#         if self.frame is None:
#             return

#         # YOLO detection
#         results = self.model.predict(
#             self.frame,
#             imgsz=640,
#             conf=0.25,
#             device='cpu',
#             verbose=False
#         )

#         # Draw bounding boxes
#         annotated_frame = results[0].plot()

#         # Display
#         cv2.imshow("TurtleBot4 YOLO Detection", annotated_frame)

#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('q'):
#             raise KeyboardInterrupt


# def main():

#     rclpy.init()

#     node = YOLODetector()

#     try:

#         while rclpy.ok():

#             rclpy.spin_once(node, timeout_sec=0.01)

#             node.detect()

#     except KeyboardInterrupt:
#         print("YOLO Detection Stopped")

#     finally:

#         node.destroy_node()

#         rclpy.shutdown()

#         cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()
