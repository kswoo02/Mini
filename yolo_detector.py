import cv2
import time
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, Image

from ultralytics import YOLO


# =================================
# 설정
# =================================

MODEL_PATH = "/home/mu-03/Downloads/best.pt"

RGB_TOPIC = '/robot3/oakd/rgb/image_raw/compressed'
DEPTH_TOPIC = '/robot3/oakd/stereo/image_raw'

DEVICE = None          # None=자동(GPU 있으면 GPU), '0'=GPU, 'cpu'=강제 CPU
IMGSZ = 640            # best.pt 학습 크기와 동일
CONF = 0.25

# 박스 중앙 몇 %에서 거리를 샘플링할지.
# 중심 몇 픽셀만 보면 차체 틈으로 배경이 잡힌다.
CENTER_FRACTION = 0.4

VALID_RANGE_M = (0.15, 10.0)   # 이 범위 밖은 거리로 인정하지 않는다


class YOLODetector(Node):

    def __init__(self):

        super().__init__('yolo_detector')

        # =================================
        # 1. YOLO 모델 로드
        # =================================

        self.model = YOLO(MODEL_PATH)

        self.get_logger().info(f"YOLO Model Loaded: {MODEL_PATH}")
        self.get_logger().info(f"Classes: {self.model.names}")

        # =================================
        # 2. 이미지 데이터 초기화
        # =================================

        self.frame = None
        self.depth_image = None

        self.frame_is_new = False      # 같은 프레임을 다시 추론하지 않기 위해
        self.last_rgb_t = 0.0          # 마지막 RGB 수신 시각
        self.last_depth_t = 0.0        # 마지막 depth 수신 시각

        self.should_exit = False
        self.n_infer = 0
        self.fps = 0.0
        self._t_fps = time.time()

        # =================================
        # 3. RGB Camera Subscriber
        # =================================

        self.rgb_subscription = self.create_subscription(
            CompressedImage,
            RGB_TOPIC,
            self.rgb_callback,
            qos_profile_sensor_data
        )

        # =================================
        # 4. Depth Camera Subscriber
        # =================================

        self.depth_subscription = self.create_subscription(
            Image,
            DEPTH_TOPIC,
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

            frame = cv2.imdecode(
                np_arr,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                self.get_logger().warning("RGB image decode failed")
                return

            self.frame = frame
            self.frame_is_new = True
            self.last_rgb_t = time.time()

        except Exception as e:

            self.get_logger().error(f"RGB Error: {e}")

    # =================================
    # Depth Image Callback
    # =================================

    def depth_callback(self, msg):

        try:

            # ROS 규약: 16UC1 = mm, 32FC1 = m
            enc = msg.encoding.lower()

            if enc in ('16uc1', 'mono16'):
                dtype, to_meter = np.uint16, 0.001

            elif enc == '32fc1':
                dtype, to_meter = np.float32, 1.0

            else:
                self.get_logger().error(
                    f"Unsupported depth encoding: {msg.encoding}",
                    throttle_duration_sec=5.0
                )
                return

            # step 에 행 패딩이 있을 수 있으므로 width 가 아니라 step 기준으로
            # 펼친 뒤 width 만큼 잘라낸다
            itemsize = np.dtype(dtype).itemsize

            depth_image = np.frombuffer(msg.data, dtype=dtype)
            depth_image = depth_image.reshape(
                msg.height,
                msg.step // itemsize
            )[:, :msg.width]

            # 미터 단위 float32 로 통일 (astype 이 사본을 만들어
            # frombuffer 의 읽기전용 문제도 함께 해결된다)
            self.depth_image = depth_image.astype(np.float32) * to_meter
            self.last_depth_t = time.time()

        except Exception as e:

            self.get_logger().error(f"Depth Error: {e}")

    # =================================
    # Bounding Box Distance
    # =================================

    def get_object_distance(self, x1, y1, x2, y2):
        """박스 중앙부에서 0 을 뺀 중앙값 거리(m). 없으면 None.

        중심 몇 픽셀이 아니라 박스 중앙 CENTER_FRACTION 영역을 본다.
        중심 한 점은 차체 틈·유리·구멍으로 배경이 잡히기 쉽다.
        """

        if self.depth_image is None or self.frame is None:
            return None

        depth = self.depth_image

        depth_h, depth_w = depth.shape

        rgb_h, rgb_w = self.frame.shape[:2]

        # RGB 좌표를 Depth 이미지 좌표로 변환
        sx = depth_w / rgb_w
        sy = depth_h / rgb_h

        cx = (x1 + x2) / 2.0 * sx
        cy = (y1 + y2) / 2.0 * sy

        # 박스 중앙부 영역
        bw = (x2 - x1) * sx * CENTER_FRACTION
        bh = (y2 - y1) * sy * CENTER_FRACTION

        rx1 = max(0, int(cx - bw / 2))
        rx2 = min(depth_w, int(cx + bw / 2) + 1)

        ry1 = max(0, int(cy - bh / 2))
        ry2 = min(depth_h, int(cy + bh / 2) + 1)

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        depth_region = depth[ry1:ry2, rx1:rx2]

        # 0 또는 무효값 제거
        lo, hi = VALID_RANGE_M

        valid_depth = depth_region[
            np.isfinite(depth_region) &
            (depth_region >= lo) &
            (depth_region <= hi)
        ]

        if valid_depth.size == 0:
            return None

        # 평균이 아니라 중앙값 — 평균은 물체와 배경을 섞어
        # 존재하지 않는 중간 거리를 만든다
        return float(np.median(valid_depth))

    # =================================
    # YOLO Detection
    # =================================

    def detect(self):

        # RGB 가 없으면 아무것도 못 한다.
        # depth 가 없어도 검출만은 보여준다 (원본은 둘 다 와야 화면이 떴다)
        if self.frame is None:
            return

        frame = self.frame
        self.frame_is_new = False       # 같은 프레임 재추론 방지

        # YOLO detection
        results = self.model.predict(
            frame,
            imgsz=IMGSZ,
            conf=CONF,
            device=DEVICE,
            verbose=False
        )

        annotated_frame = frame.copy()

        have_depth = self.depth_image is not None

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

            # 박스 중앙부 거리 계산
            distance = self.get_object_distance(x1, y1, x2, y2) \
                if have_depth else None

            # 거리 표시
            if distance is not None:

                label = f"{class_name} {confidence:.2f} {distance:.2f}m"
                color = (0, 255, 0)

                self.get_logger().info(
                    f"{class_name}: Distance = {distance:.2f} m",
                    throttle_duration_sec=1.0
                )

            else:

                label = f"{class_name} {confidence:.2f} No Depth"
                color = (0, 165, 255)

            # Bounding Box 표시
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
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
                color,
                2
            )

        self.draw_hud(annotated_frame, len(results[0].boxes))

        self.show(annotated_frame)

    # =================================
    # 상태 표시 — 멈췄는지 눈으로 보려고
    # =================================

    def draw_hud(self, img, n_det):

        self.n_infer += 1
        dt = time.time() - self._t_fps

        if dt >= 1.0:
            self.fps = self.n_infer / dt
            self.n_infer = 0
            self._t_fps = time.time()

        now = time.time()
        rgb_age = now - self.last_rgb_t if self.last_rgb_t else 99.9
        dep_age = now - self.last_depth_t if self.last_depth_t else 99.9

        hud = (f"{self.fps:4.1f} fps  det {n_det}  "
               f"rgb {rgb_age:4.1f}s  depth "
               + (f"{dep_age:4.1f}s" if self.last_depth_t else "--"))

        # 데이터가 끊기면 빨갛게
        col = (255, 255, 255) if max(rgb_age, dep_age) < 1.0 else (0, 0, 255)

        cv2.putText(img, hud, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    # =================================
    # 화면 출력 — 프레임이 안 와도 창은 살아 있어야 한다
    # =================================

    def show(self, img=None):

        if img is not None:
            self._last_vis = img

        vis = getattr(self, '_last_vis', None)

        if vis is None:
            return

        cv2.imshow("TurtleBot4 YOLO + Depth Detection", vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.should_exit = True


# =================================
# Main
# =================================

def main():

    rclpy.init()

    node = YOLODetector()

    try:

        while rclpy.ok() and not node.should_exit:

            # spin_once 는 준비된 작업을 '한 개'만 처리한다.
            # 구독이 둘(RGB+depth)이라 한 번만 부르면 큐가 밀린다.
            # 새 데이터를 잠깐 기다린 뒤, 쌓인 것을 마저 비운다.
            rclpy.spin_once(node, timeout_sec=0.05)

            for _ in range(20):
                rclpy.spin_once(node, timeout_sec=0.0)

            # 새 프레임이 있을 때만 추론한다
            if node.frame_is_new:
                node.detect()
            else:
                node.show()        # 창은 계속 갱신 (먹통 방지)

    except KeyboardInterrupt:

        print("YOLO + Depth Detection Stopped")

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
