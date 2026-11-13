#!/usr/bin/env python3
"""터틀봇(OAK-D) 카메라 화면 보기.

amr_ab_controller.py 가 YOLO 박스/거리/map 좌표를 그려 발행하는 /robot3/amr/camera_view/compressed 를 표시.
(컨트롤러가 이미 받은 영상을 재사용 -> WiFi 추가 사용 없음, 컨트롤러가 켜져 있어야 함)

실행:  ros2 run amr_ab_control turtle_cam_view [--namespace /robot3]        q/ESC 종료
"""
import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

WIN = 'TurtleBot4 OAK-D (YOLO view)'


class View(Node):
    def __init__(self, namespace):
        super().__init__('turtle_cam_view', namespace=namespace)
        self.img = None
        self.stamps = []
        self.create_subscription(CompressedImage, 'amr/camera_view/compressed', self.cb, 1)

    def cb(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:  # 깨진 메시지 하나로 대기 화면으로 돌아가지 않도록
            self.img = img
            self.stamps.append(time.monotonic())


def main():
    ap = argparse.ArgumentParser(description='TurtleBot4 OAK-D YOLO view')
    ap.add_argument('--namespace', default='/robot3')
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    n = View(args.namespace)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 704, 704)
    waiting = np.zeros((300, 704, 3), np.uint8)
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.05)
            n.stamps = [t for t in n.stamps if time.monotonic() - t < 3.0]  # 스트림이 멈추면 fps 가 0 으로 떨어지게
            if n.img is None:
                frame = waiting.copy()
                cv2.putText(frame, f'Waiting for {n.get_namespace().rstrip("/")}/amr/camera_view ...', (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, '(amr_ab_controller.py must be running)', (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            else:
                frame = n.img.copy()
                fps = len(n.stamps) / 3.0
                cv2.putText(frame, f'{fps:.1f} fps', (frame.shape[1] - 110, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WIN, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
