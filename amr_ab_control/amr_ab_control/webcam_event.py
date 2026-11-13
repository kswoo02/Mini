#!/usr/bin/env python3
"""외장 웹캠(CCTV 역할)에서 car 를 감지하면 로봇에 START 이벤트를 보낸다.

  웹캠 car 연속 감지 -> /robot3/amr/event  "START"  -> amr_ab_controller.py 가 Undock -> B -> C -> 접근

실행:
  ros2 run amr_ab_control webcam_event                 # 외장 USB 카메라 /dev/video2 (기본값)
  ros2 run amr_ab_control webcam_event --camera 2 --frames 5 --conf 0.5

키:  q/ESC 종료,  s 수동 START,  x STOP,  r RESET(FAULT 해제)
화면 상단에 로봇 상태(/robot3/amr/status)가 같이 표시된다.
"""
import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from ultralytics import YOLO

from amr_ab_control.paths import WEBCAM_MODEL

DEFAULT_MODEL = WEBCAM_MODEL  # 패키지 models/webcam_car_dummy_aug.pt (car_dummy_aug best)


class EventBridge(Node):
    def __init__(self, namespace):
        super().__init__('webcam_event', namespace=namespace)
        self.pub = self.create_publisher(String, 'amr/event', 10)
        # 감지 화면 (bag 녹화용, 이 PC 로컬 토픽)
        self.view_pub = self.create_publisher(CompressedImage, 'amr/webcam_view/compressed', 1)
        self.last_view = 0.0
        self.robot_state = '?'
        self.robot_msg = ''
        self.create_subscription(String, 'amr/status', self.on_status, 10)

    def on_status(self, msg):
        try:
            d = json.loads(msg.data)
            self.robot_state = d.get('state', '?')
            if d.get('message'):  # 1초 heartbeat(빈 메시지)는 마지막 메시지를 지우지 않음
                self.robot_msg = d['message']
        except Exception:
            self.robot_state, self.robot_msg = '?', msg.data

    def send(self, event):
        if self.pub.get_subscription_count() == 0:
            self.get_logger().warn(f'{event}: amr_ab_controller 가 안 보입니다 (구독자 0) - 그래도 발행합니다')
        self.pub.publish(String(data=event))
        self.get_logger().info(f'>>> 이벤트 발행: {event}')


def main():
    ap = argparse.ArgumentParser(description='Webcam car detection -> AMR START event')
    ap.add_argument('--camera', type=int, default=2,
                    help='외장 USB 웹캠 번호 (/dev/videoN). 노트북 내장 카메라가 0 이라 외장은 2 (v4l2-ctl --list-devices 로 확인)')
    ap.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    ap.add_argument('--conf', type=float, default=0.7)  # 웹캠 car 0.91 -> 0.7 로 상향
    ap.add_argument('--frames', type=int, default=5, help='연속 감지 프레임 수 (오탐 방지)')
    ap.add_argument('--cooldown', type=float, default=30.0, help='START 재발행 최소 간격(초)')
    ap.add_argument('--namespace', default='/robot3')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    args = ap.parse_args()

    if not args.model.is_file():
        raise SystemExit(f'[ERROR] 모델 없음: {args.model}')
    model = YOLO(str(args.model))
    print(f'[INFO] 모델 {args.model.name} 클래스 {model.names}')

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f'[ERROR] /dev/video{args.camera} 열기 실패')
    # 이 외장 카메라(Jieli USB Camera)는 YUYV 로는 프레임이 안 나오고 MJPG 에서만 동작
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    rclpy.init()
    node = EventBridge(args.namespace)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    streak = 0
    last_sent = -1e9
    sent_flash_until = 0.0
    try:
        while rclpy.ok():
            ok, frame = cap.read()
            if not ok:
                print('[WARN] 웹캠 프레임 읽기 실패')
                time.sleep(0.1)
                continue
            res = model.predict(frame, conf=args.conf, verbose=False)[0]
            img = res.plot()
            cars = [b for b in res.boxes if res.names[int(b.cls[0])].lower() == 'car']
            streak = streak + 1 if cars else 0

            now = time.monotonic()
            ready = now - last_sent >= args.cooldown
            if streak >= args.frames and ready and node.robot_state in ('IDLE', '?'):
                node.send('START')
                last_sent = now
                sent_flash_until = now + 2.0
                streak = 0

            # 오버레이
            h = 58
            cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
            cd = max(0.0, args.cooldown - (now - last_sent))
            line1 = f'car streak {streak}/{args.frames}   cooldown {cd:.0f}s   [s]START [x]STOP [r]RESET [q]quit'
            line2 = f'ROBOT: {node.robot_state}  {node.robot_msg[:60]}'
            cv2.putText(img, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, line2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if now < sent_flash_until:
                cv2.putText(img, 'START SENT!', (img.shape[1] // 2 - 110, img.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('Webcam -> AMR event', img)
            if node.view_pub.get_subscription_count() > 0 and now - node.last_view >= 0.2:  # 5Hz
                node.last_view = now
                vm = CompressedImage()
                vm.header.stamp = node.get_clock().now().to_msg()
                vm.format = 'jpeg'
                vm.data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                node.view_pub.publish(vm)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                node.send('START'); last_sent = now; sent_flash_until = now + 2.0
            elif key == ord('x'):
                node.send('STOP')
            elif key == ord('r'):
                node.send('RESET')
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
