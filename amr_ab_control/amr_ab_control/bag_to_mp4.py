#!/usr/bin/env python3
"""bag 안의 카메라 영상(압축 이미지 토픽)을 mp4 로 변환.

  ros2 run amr_ab_control bag_to_mp4 ~/rokey_ws/bags/test_20261112_200000
  -> 같은 폴더에 turtle_cam.mp4, webcam.mp4 생성 (재생 속도 = 실제 시간)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

# 토픽 끝부분으로 찾는다 -> 로봇 네임스페이스(/robot3 등)가 달라도 동작
TOPICS = {'/amr/camera_view/compressed': 'turtle_cam.mp4',
          '/amr/webcam_view/compressed': 'webcam.mp4'}
FPS = 10.0


def main():
    if len(sys.argv) != 2:
        raise SystemExit('사용법: ros2 run amr_ab_control bag_to_mp4 <bag 폴더>')
    bag = Path(sys.argv[1]).expanduser()
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    frames = {}  # 실제 토픽 이름 -> [(t_ns, jpeg bytes)]
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if any(topic.endswith(s) for s in TOPICS):
            msg = deserialize_message(data, CompressedImage)
            frames.setdefault(topic, []).append((t_ns, bytes(msg.data)))
    if not frames:
        print(f'영상 토픽 없음: {", ".join(TOPICS)}')
    for topic, items in frames.items():
        decoded = (cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for _, b in items)
        first = next((d for d in decoded if d is not None), None)  # 앞쪽 깨진 프레임은 건너뜀
        if first is None:
            print(f'{topic}: 디코딩 가능한 프레임 없음')
            continue
        h, w = first.shape[:2]
        out = bag / next(name for s, name in TOPICS.items() if topic.endswith(s))
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
        t0 = items[0][0]
        idx = 0
        n_out = int((items[-1][0] - t0) / 1e9 * FPS) + 1
        img = first
        for k in range(n_out):  # 실제 시간 간격대로 프레임 반복 -> 재생 속도 = 실시간
            t = t0 + k * 1e9 / FPS
            while idx + 1 < len(items) and items[idx + 1][0] <= t:
                idx += 1
                dec = cv2.imdecode(np.frombuffer(items[idx][1], np.uint8), cv2.IMREAD_COLOR)
                if dec is not None:
                    img = cv2.resize(dec, (w, h)) if dec.shape[:2] != (h, w) else dec
            vw.write(img)
        vw.release()
        print(f'{topic}: {len(items)}장 -> {out} ({n_out / FPS:.1f}초)')


if __name__ == '__main__':
    main()
