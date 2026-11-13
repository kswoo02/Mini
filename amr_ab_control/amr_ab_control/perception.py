"""AMRController 기능 분리: OAK-D RGB/depth 수신, depth 디코딩·동기화, YOLO 감지·추적, 감지 화면 발행.

AMRController 가 이 클래스를 상속해 쓴다. self 의 상태·파라미터는 amr_ab_controller.py 의 __init__ 에서 만든다.
"""
import json
import math
import time

import cv2
import numpy as np

from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_point

from amr_ab_control.approach_core import eligible_target_class, robust_depth_m

MISSION_STATES = ('NAV_SEARCH', 'SCAN', 'SEARCH', 'APPROACH', 'HOLD')  # 감지 -> 접근이 동작하는 상태


class PerceptionMixin:
    def on_camera_info(self, msg):
        if msg.k[0] > 0 and msg.k[4] > 0:
            self.camera_info = msg

    def _decode_compressed_depth(self, msg):  # '16UC1; compressedDepth png' (헤더 + PNG) -> uint16 mm
        data = bytes(msg.data)
        if not self.logged_depth_format:
            self.get_logger().info(f'Depth transport format: "{msg.format}" ({len(data)//1024} KB)')
            self.logged_depth_format = True
        start = data.find(b'\x89PNG')
        if start < 0:
            raise ValueError(f'unsupported compressedDepth format "{msg.format}" (PNG only)')
        depth = cv2.imdecode(np.frombuffer(data[start:], np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            raise ValueError('compressedDepth PNG decode failed')
        if depth.dtype != np.uint16:  # 32FC1(inverse-depth) 미지원
            raise ValueError(f'compressedDepth decoded dtype {depth.dtype} (expected uint16 mm)')
        return depth

    @staticmethod
    def _overlap_ratio(a, b):  # 교집합 / 작은 박스 면적
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        area = min((a[2]-a[0]) * (a[3]-a[1]), (b[2]-b[0]) * (b[3]-b[1]))
        return inter / area if area > 0 else 0.0

    def _publish_view(self, img, candidates):  # 감지 화면을 PC 로컬 토픽으로 (WiFi 추가 사용 없음)
        if self.view_pub.get_subscription_count() == 0:
            return
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, f'STATE {self.state} | car(conf>={self.min_conf}) with depth: {len(candidates)}',
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        msg.data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        self.view_pub.publish(msg)

    def _depth_for(self, rgb_msg, shape):  # RGB 시각에 가장 가까운 depth(±sync_slop) -> (depth, 이유)
        t = Time.from_msg(rgb_msg.header.stamp).nanoseconds
        best = min(self.depth_buf, default=None,
                   key=lambda m: abs(Time.from_msg(m.header.stamp).nanoseconds - t))
        if best is None:
            return None, 'no depth received'
        dt = abs(Time.from_msg(best.header.stamp).nanoseconds - t) / 1e9
        if dt > self.sync_slop:
            return None, f'depth/RGB time gap {dt:.2f}s > {self.sync_slop}s'
        try:
            depth = self._decode_compressed_depth(best)
        except Exception as e:
            return None, f'depth decode error: {e}'
        if depth.shape[:2] != shape:
            return None, f'depth size {depth.shape[:2]} != rgb {shape}'
        return depth, ''

    def on_rgb(self, rgb_msg):
        now = time.monotonic()
        mission = self.state in MISSION_STATES  # 화면은 항상 갱신, 접근 동작은 미션 상태에서만
        if not self.depth_confirmed or self.camera_info is None:
            return
        if rgb_msg.header.frame_id == '':
            return
        rgb = cv2.imdecode(np.frombuffer(rgb_msg.data, np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            self.get_logger().warn('RGB decode failed', throttle_duration_sec=2.0)
            return
        depth_cache = []  # car 가 있을 때만 1회 디코딩

        def get_depth():
            if not depth_cache:
                depth_cache.append(self._depth_for(rgb_msg, rgb.shape[:2]))
            return depth_cache[0]
        info = self.camera_info
        if info.width != rgb.shape[1] or info.height != rgb.shape[0]:
            return
        try:
            if self.use_tracking:  # ByteTrack: 프레임 간 같은 물체에 같은 ID
                output = self.model.track(rgb, persist=True, conf=self.min_conf,
                                          tracker='bytetrack.yaml', verbose=False)
            else:
                output = self.model.predict(rgb, conf=self.min_conf, verbose=False)
        except Exception as e:
            self.get_logger().warn(f'YOLO detection failure: {e}')
            return
        if not output or output[0].boxes is None:
            self.candidate_count = 0
            self._publish_view(rgb, [])
            return
        results = output[0]
        view = results.plot()
        candidates = []
        dummy_boxes = [b.xyxy[0].tolist() for b in results.boxes
                       if str(self.model.names[int(b.cls[0])]).lower().strip() == 'dummy']
        for box in results.boxes:
            name = str(self.model.names[int(box.cls[0])]).lower().strip()
            if not eligible_target_class(name):
                continue  # car 만 대상, dummy 제외
            ident = int(box.id[0]) if (self.use_tracking and box.id is not None) else -1
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            u, v = (x1+x2)//2, (y1+y2)//2
            if any(self._overlap_ratio((x1, y1, x2, y2), d) >= self.dummy_overlap for d in dummy_boxes):
                cv2.putText(view, 'IGNORED: DUMMY OVERLAP', (x1, min(y2 + 22, view.shape[0] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)  # car/dummy 동시 감지 -> 접근 안 함
                if mission:
                    self.get_logger().info(f'OAK-D car conf={confidence:.2f} px=({u},{v}) ignored: DUMMY OVERLAP',
                                           throttle_duration_sec=1.0)
                continue
            depth, why = get_depth()
            z = None if depth is None else robust_depth_m(depth, (x1, y1, x2, y2), '16UC1')
            if z is None:
                why = why or 'no valid depth (0.3~5m) in bbox center'
                cv2.putText(view, 'NO DEPTH', (x1, min(y2 + 22, view.shape[0] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if mission:
                    self.get_logger().warn(f'car detected but {why}', throttle_duration_sec=2.0)
                continue
            fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
            if fx <= 0 or fy <= 0:
                continue
            pt = PointStamped()
            pt.header = rgb_msg.header
            pt.point.x, pt.point.y, pt.point.z = (u-cx)*z/fx, (v-cy)*z/fy, z  # 픽셀 + depth -> 카메라 좌표
            try:
                tf = self.tf_buffer.lookup_transform(self.map_frame,
                    rgb_msg.header.frame_id, Time.from_msg(rgb_msg.header.stamp),
                    timeout=Duration(seconds=0.0))
            except Exception:
                try:  # 이미지 시각 TF 가 없으면 최신 TF (WiFi 지연)
                    tf = self.tf_buffer.lookup_transform(self.map_frame,
                        rgb_msg.header.frame_id, Time(), timeout=Duration(seconds=0.0))
                except Exception as e:
                    self.get_logger().warn(f'TF map<-camera failed: {e}', throttle_duration_sec=2.0)
                    continue
            mapped = do_transform_point(pt, tf)
            candidates.append({'class': name, 'track_id': ident, 'confidence': confidence,
                'pixel': [u, v], 'depth_m': round(z, 3), 'x': float(mapped.point.x),
                'y': float(mapped.point.y), 'seen': now,
                'bearing': math.atan2(cx - u, fx)})  # 화면 중앙 기준 방향 (+ 왼쪽)
            if mission:
                robot = self._robot_xy()
                rd = '' if robot is None else \
                    f' robot->car {math.dist(robot, (mapped.point.x, mapped.point.y)):.2f}m'
                self.get_logger().info(
                    f'OAK-D car conf={confidence:.2f} px=({u},{v}) depth={z:.2f}m '
                    f'map=({mapped.point.x:.2f},{mapped.point.y:.2f}){rd} '
                    f'[{self.candidate_count + 1}/{self.confirm_frames}]')
            cv2.circle(view, (u, v), 6, (0, 0, 255), -1)
            cv2.putText(view, f'{z:.2f}m  map({mapped.point.x:.2f},{mapped.point.y:.2f})',
                        (x1, min(y2 + 22, view.shape[0] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        self._publish_view(view, candidates)
        if not mission:
            return
        if not candidates:
            self.candidate_count = 0
            return
        if self.locked_target_id is not None:
            selected = next((c for c in candidates if c['track_id'] == self.locked_target_id), None)  # 잠근 ID 추적
            if selected is None and self.target is not None:  # ID 끊김 -> 마지막 위치 근처 car 로 재연결
                near = [c for c in candidates
                        if math.dist((c['x'], c['y']), (self.target['x'], self.target['y'])) <= self.reid_distance_m]
                if near:
                    selected = min(near, key=lambda c: math.dist((c['x'], c['y']),
                                                                 (self.target['x'], self.target['y'])))
                    self.get_logger().info(f'Track re-linked: ID {self.locked_target_id} -> {selected["track_id"]}')
                    self.locked_target_id = selected['track_id']
                    self.candidate_id = selected['track_id']
            if selected is None:
                return
        else:
            selected = max(candidates, key=lambda c: c['confidence'])
        ident = selected['track_id']
        if ident == self.candidate_id:
            self.candidate_count += 1
        else:
            self.candidate_id = ident
            self.candidate_count = 1
        if self.candidate_count < self.confirm_frames:
            return
        self.target = selected
        self.last_bearing = selected['bearing']
        report = {k: v for k, v in selected.items() if k != 'seen'}
        self.target_pub.publish(String(data=json.dumps(report, ensure_ascii=False)))
        if self.state == 'NAV_SEARCH' and self.detect_on_way and self.cancel_pending is None:
            self.status(f'Confirmed car ID {ident}; canceling search for approach')
            self._cancel_nav('APPROACH')
        elif self.state in ('SEARCH', 'SCAN'):
            self._turn(0.0)
            self._start_approach()
        elif (self.state == 'HOLD' and self.follow_in_hold and self.hold_target_xy is not None
              and math.dist((selected['x'], selected['y']), self.hold_target_xy) > self.replan_distance_m):
            self.status(f'Tracked car ID {ident} moved '
                        f'{math.dist((selected["x"], selected["y"]), self.hold_target_xy):.2f} m; following')
            self.state = 'SEARCH'  # _start_approach 진입 조건
            self._start_approach()
