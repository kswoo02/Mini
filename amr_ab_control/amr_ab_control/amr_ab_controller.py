#!/usr/bin/env python3
"""TurtleBot4 /robot3: START -> Undock -> A 로 이동 -> 제자리 회전 탐색 -> YOLO 감지 -> 접근 -> 화면 중앙 유지 + follow.
A/B 순찰 버전은 amr_ab_controller_patrol.py (태그 patrol-v1). --execute 없으면 dry-run.

기능별 파일 (AMRController 가 아래 Mixin 들을 상속):
  amr_ab_controller.py  파라미터·토픽 설정, 이벤트(START/STOP/RESET), 상태 머신(tick), main
  docking.py            도크 상태, Undock, 초기 위치·AMCL 확인
  navigation.py         Nav2 이동·결과·취소, 탐색 지점 이동, 제자리 회전, 접근 시작
  perception.py         RGB/depth 수신·동기화, YOLO 감지·추적, 감지 화면 발행
"""
import argparse
import json
import math
import time
from collections import deque

from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, TwistStamped
from irobot_create_msgs.msg import DockStatus
from irobot_create_msgs.action import Undock
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from amr_ab_control.paths import MAP_YAML, OAKD_MODEL
from amr_ab_control.approach_core import A, MapGuard, find_approach_goal
from amr_ab_control.docking import DockingMixin
from amr_ab_control.navigation import NavigationMixin
from amr_ab_control.perception import PerceptionMixin


class AMRController(DockingMixin, NavigationMixin, PerceptionMixin, Node):
    def __init__(self, execute=False):
        super().__init__('amr_ab_controller')
        self.execute = execute

        def param(name, value):
            self.declare_parameter(name, value)
            return self.get_parameter(name).value

        self.model_path = str(param('model_path', str(OAKD_MODEL)))  # 기본: models/oakd_best1.pt
        self.map_yaml = str(param('map_yaml', str(MAP_YAML)))
        self.init_x = float(param('initial_x', float('nan')))  # 미지정 시 RViz 2D Pose Estimate 위치 사용
        self.init_y = float(param('initial_y', float('nan')))
        self.init_yaw_deg = float(param('initial_yaw_deg', float('nan')))
        self.depth_confirmed = bool(param('confirm_aligned_depth', True))  # robot3: depth 가 RGB 정렬(704x704) 확인됨
        self.post_undock_localize_s = float(param('post_undock_localize_s', 20.0))  # 언독 후 AMCL 위치 대기 최대 시간
        self.rgb_topic = str(param('rgb_topic', 'oakd/rgb/image_raw/compressed'))  # raw 는 WiFi 로 0.4fps 라 압축 사용
        self.depth_topic = str(param('depth_topic', 'oakd/stereo/image_raw/compressedDepth'))
        self.info_topic = str(param('camera_info_topic', 'oakd/rgb/camera_info'))
        self.confirm_frames = int(param('confirm_frames', 2))  # N 프레임 연속 car 여야 인정 (검은 SUV 1회 오인식 대책)
        self.dummy_overlap = float(param('dummy_overlap', 0.3))  # car 박스가 dummy 와 이 비율 이상 겹치면 무시
        self.restrict_zone = bool(param('restrict_to_corridor', False))  # True: A-B 통로 안 car 만 접근
        self.sync_slop = float(param('sync_slop', 0.2))  # RGB-depth 시간차 허용 (근사 시간 동기화)
        self.map_frame = str(param('map_frame', 'map'))
        self.base_frame = str(param('base_frame', 'base_link'))
        self.standoff_m = float(param('standoff_m', 0.80))  # 차 중심 -> 접근 목표 거리 (실제 정지는 Nav2 오차로 ~+0.25 m)
        self.search_xy = (float(param('search_x', A[0])), float(param('search_y', A[1])))  # 회전 탐색 지점 = A 위쪽 방 (RViz Publish Point 로 변경)
        self.detect_on_way = bool(param('detect_on_way', False))  # False: A 까지 가는 동안 본 차는 무시
        self.scan_speed = float(param('scan_speed', 0.5))  # 회전 탐색 속도 rad/s (한 바퀴 ~13s)
        self.track_gain = float(param('track_gain', 1.5))  # 정지 중 차를 화면 중앙에 두는 회전 비례 게인
        self.track_max_turn = float(param('track_max_turn', 0.8))  # 중앙 유지 최대 회전 rad/s
        self.track_deadband_deg = float(param('track_deadband_deg', 6.0))  # 중앙에서 이 각도 안이면 회전 안 함
        self.lost_turn_after_s = float(param('lost_turn_after_s', 1.0))  # 이 시간 안 보이면 나간 쪽으로 회전 (0.5 는 YOLO 순간 누락에 흔들림)
        self.min_conf = float(param('min_confidence', 0.7))  # 진짜 car 0.88~0.96, 오탐 0.51 -> 0.7
        self.target_timeout_s = float(param('target_timeout_s', 1.5))  # 접근 중 이 시간 안 보이면 놓침
        self.replan_distance_m = float(param('replan_distance_m', 0.45))  # car 가 이만큼 움직이면 다시 접근 (follow)
        self.use_tracking = bool(param('use_tracking', True))  # ByteTrack ID 로 같은 car 추적
        self.reid_distance_m = float(param('reid_distance_m', 0.5))  # ID 가 바뀌어도 이 거리 안이면 같은 car
        self.follow_in_hold = bool(param('follow_in_hold', True))  # 정지 후 car 가 움직이면 따라감
        self.guard = MapGuard(self.map_yaml)
        self.model = YOLO(self.model_path)

        self.state = 'IDLE'
        self.docked = None
        self.camera_info = None
        self.use_current_pose = False
        self.initialized_at = None
        self.undock_sent_at = 0.0
        self.last_pose_pub = 0.0
        self.last_amcl_time = 0.0
        self.last_amcl_xy = None
        self.undock_deadline = 0.0
        self.localize_deadline = 0.0
        self.candidate_id = None
        self.candidate_count = 0
        self.target = None
        self.locked_target_id = None
        self.hold_target_xy = None
        self.nav_handle = None
        self.undock_handle = None
        self.undock_pending = False
        self.nav_goal_pending = False
        self.nav_seq = 0
        self.nav_retries = 0
        self.nav_goal_xy = None
        self.nav_sent_time = 0.0
        self.cancel_pending = None
        self.search_dwell_until = 0.0
        self.scan_dir = 1.0
        self.scan_until = 0.0
        self.last_bearing = 0.0
        self.turning = False
        self.logged_depth_format = False
        self.last_heartbeat = 0.0

        self.undock_client = ActionClient(self, Undock, 'undock')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.initial_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.target_pub = self.create_publisher(String, 'amr/target_info', 10)
        self.status_pub = self.create_publisher(String, 'amr/status', 10)
        self.view_pub = self.create_publisher(CompressedImage, 'amr/camera_view/compressed', 1)
        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)  # 제자리 회전 (탐색/중앙 유지)
        self.create_subscription(String, 'amr/event', self.on_event, 10)
        self.create_subscription(DockStatus, 'dock_status', self.on_dock, qos_profile_sensor_data)
        self.create_subscription(PointStamped, 'clicked_point', self.on_clicked_point, 10)  # RViz Publish Point
        amcl_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)  # 늦게 켜도 마지막 AMCL 위치 수신
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.on_amcl, amcl_qos)
        self.create_subscription(CameraInfo, self.info_topic, self.on_camera_info, qos_profile_sensor_data)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.depth_buf = deque(maxlen=20)  # depth 는 쌓아두고 car 감지 시에만 RGB 시각에 맞는 것 디코딩 (2.3 -> 9.9fps)
        self.create_subscription(CompressedImage, self.depth_topic, lambda m: self.depth_buf.append(m),
                                 qos_profile_sensor_data)  # rclpy 는 builtin(append) 콜백 불가 -> lambda
        self.create_subscription(CompressedImage, self.rgb_topic, self.on_rgb, qos_profile_sensor_data)
        self.timer = self.create_timer(0.2, self.tick)
        self.get_logger().info(
            'A-B controller ready. Mode=%s. START event required. '
            'Depth alignment confirmed=%s' % ('EXECUTE' if execute else 'DRY-RUN', self.depth_confirmed))
        if not self.depth_confirmed:
            self.get_logger().warn('Approach disabled: confirm RGB-registered depth before enabling.')
        if not self.execute:
            self.get_logger().warn('DRY-RUN: no Undock or NavigateToPose actions will be sent.')

    def status(self, message=''):
        data = {'state': self.state, 'message': message, 'time': time.time()}
        self.status_pub.publish(String(data=json.dumps(data, ensure_ascii=False)))
        if message:
            self.get_logger().info(f'[{self.state}] {message}')

    def fault(self, reason):
        self._turn(0.0)
        self.state = 'FAULT'
        self.status(reason)
        if self.undock_handle is not None:
            self.undock_handle.cancel_goal_async()
        self._cancel_nav(None)

    def on_clicked_point(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            return
        self.search_xy = (float(msg.point.x), float(msg.point.y))
        self.status(f'Search point set from RViz: ({self.search_xy[0]:.2f}, {self.search_xy[1]:.2f})')

    def on_event(self, msg):
        event = msg.data.strip().upper()
        if event == 'START':
            if self.state not in ('IDLE', 'HOLD'):
                self.status(f'Ignored START while state={self.state}')
                return
            if self.state == 'HOLD':  # 접근 후 START -> 다시 탐색
                self.locked_target_id = None
                self.target = None
                self.candidate_count = 0
                self.state = 'SEARCH'
                self.search_dwell_until = time.monotonic() + 2
                self.status('Resume search after approach')
                return
            if self.undock_pending:
                self.status('Undock action still pending; do not restart')
                return
            if not self.execute:
                self.status('DRY-RUN START: would undock, search near car point and approach car')
                return
            self.use_current_pose = not all(
                math.isfinite(v) for v in (self.init_x, self.init_y, self.init_yaw_deg))
            if not self.use_current_pose and not self.guard.cell((self.init_x, self.init_y)):
                self.fault('Initial pose outside supplied SLAM map')
                return
            self.target = None
            self.candidate_id = None
            self.candidate_count = 0
            self.locked_target_id = None
            self.state = 'WAIT_DOCK_STATUS'  # AMCL 확인은 도크 상태를 안 뒤로 미룸
            self.undock_deadline = time.monotonic() + 15
            self.status('Event accepted; checking dock status')
            self._start_if_status_available()
        elif event == 'STOP':
            self._turn(0.0)
            self.state = 'STOPPED'
            if self.undock_handle is not None:
                self.undock_handle.cancel_goal_async()
            self.target = None
            self.status('STOP: cancelling motion; no auto-dock')
            self._cancel_nav('IDLE')
            if self.nav_handle is None and not self.nav_goal_pending and not self.undock_pending:
                self.state = 'IDLE'
        elif event == 'RESET' and self.state in ('FAULT', 'STOPPED'):
            self.state = 'IDLE'
            self.status('Fault reset; verify hazards before START')

    def tick(self):
        now = time.monotonic()
        if now - self.last_heartbeat >= 1.0:  # 1초마다 상태 발행 (webcam_event 용, 로그 없음)
            self.last_heartbeat = now
            self.status_pub.publish(String(data=json.dumps(
                {'state': self.state, 'message': '', 'time': time.time()}, ensure_ascii=False)))
        if self.state == 'WAIT_DOCK_STATUS':
            if now > self.undock_deadline:
                self.fault('Dock status timeout')
            else:
                self._start_if_status_available()
        elif self.state == 'UNDOCKING':
            if now > self.undock_deadline:
                self.fault('Undock action timed out; check dock/robot condition')
        elif self.state == 'LOCALIZING':
            if self.use_current_pose:  # 도크 출발: 언독 시작 이후 AMCL 위치 + map TF 가 있어야 출발
                if now > self.localize_deadline:
                    self.fault('No fresh AMCL estimate after undock: check lidar is on (/robot3/scan), '
                               '2D Pose Estimate on dock, then RESET + START')
                elif (self.last_amcl_time > self.initialized_at and self.last_amcl_xy is not None
                      and self._robot_xy() is not None):
                    self.status(f'Fresh AMCL estimate after undock: '
                                f'({self.last_amcl_xy[0]:.2f}, {self.last_amcl_xy[1]:.2f}); '
                                f'searching car near search point')
                    self._go_search()
                return
            if now > self.localize_deadline:
                self.fault('AMCL localization timeout; check init pose, TF and map')
                return
            if now-self.last_pose_pub > 2.0:
                self._publish_initial_pose()
            if self.last_amcl_time > self.initialized_at and self.last_amcl_xy is not None:
                dx = self.last_amcl_xy[0]-self.init_x
                dy = self.last_amcl_xy[1]-self.init_y
                if math.hypot(dx, dy) < 0.9 and self._robot_xy() is not None:
                    self.status('Fresh AMCL estimate + map TF confirmed; searching car near search point')
                    self._go_search()
        elif self.state == 'SEARCH':  # 탐색 위치 근처로 이동
            if now >= self.search_dwell_until and not self.nav_goal_pending and self.nav_handle is None:
                self._go_search()
        elif self.state == 'SCAN':
            if now < self.scan_until:
                self._turn(self.scan_dir * self.scan_speed)
                return
            self._turn(0.0)
            robot = self._robot_xy()
            if robot is not None and math.dist(robot, self.search_xy) < 0.3:  # 탐색 지점에서 한 바퀴 -> 계속 회전
                self._start_scan(self.scan_dir, 'Car not found after full turn; spinning again')
            else:  # 다른 곳에서 놓친 뒤 한 바퀴 -> 탐색 지점(A) 으로 복귀
                self.status('Car not found after full turn; going back to search point')
                self.state = 'SEARCH'
                self.search_dwell_until = now + 0.5
        elif self.state == 'APPROACH':
            if self.cancel_pending is not None:
                return
            if self.target is None or now-self.target['seen'] > self.target_timeout_s:
                self._lose_target('Lost target during approach')
                self._cancel_nav('SCAN')  # 멈추고 차가 나간 쪽으로 회전
            elif self.nav_goal_xy and now-self.nav_sent_time > 2.5:  # 2.5초마다 목표 재계산 (follow)
                robot = self._robot_xy()
                if robot:
                    try:
                        updated = find_approach_goal(robot, (self.target['x'], self.target['y']),
                                                     self.guard, self.standoff_m,
                                                     restrict_zone=self.restrict_zone)
                        if updated is not None and math.dist(updated[:2], self.nav_goal_xy) > self.replan_distance_m:
                            self.status('Target moved; cancelling to recompute approach goal')
                            self._cancel_nav('APPROACH')
                    except ValueError:
                        self._lose_target('Target moved to unreachable place')
                        self._cancel_nav('SEARCH')
        elif self.state == 'HOLD':
            t = self.target
            if t is not None and now - t['seen'] < self.lost_turn_after_s:  # 보이는 동안: 화면 중앙에 유지
                b = t['bearing']
                wz = 0.0 if abs(b) < math.radians(self.track_deadband_deg) else \
                    max(-self.track_max_turn, min(self.track_max_turn, self.track_gain * b))
                self._turn(wz)
            else:  # 화면에서 사라짐 -> 나간 쪽으로 바로 회전
                self._turn(0.0)
                self._lose_target('Car left camera view in HOLD')
                self._start_scan(self._side(), 'Turning toward the side the car left')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true',
                    help='Allow physical undock and Nav2 motion on START event')
    ap.add_argument('--namespace', default='/robot3',
                    help="Robot namespace (default /robot3); TF is remapped to <ns>/tf")
    args, ros_args = ap.parse_known_args()
    ns = '/' + args.namespace.strip('/') if args.namespace.strip('/') else ''
    defaults = ['--ros-args', '-r', f'/tf:={ns}/tf', '-r', f'/tf_static:={ns}/tf_static']  # 사용자 --ros-args 가 우선
    if ns:
        defaults += ['-r', f'__ns:={ns}']
    rclpy.init(args=ros_args + defaults)
    node = None
    try:
        node = AMRController(execute=args.execute)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._cancel_nav('IDLE')
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
