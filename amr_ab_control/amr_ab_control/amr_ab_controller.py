#!/usr/bin/env python3
"""TurtleBot4 /robot3: START -> Undock -> A 로 이동 -> 제자리 회전 탐색 -> YOLO 감지 -> 접근 -> 화면 중앙 유지 + follow.
A/B 순찰 버전은 amr_ab_controller_patrol.py (태그 patrol-v1). --execute 없으면 dry-run."""
import argparse
import json
import math
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, TwistStamped
from irobot_create_msgs.msg import DockStatus
from irobot_create_msgs.action import Undock
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point

from amr_ab_control.paths import MAP_YAML, OAKD_MODEL
from amr_ab_control.approach_core import A, MapGuard, find_approach_goal, eligible_target_class, robust_depth_m

MISSION_STATES = ('NAV_SEARCH', 'SCAN', 'SEARCH', 'APPROACH', 'HOLD')  # 감지 -> 접근이 동작하는 상태
IDLE_STATES = ('FAULT', 'STOPPED', 'IDLE')


class AMRController(Node):
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

    def on_dock(self, msg):
        self.docked = bool(msg.is_docked)

    def on_clicked_point(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            return
        self.search_xy = (float(msg.point.x), float(msg.point.y))
        self.status(f'Search point set from RViz: ({self.search_xy[0]:.2f}, {self.search_xy[1]:.2f})')

    def on_amcl(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            return
        self.last_amcl_time = time.monotonic()
        self.last_amcl_xy = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))

    def on_camera_info(self, msg):
        if msg.k[0] > 0 and msg.k[4] > 0:
            self.camera_info = msg

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

    def _start_if_status_available(self):
        if self.state != 'WAIT_DOCK_STATUS' or self.docked is None:
            return
        if self.docked:
            if not self.undock_client.server_is_ready():
                if not self.undock_client.wait_for_server(timeout_sec=0.0):
                    self.fault('Undock action server missing (check namespace)')
                    return
            self.state = 'UNDOCKING'
            self.undock_pending = True
            self.undock_sent_at = time.monotonic()
            self.undock_deadline = self.undock_sent_at + 65
            self.status('Undock action sent' + (
                ' (AMCL pose will be confirmed after undock)' if self.use_current_pose else ''))
            fut = self.undock_client.send_goal_async(Undock.Goal())
            fut.add_done_callback(self._undock_goal_response)
        else:
            self.status('Robot already undocked, skipping Undock action')
            if self.use_current_pose and not self._check_current_pose():
                return
            self._initialize_pose()

    def _check_current_pose(self):  # 언도킹 상태 시작: 2D Pose Estimate 로 잡은 AMCL 위치가 있어야 출발
        if self.last_amcl_xy is None or self._robot_xy() is None:
            self.fault('No initial_x/y/yaw_deg and no AMCL pose: set 2D Pose Estimate in RViz, then RESET + START')
            return False
        self.status(f'Using current AMCL pose as initial pose: '
                    f'({self.last_amcl_xy[0]:.2f}, {self.last_amcl_xy[1]:.2f})')
        return True

    def _undock_goal_response(self, future):
        try:
            gh = future.result()
            if not gh or not gh.accepted:
                self.undock_pending = False
                if self.state == 'UNDOCKING':
                    self.fault('Undock action was rejected')
                elif self.state == 'STOPPED':
                    self.state = 'IDLE'
                return
            self.undock_handle = gh
            gh.get_result_async().add_done_callback(self._undock_result)
            if self.state != 'UNDOCKING':
                gh.cancel_goal_async()  # 수락 전에 STOP/FAULT 가 들어온 경우
        except Exception as e:
            self.undock_pending = False
            if self.state == 'UNDOCKING':
                self.fault(f'Undock action error: {e}')

    def _undock_result(self, future):
        self.undock_handle = None
        self.undock_pending = False
        if self.state == 'STOPPED':
            self.state = 'IDLE'
            self.status('Undock cancel/completion reported; verify dock status')
            return
        if self.state != 'UNDOCKING':
            return
        try:
            result = future.result()
            if result.status != GoalStatus.STATUS_SUCCEEDED:
                self.fault(f'Undock failed, action status={result.status}')
                return
            self._initialize_pose(after_undock=True)
        except Exception as e:
            self.fault(f'Undock result error: {e}')

    def _initialize_pose(self, after_undock=False):
        if self.use_current_pose and after_undock:
            self.state = 'LOCALIZING'
            self.initialized_at = self.undock_sent_at  # 언독 '도중' AMCL 갱신도 인정 (멈추면 AMCL 이 안 냄, 11/13 실기)
            self.localize_deadline = time.monotonic() + self.post_undock_localize_s
            self.status(f'Undocked; waiting up to {self.post_undock_localize_s:.0f}s '
                        f'for AMCL estimate since undock')
            return
        if self.use_current_pose:  # 이미 AMCL 이 추적 중 -> initialpose 덮어쓰지 않음
            if self._robot_xy() is None:
                self.fault('map->base_link TF unavailable; check AMCL / 2D Pose Estimate')
                return
            self.state = 'LOCALIZING'
            self.status('AMCL pose + map TF confirmed; searching car near search point')
            self._go_search()
            return
        self.state = 'LOCALIZING'  # initial_x/y/yaw 파라미터로 초기 위치 발행
        self.localize_deadline = time.monotonic() + 25
        self.initialized_at = time.monotonic()
        self.last_amcl_time = 0.0
        self.status('Publishing post-undock initial pose; waiting for fresh AMCL estimate')
        self._publish_initial_pose()

    def _publish_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = self.init_x, self.init_y
        yaw = math.radians(self.init_yaw_deg)
        msg.pose.pose.orientation.z = math.sin(yaw/2)
        msg.pose.pose.orientation.w = math.cos(yaw/2)
        msg.pose.covariance[0] = 0.10
        msg.pose.covariance[7] = 0.10
        msg.pose.covariance[35] = 0.12
        self.initial_pub.publish(msg)
        self.last_pose_pub = time.monotonic()

    def _robot_xy(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
            return float(tf.transform.translation.x), float(tf.transform.translation.y)
        except TransformException:
            return None

    def _goal_pose(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = Time().to_msg()  # stamp=0: 최신 TF 사용 (now() 면 정지 중 'TF too old' 로 ABORT)
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.orientation.w = math.cos(yaw / 2)
        return msg

    def _navigate(self, x, y, yaw, stage):
        if self.state in IDLE_STATES:
            return
        if self.nav_goal_pending or self.nav_handle is not None:
            self.status('Navigation busy; refusing overlapping goal')
            return
        self._turn(0.0)
        safe = self.guard.safe_goal if self.restrict_zone else self.guard.safe_goal_anywhere
        if not safe((x, y)):
            self.fault(f'Rejected {stage} goal: not free / too close to walls: ({x:.2f},{y:.2f})')
            return
        if not self.nav_client.wait_for_server(timeout_sec=0.0):
            self.fault('Nav2 navigate_to_pose action unavailable')
            return
        self.state = stage
        self.nav_goal_xy = (x, y)
        self.nav_sent_time = time.monotonic()
        self.nav_goal_pending = True
        self.nav_seq += 1
        seq = self.nav_seq
        req = NavigateToPose.Goal()
        req.pose = self._goal_pose(x, y, yaw)
        self.status(f'Nav2 goal ({x:.2f}, {y:.2f}), yaw={math.degrees(yaw):.1f} deg')
        f = self.nav_client.send_goal_async(req)
        f.add_done_callback(lambda fut: self._nav_accepted(fut, seq))

    def _nav_accepted(self, future, seq):
        if seq != self.nav_seq:
            return
        self.nav_goal_pending = False
        try:
            gh = future.result()
            if not gh or not gh.accepted:
                self.fault('Nav2 rejected goal')
                return
            self.nav_handle = gh
            gh.get_result_async().add_done_callback(lambda fut: self._nav_result(fut, seq))
            if self.cancel_pending is not None:
                gh.cancel_goal_async()
        except Exception as e:
            self.fault(f'Nav2 goal exception: {e}')

    def _nav_result(self, future, seq):
        if seq != self.nav_seq:
            return
        self.nav_handle = None
        self.nav_goal_pending = False
        try:
            status = future.result().status
        except Exception as e:
            self.fault(f'Nav2 result exception: {e}')
            return
        if self.state in ('FAULT', 'STOPPED') and self.cancel_pending != 'IDLE':
            self.cancel_pending = None
            return
        if self.cancel_pending is not None:  # 취소 후 다음 동작
            after = self.cancel_pending
            self.cancel_pending = None
            if after == 'IDLE':
                self.state = 'IDLE'
                self.status('Navigation stopped')
            elif after == 'APPROACH':
                self._start_approach()
            elif after == 'SEARCH':
                self.state = 'SEARCH'
                self.search_dwell_until = time.monotonic() + 2
                self.status('Target lost/moved: reacquiring')
            elif after == 'SCAN':
                self._start_scan(self._side(), 'Car left camera view; turning toward the side it left')
            return
        if self.state in IDLE_STATES:
            return
        if status != GoalStatus.STATUS_SUCCEEDED:
            if self.state == 'NAV_SEARCH' and self.nav_retries < 2:  # 1초 후 재시도 (최대 2회)
                self.nav_retries += 1
                self.state = 'SEARCH'
                self.search_dwell_until = time.monotonic() + 1.0
                self.status(f'Nav2 status={status}; retry {self.nav_retries}/2 to search point')
                return
            self.fault(f'Nav2 failed/canceled unexpectedly (status={status})')
            return
        self.nav_retries = 0
        if self.state == 'NAV_SEARCH':
            self._start_scan(1.0, 'Arrived at search point; spinning to find car')
        elif self.state == 'APPROACH':
            self.state = 'HOLD'
            self.hold_target_xy = None if self.target is None else (self.target['x'], self.target['y'])
            robot = self._robot_xy()
            if robot is not None and self.target is not None:
                d = math.dist(robot, (self.target['x'], self.target['y']))
                self.status(f'Approach goal reached; robot center -> car {d:.2f} m '
                            f'(target {self.standoff_m:.2f} m, last camera depth {self.target["depth_m"]:.2f} m, '
                            f'car map=({self.target["x"]:.2f},{self.target["y"]:.2f}), '
                            f'robot map=({robot[0]:.2f},{robot[1]:.2f})); waiting for START or STOP')
            else:
                self.status('Approach goal reached; waiting for START or STOP')

    def _cancel_nav(self, after):
        if after is not None:
            self.cancel_pending = after
        if self.nav_handle is not None:
            self.nav_handle.cancel_goal_async()
        elif self.nav_goal_pending:
            pass  # 수락되면 _nav_accepted 에서 취소
        elif self.cancel_pending is not None:
            task = self.cancel_pending
            self.cancel_pending = None
            if task == 'IDLE':
                self.state = 'IDLE'
            elif task == 'APPROACH':
                self._start_approach()
            elif task == 'SEARCH':
                self.state = 'SEARCH'
                self.search_dwell_until = time.monotonic() + 2
            elif task == 'SCAN':
                self._start_scan(self._side(), 'Car left camera view; turning toward the side it left')

    def _go_search(self):  # 탐색 지점(A) 으로 이동. 이미 도착해 있으면 바로 회전 탐색
        if self.state in IDLE_STATES:
            return
        robot = self._robot_xy()
        if robot is None:
            self.fault('Cannot transform base_link to map before search')
            return
        sx, sy = self.search_xy
        if math.dist(robot, (sx, sy)) < 0.3:
            self._start_scan(1.0, f'At search point ({sx:.2f},{sy:.2f}); spinning to find car')
            return
        self.status(f'Going to search point ({sx:.2f},{sy:.2f})')
        self._navigate(sx, sy, math.atan2(sy-robot[1], sx-robot[0]), 'NAV_SEARCH')

    def _start_scan(self, direction, message):  # 제자리에서 direction(+1 왼쪽/-1 오른쪽) 으로 한 바퀴 회전 (tick: SCAN)
        self.state = 'SCAN'
        self.scan_dir = 1.0 if direction >= 0 else -1.0
        self.scan_until = time.monotonic() + 2 * math.pi / max(self.scan_speed, 0.1)
        self.status(message)

    def _side(self):  # 차가 마지막으로 있던 쪽 (+1 왼쪽, -1 오른쪽)
        return 1.0 if self.last_bearing >= 0 else -1.0

    def _turn(self, wz):  # cmd_vel 로 제자리 회전. 0 이면 회전 중일 때만 정지 명령 1회
        if wz == 0.0 and not self.turning:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.angular.z = float(wz)
        self.cmd_pub.publish(msg)
        self.turning = wz != 0.0

    def _lose_target(self, reason):  # ID 잠금 해제 (마지막 방향 last_bearing 은 유지)
        last = '' if self.target is None else f' (last seen map=({self.target["x"]:.2f},{self.target["y"]:.2f}))'
        self.status(f'{reason}{last}; unlock ID {self.locked_target_id}')
        self.locked_target_id = None
        self.candidate_id = None
        self.candidate_count = 0
        self.target = None
        self.hold_target_xy = None

    def _start_approach(self):
        if self.state in IDLE_STATES:
            return
        if not self.depth_confirmed:
            self.state = 'SEARCH'
            self.status('Approach blocked: RGB-registered depth not confirmed')
            return
        if self.target is None or time.monotonic()-self.target['seen'] > self.target_timeout_s:
            self.state = 'SEARCH'
            self.search_dwell_until = time.monotonic() + 1.0
            self._lose_target('Target is stale')
            return
        robot = self._robot_xy()
        if robot is None:
            self.fault('Cannot transform base_link to map before approach')
            return
        try:
            goal = find_approach_goal(robot, (self.target['x'], self.target['y']),
                                      self.guard, self.standoff_m, restrict_zone=self.restrict_zone)
        except ValueError as e:
            self.state = 'SEARCH'
            self.search_dwell_until = time.monotonic() + 1
            self.status(f'Unsafe target approach rejected: {e}')
            return
        if goal is None:
            self.state = 'HOLD'
            self.hold_target_xy = (self.target['x'], self.target['y'])  # 기준 갱신 (안 하면 매 프레임 'moved' 반복, 11/13 실기)
            self.status('Already within standoff range; no movement')
            return
        self.locked_target_id = self.target['track_id']
        self._navigate(*goal, stage='APPROACH')

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
