#!/usr/bin/env python3
"""[백업: A/B 순찰 버전, 2026-11-13 커밋 3e53455] Event-driven TurtleBot4 /robot3 controller: undock, localize, A-B, car approach.

Requires ROS 2 Jazzy, irobot_create_msgs, nav2_msgs, tf2_geometry_msgs,
message_filters, cv_bridge, ultralytics, PyYAML, NumPy, OpenCV.

SAFE DEFAULT: --execute must be specified. Further, no START mission begins
without user-provided initial pose. Target navigation is DISABLED unless
confirm_aligned_depth:=true is set after physical RGB/Depth alignment checks.
"""
import argparse
import json
import math
from pathlib import Path
import time

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
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
from irobot_create_msgs.msg import DockStatus
from irobot_create_msgs.action import Undock
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String
from collections import deque
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point

from amr_ab_control.paths import MAP_YAML, OAKD_MODEL
from amr_ab_control.approach_core import A, B, MapGuard, find_approach_goal, eligible_target_class, robust_depth_m


class AMRController(Node):
    def __init__(self, execute=False):
        super().__init__('amr_ab_controller')
        self.execute = execute
        def param(name, value):
            self.declare_parameter(name, value)
            return self.get_parameter(name).value

        # 기본: 패키지 models/oakd_best1.pt (turtle_car_dummy_v8n best1, 모델 비교 결과 OAK-D 용 최선)
        self.model_path = str(param('model_path', str(OAKD_MODEL)))
        self.map_yaml = str(param('map_yaml', str(MAP_YAML)))
        self.init_x = float(param('initial_x', float('nan')))
        self.init_y = float(param('initial_y', float('nan')))
        self.init_yaw_deg = float(param('initial_yaw_deg', float('nan')))
        # robot3 확인 결과: oakd/stereo/image_raw 는 RGB와 같은 frame/해상도(704x704)/intrinsics
        # (RGB-aligned depth) 이므로 기본 활성화.
        self.depth_confirmed = bool(param('confirm_aligned_depth', True))
        self.use_current_pose = False  # initial_x/y/yaw 미지정 시 현재 AMCL 위치 사용
        # 도크 출발 시 언도킹 후 새 AMCL 위치를 기다리는 최대 시간 (라이다 재기동 ~수 초 + AMCL 갱신)
        self.post_undock_localize_s = float(param('post_undock_localize_s', 20.0))
        # 압축 토픽 사용 (raw RGB 1452KB/장 -> compressed 122KB/장). WiFi 대역폭 때문에 raw 는
        # 실측 0.4 프레임/s, 지연 1.8s 로 감지 불가였음.
        self.rgb_topic = str(param('rgb_topic', 'oakd/rgb/image_raw/compressed'))
        self.depth_topic = str(param('depth_topic', 'oakd/stereo/image_raw/compressedDepth'))
        # 연속 N 프레임 car 여야 감지 인정 (1회 오인식 방지, 실기: 검은 SUV 를 car 0.70 으로 1회 오인식)
        self.confirm_frames = int(param('confirm_frames', 2))
        self.dummy_overlap = float(param('dummy_overlap', 0.3))   # car 박스가 dummy 박스와 이 비율 이상 겹치면 무시
        self.restrict_zone = bool(param('restrict_to_corridor', False))  # False: 순찰구역 밖 car 도 접근
        self.sync_slop = float(param('sync_slop', 0.2))
        self.logged_depth_format = False
        self.info_topic = str(param('camera_info_topic', 'oakd/rgb/camera_info'))
        self.map_frame = str(param('map_frame', 'map'))
        self.base_frame = str(param('base_frame', 'base_link'))
        # Nav2 접근 목표점: 차 중심에서 이 거리. Nav2 는 목표 반경 ~0.25 m 안이면 도착 판정하므로
        # 실제 정지 거리 = standoff_m + ~0.25 m. 실기(11/12): 0.80 -> 실제 1.03~1.08 m (4회)
        # -> 0.55 로 낮춰 실제 ~0.80 m 에서 정지하도록 보정.
        self.standoff_m = float(param('standoff_m', 0.55))
        self.min_conf = float(param('min_confidence', 0.7))  # 실기: 진짜 car 0.88~0.96, 오탐 0.51 -> 0.7 로 상향
        self.target_timeout_s = float(param('target_timeout_s', 1.5))
        self.replan_distance_m = float(param('replan_distance_m', 0.45))
        # 추적(tracking): ByteTrack ID 로 같은 car 를 계속 따라감
        self.use_tracking = bool(param('use_tracking', True))
        self.reid_distance_m = float(param('reid_distance_m', 0.5))   # ID 가 바뀌어도 이 거리 안이면 같은 car
        self.follow_in_hold = bool(param('follow_in_hold', True))     # 정지 후 car 가 움직이면 다시 따라감
        self.hold_target_xy = None
        self.lost_timeout_s = float(param('lost_timeout_s', 3.0))   # HOLD 중 car 가 이만큼 안 보이면 순찰 복귀
        self.guard = MapGuard(self.map_yaml)
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.state = 'IDLE'
        self.docked = None
        self.camera_info = None
        self.initialized_at = None
        self.last_pose_pub = 0.0
        self.last_amcl_time = 0.0
        self.last_amcl_xy = None
        self.undock_deadline = 0.0
        self.localize_deadline = 0.0
        self.mission_started = False
        self.last_frame_processed = 0.0
        self.candidate_id = None
        self.candidate_count = 0
        self.target = None
        self.locked_target_id = None
        self.nav_handle = None
        self.undock_handle = None
        self.undock_pending = False
        self.nav_goal_pending = False
        self.nav_seq = 0
        self.nav_retries = 0
        self.nav_goal_xy = None
        self.nav_sent_time = 0.0
        self.cancel_pending = None
        self.next_patrol = 'A'
        self.search_dwell_until = 0.0
        self.last_status = None

        self.undock_client = ActionClient(self, Undock, 'undock')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.initial_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.target_pub = self.create_publisher(String, 'amr/target_info', 10)
        self.status_pub = self.create_publisher(String, 'amr/status', 10)
        self.view_pub = self.create_publisher(CompressedImage, 'amr/camera_view/compressed', 1)
        self.create_subscription(String, 'amr/event', self.on_event, 10)
        self.create_subscription(DockStatus, 'dock_status', self.on_dock, qos_profile_sensor_data)
        # AMCL 은 amcl_pose 를 TRANSIENT_LOCAL 로 발행(정지 중엔 재발행 없음) -> 같은 QoS 로 받아야
        # 컨트롤러를 2D Pose Estimate 이후에 켜도 마지막 위치를 받을 수 있음
        amcl_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.on_amcl, amcl_qos)
        self.create_subscription(CameraInfo, self.info_topic, self.on_camera_info, qos_profile_sensor_data)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # FPS 향상: RGB 는 들어오는 대로 전부 처리(YOLO/화면), depth 는 버퍼에 쌓아두고
        # car 가 잡혔을 때만 RGB 시각과 가장 가까운(±sync_slop) depth 를 골라 디코딩.
        # (기존 ApproximateTimeSynchronizer + 0.25s 제한 -> 실측 2.3fps)
        self.depth_buf = deque(maxlen=20)
        self.create_subscription(CompressedImage, self.depth_topic, lambda m: self.depth_buf.append(m),
                                 qos_profile_sensor_data)
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
        self.state = 'FAULT'
        self.mission_started = False
        self.status(reason)
        if self.undock_handle is not None:
            self.undock_handle.cancel_goal_async()
        self._cancel_nav(None)

    def on_dock(self, msg):
        self.docked = bool(msg.is_docked)

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
            if self.state == 'HOLD':
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
                self.status('DRY-RUN START: would undock, publish initial pose, navigate A->B and approach car')
                return
            self.use_current_pose = not all(
                math.isfinite(v) for v in (self.init_x, self.init_y, self.init_yaw_deg))
            # use_current_pose 일 때 AMCL 위치 확인은 도크 상태를 안 뒤(_start_if_status_available)로 미룸:
            # 도크 위에서는 power_saver 가 라이다/OAK-D 를 꺼서 AMCL 이 위치(map TF)를 확정하지 못함
            if not self.use_current_pose and not self.guard.cell((self.init_x, self.init_y)):
                self.fault('Initial pose outside supplied SLAM map')
                return
            self.target = None
            self.candidate_id = None
            self.candidate_count = 0
            self.locked_target_id = None
            self.mission_started = True
            self.state = 'WAIT_DOCK_STATUS'
            self.undock_deadline = time.monotonic() + 15
            self.status('Event accepted; checking dock status')
            self._start_if_status_available()
        elif event == 'STOP':
            self.state = 'STOPPED'
            if self.undock_handle is not None:
                self.undock_handle.cancel_goal_async()
            self.target = None
            self.mission_started = False
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

    def _check_current_pose(self):
        """언도킹 상태에서 시작: RViz 2D Pose Estimate 로 잡아둔 현재 AMCL 위치가 있어야 출발."""
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
                gh.cancel_goal_async()  # STOP/FAULT arrived before action accepted
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
            # 도크에서 출발: AMCL 이 (도크 위에서 찍은 2D Pose Estimate + 언도킹 이동량) 으로 위치를 갱신함.
            # 기준 시각 = Undock 명령을 보낸 순간. AMCL 은 '움직일 때만' amcl_pose 를 내므로
            # 언도킹 '완료' 이후만 인정하면, 멈춰 있는 로봇은 새 위치가 안 와서 타임아웃 FAULT 가 남
            # (11/13 실기: 언도킹 중 +2.6~4.3s 에 13회 갱신, 완료는 +6.9s -> 20s 뒤 FAULT).
            # 도크 위 라이다가 꺼져 있으면(power_saver) 언도킹 이동량 때문에 첫 스캔에서 갱신됨.
            self.state = 'LOCALIZING'
            self.initialized_at = getattr(self, 'undock_sent_at', time.monotonic())
            self.localize_deadline = time.monotonic() + self.post_undock_localize_s
            self.status(f'Undocked; waiting up to {self.post_undock_localize_s:.0f}s '
                        f'for AMCL estimate since undock')
            return
        if self.use_current_pose:
            # AMCL 이 이미 위치를 추적 중 -> initialpose 를 덮어쓰지 않고 TF만 확인 후 출발
            if self._robot_xy() is None:
                self.fault('map->base_link TF unavailable; check AMCL / 2D Pose Estimate')
                return
            self.state = 'LOCALIZING'
            self.status('AMCL pose + map TF confirmed; navigating to A')
            self._navigate(A[0], A[1], -0.95, 'NAV_A')
            return
        self.state = 'LOCALIZING'
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
        # stamp=0 -> Nav2 가 '가장 최근 TF' 사용. now() 로 찍으면 로봇 정지 중 AMCL 이 map->odom 을
        # 갱신하지 않아 "Transform data too old" 로 목표가 즉시 ABORT 됨 (실기에서 확인)
        msg.header.stamp = Time().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.orientation.w = math.cos(yaw / 2)
        return msg

    def _navigate(self, x, y, yaw, stage):
        if self.state in ('FAULT', 'IDLE', 'STOPPED'):
            return
        if self.nav_goal_pending or self.nav_handle is not None:
            self.status('Navigation busy; refusing overlapping goal')
            return
        safe = (self.guard.safe_goal_anywhere if (stage == 'APPROACH' and not self.restrict_zone)
                else self.guard.safe_goal)
        if not safe((x, y)):
            self.fault(f'Rejected {stage} goal outside clearance-screened A-B corridor: ({x:.2f},{y:.2f})')
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
        if self.cancel_pending is not None:
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
            return
        if self.state in ('FAULT', 'STOPPED', 'IDLE'):
            return
        if status != GoalStatus.STATUS_SUCCEEDED:
            if self.state in ('NAV_A', 'NAV_B') and self.nav_retries < 2:
                # 순찰 목표 실패 시 1초 후 같은 지점 재시도 (최대 2회)
                self.nav_retries += 1
                self.next_patrol = 'A' if self.state == 'NAV_A' else 'B'
                self.state = 'SEARCH'
                self.search_dwell_until = time.monotonic() + 1.0
                self.status(f'Nav2 status={status}; retry {self.nav_retries}/2 to {self.next_patrol}')
                return
            self.fault(f'Nav2 failed/canceled unexpectedly (status={status})')
            return
        self.nav_retries = 0
        if self.state == 'NAV_A':
            if not self.mission_started:
                return
            self._navigate(B[0], B[1], -1.0, 'NAV_B')
        elif self.state == 'NAV_B':
            self.state = 'SEARCH'
            self.next_patrol = 'A'
            self.search_dwell_until = time.monotonic() + 2.0
            self.status('Arrived B; searching for valid car target')
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
            # Once the goal is accepted, the result handler will cancel it.
            pass
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

    def _lose_target(self, reason):
        """추적 대상을 놓침 -> ID 잠금 해제. 순찰 루트(A<->B)로 돌아가 어떤 car 든 다시 찾으면 재추적."""
        last = '' if self.target is None else f' (last seen map=({self.target["x"]:.2f},{self.target["y"]:.2f}))'
        self.status(f'{reason}{last}; unlock ID {self.locked_target_id}, back to patrol route to re-acquire')
        self.locked_target_id = None
        self.candidate_id = None
        self.candidate_count = 0
        self.target = None
        self.hold_target_xy = None

    def _start_approach(self):
        if self.state in ('FAULT', 'STOPPED', 'IDLE'):
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
            self.status('Already within standoff range; no movement')
            return
        self.locked_target_id = self.target['track_id']
        self._navigate(*goal, stage='APPROACH')

    def _decode_compressed_depth(self, msg):
        """compressedDepth (format '16UC1; compressedDepth png': 12byte 헤더 + PNG) -> uint16(mm) 배열."""
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
        if depth.dtype != np.uint16:  # 32FC1 은 inverse-depth 양자화 방식이라 미지원
            raise ValueError(f'compressedDepth decoded dtype {depth.dtype} (expected uint16 mm)')
        return depth

    @staticmethod
    def _overlap_ratio(a, b):
        """교집합 / 작은 박스 면적 (같은 물체에 car·dummy 가 동시에 잡혔는지 판단)."""
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        area = min((a[2]-a[0]) * (a[3]-a[1]), (b[2]-b[0]) * (b[3]-b[1]))
        return inter / area if area > 0 else 0.0

    def _publish_view(self, img, candidates):
        """감지 결과를 그린 터틀봇 카메라 화면을 이 PC 로컬 토픽으로 발행 (WiFi 추가 사용 없음)."""
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

    def _depth_for(self, rgb_msg, shape):
        """RGB 시각과 가장 가까운 depth(±sync_slop) 를 디코딩해서 반환. 없으면 (None, 이유)."""
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
        self.last_frame_processed = now
        # 화면(amr/camera_view)은 모든 상태에서 갱신, 감지 -> 접근 동작은 미션 상태에서만
        mission = self.state in ('NAV_A','NAV_B','SEARCH','APPROACH','HOLD')
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
        view = results.plot()  # car/dummy 모든 박스 + conf
        candidates = []
        dummy_boxes = [b.xyxy[0].tolist() for b in results.boxes
                       if str(self.model.names[int(b.cls[0])]).lower().strip() == 'dummy']
        for box in results.boxes:
            cls = int(box.cls[0])
            name = str(self.model.names[cls]).lower().strip()
            if not eligible_target_class(name):
                continue  # Explicitly exclude dummy and any other class
            ident = int(box.id[0]) if (self.use_tracking and box.id is not None) else -1
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            u, v = (x1+x2)//2, (y1+y2)//2
            reject = None
            if any(self._overlap_ratio((x1, y1, x2, y2), d) >= self.dummy_overlap for d in dummy_boxes):
                reject = 'DUMMY OVERLAP'  # 같은 물체가 car/dummy 동시에 잡힘 -> 애매하면 접근 안 함
            if reject:
                cv2.putText(view, f'IGNORED: {reject}', (x1, min(y2 + 22, view.shape[0] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                if mission:
                    self.get_logger().info(f'OAK-D car conf={confidence:.2f} px=({u},{v}) ignored: {reject}',
                                           throttle_duration_sec=1.0)
                continue
            depth, why = get_depth()
            z = None if depth is None else robust_depth_m(depth, (x1,y1,x2,y2), '16UC1')
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
            pt.point.x, pt.point.y, pt.point.z = (u-cx)*z/fx, (v-cy)*z/fy, z
            try:
                tf = self.tf_buffer.lookup_transform(self.map_frame,
                    rgb_msg.header.frame_id, Time.from_msg(rgb_msg.header.stamp),
                    timeout=Duration(seconds=0.0))
            except Exception:
                try:  # 이미지 시각의 TF 가 없으면 최신 TF 로 대체 (WiFi 지연 대응)
                    tf = self.tf_buffer.lookup_transform(self.map_frame,
                        rgb_msg.header.frame_id, Time(), timeout=Duration(seconds=0.0))
                except Exception as e:
                    self.get_logger().warn(f'TF map<-camera failed: {e}', throttle_duration_sec=2.0)
                    continue
            mapped = do_transform_point(pt, tf)
            candidates.append({'class': name, 'track_id': ident, 'confidence': confidence,
                'pixel': [u,v], 'depth_m': round(z,3), 'x': float(mapped.point.x),
                'y': float(mapped.point.y), 'seen': now})
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
            # 1) 잠근 ID 그대로 추적
            selected = next((c for c in candidates if c['track_id'] == self.locked_target_id), None)
            if selected is None and self.target is not None:
                # 2) ID 가 끊겼으면(가림/프레임 누락) 마지막 위치 근처 car 를 같은 대상으로 재연결
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
        report = {k: v for k,v in selected.items() if k != 'seen'}
        self.target_pub.publish(String(data=json.dumps(report, ensure_ascii=False)))
        if self.state in ('NAV_A', 'NAV_B') and self.cancel_pending is None:
            self.status(f'Confirmed car ID {ident}; canceling patrol for approach')
            self._cancel_nav('APPROACH')
        elif self.state == 'SEARCH':
            self._start_approach()
        elif (self.state == 'HOLD' and self.follow_in_hold and self.hold_target_xy is not None
              and math.dist((selected['x'], selected['y']), self.hold_target_xy) > self.replan_distance_m):
            # 추적 중인 car 가 정지 후 움직임 -> 다시 따라감
            self.status(f'Tracked car ID {ident} moved '
                        f'{math.dist((selected["x"], selected["y"]), self.hold_target_xy):.2f} m; following')
            self.state = 'SEARCH'  # _start_approach 진입 조건
            self._start_approach()

    def tick(self):
        now = time.monotonic()
        if now - getattr(self, '_last_heartbeat', 0.0) >= 1.0:
            # 1초마다 현재 상태 발행 (webcam_event 화면/자동 START 판단용, 로그는 남기지 않음)
            self._last_heartbeat = now
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
            if self.use_current_pose:
                # 도크 출발 후: 언도킹 완료 이후에 들어온 amcl_pose + map TF 가 있어야 출발
                if now > self.localize_deadline:
                    self.fault('No fresh AMCL estimate after undock: check lidar is on (/robot3/scan), '
                               '2D Pose Estimate on dock, then RESET + START')
                elif (self.last_amcl_time > self.initialized_at and self.last_amcl_xy is not None
                      and self._robot_xy() is not None):
                    self.status(f'Fresh AMCL estimate after undock: '
                                f'({self.last_amcl_xy[0]:.2f}, {self.last_amcl_xy[1]:.2f}); navigating to A')
                    self._navigate(A[0], A[1], -0.95, 'NAV_A')
                return
            if now > self.localize_deadline:
                self.fault('AMCL localization timeout; check init pose, TF and map')
                return
            if now-self.last_pose_pub > 2.0:
                self._publish_initial_pose()
            if self.last_amcl_time > self.initialized_at and self.last_amcl_xy is not None:
                dx = self.last_amcl_xy[0]-self.init_x
                dy = self.last_amcl_xy[1]-self.init_y
                if math.hypot(dx,dy) < 0.9 and self._robot_xy() is not None:
                    self.status('Fresh AMCL estimate + map TF confirmed; navigating to A')
                    self._navigate(A[0], A[1], -0.95, 'NAV_A')
        elif self.state == 'SEARCH':
            if now >= self.search_dwell_until and not self.nav_goal_pending and self.nav_handle is None:
                if self.next_patrol == 'A':
                    self.next_patrol = 'B'
                    self._navigate(A[0], A[1], 0.0, 'NAV_A')
                else:
                    self.next_patrol = 'A'
                    self._navigate(B[0], B[1], -1.0, 'NAV_B')
        elif self.state == 'APPROACH':
            if self.cancel_pending is not None:
                return
            if self.target is None or now-self.target['seen'] > self.target_timeout_s:
                self._lose_target('Lost target during approach')
                self._cancel_nav('SEARCH')
            elif self.nav_goal_xy and now-self.nav_sent_time > 2.5:
                robot = self._robot_xy()
                if robot:
                    try:
                        updated = find_approach_goal(robot, (self.target['x'],self.target['y']),
                                                      self.guard,self.standoff_m,
                                                      restrict_zone=self.restrict_zone)
                        if updated is not None and math.dist(updated[:2], self.nav_goal_xy) > self.replan_distance_m:
                            self.status('Target moved; cancelling to recompute approach goal')
                            self._cancel_nav('APPROACH')
                    except ValueError:
                        self._lose_target('Target moved to unreachable place')
                        self._cancel_nav('SEARCH')
        elif self.state == 'HOLD' and self.follow_in_hold:
            # 정지 후 추적 중인 car 가 시야에서 사라지면 -> 순찰 루트로 복귀해서 다시 찾기
            if self.target is None or now - self.target['seen'] > self.lost_timeout_s:
                self._lose_target(f'Tracked car not seen for {self.lost_timeout_s:.0f}s in HOLD')
                self.state = 'SEARCH'
                self.search_dwell_until = now + 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true',
                    help='Allow physical undock and Nav2 motion on START event')
    ap.add_argument('--namespace', default='/robot3',
                    help="Robot namespace (default /robot3); TF is remapped to <ns>/tf")
    args, ros_args = ap.parse_known_args()
    # 사용자가 준 --ros-args 가 먼저 적용되고(first match), 없으면 /robot3 + TF 리맵 기본값 사용
    ns = '/' + args.namespace.strip('/') if args.namespace.strip('/') else ''
    defaults = ['--ros-args', '-r', f'/tf:={ns}/tf', '-r', f'/tf_static:={ns}/tf_static']
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
