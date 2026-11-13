"""AMRController 기능 분리: Nav2 이동 / 취소, 탐색 지점 이동, 제자리 회전, 접근 시작.

AMRController 가 이 클래스를 상속해 쓴다. self 의 상태·파라미터는 amr_ab_controller.py 의 __init__ 에서 만든다.
"""
import math
import time

from rclpy.time import Time
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import NavigateToPose
from tf2_ros import TransformException

from amr_ab_control.approach_core import find_approach_goal

IDLE_STATES = ('FAULT', 'STOPPED', 'IDLE')


class NavigationMixin:
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
