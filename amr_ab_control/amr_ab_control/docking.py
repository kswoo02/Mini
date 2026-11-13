"""AMRController 기능 분리: 도킹 / 언독 / 초기 위치(AMCL) 확인.

AMRController 가 이 클래스를 상속해 쓴다. self 의 상태·파라미터는 amr_ab_controller.py 의 __init__ 에서 만든다.
"""
import math
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from irobot_create_msgs.action import Undock


class DockingMixin:
    def on_dock(self, msg):
        self.docked = bool(msg.is_docked)

    def on_amcl(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            return
        self.last_amcl_time = time.monotonic()
        self.last_amcl_xy = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))

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
