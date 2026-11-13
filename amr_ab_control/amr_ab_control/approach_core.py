"""ROS-independent AMR target standoff geometry and SLAM-map corridor screening."""
import heapq
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

A = (-2.15, 2.20)  # 위쪽 방 안쪽으로 0.8m 더 진입 (벽까지 ~1.0m), 기존 (-2.15, 1.40)
B = (-1.00, -0.50)


def standoff_goal(robot_xy, target_xy, distance=0.8):
    """Return (goal_x, goal_y, face_target_yaw), or None if close enough.

    The goal is located distance meters before the detected object, as viewed
    from the robot's current planar position. No collision checking here.
    """
    rx, ry = map(float, robot_xy)
    tx, ty = map(float, target_xy)
    dx, dy = tx - rx, ty - ry
    norm = math.hypot(dx, dy)
    if not all(map(math.isfinite, (rx, ry, tx, ty))) or distance <= 0:
        raise ValueError('Nonfinite coordinate or invalid standoff distance')
    if norm <= distance:
        return None
    x = tx - distance * dx / norm
    y = ty - distance * dy / norm
    return (x, y, math.atan2(ty - y, tx - x))


def eligible_target_class(name):
    """Only the exact car class is eligible, never dummy or other objects."""
    return str(name).strip().lower() == 'car'


def robust_depth_m(depth, bbox, encoding):
    """Median central ROI depth in meters; ignore zero/out-of-range pixels."""
    import numpy as np
    x1, y1, x2, y2 = map(int, bbox)
    u, v = (x1+x2)//2, (y1+y2)//2
    h, w = depth.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None
    k = max(2, min(x2-x1, y2-y1)//12)
    roi = depth[max(0,v-k):min(h,v+k+1), max(0,u-k):min(w,u+k+1)].astype('float64')
    if encoding == '16UC1':
        roi /= 1000.0
    elif encoding != '32FC1':
        return None
    valid = roi[np.isfinite(roi) & (roi > 0.30) & (roi < 5.0)]
    if valid.size < 6:
        return None
    return float(np.median(valid))


class MapGuard:
    """Restrict approach goals to free cells near a clearance-screened A-B route.

    Geometry screen only. This does NOT configure Nav2 Keepout Filters and does
    NOT guarantee the whole planned/actual path stays inside this corridor.
    """

    def __init__(self, yaml_path, clearance_m=0.35, corridor_half_width_m=1.0):
        cfg_file = Path(yaml_path).expanduser().resolve()
        cfg = yaml.safe_load(cfg_file.read_text(encoding='utf8'))
        self.image = cv2.imread(str(cfg_file.parent / cfg['image']), cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError('SLAM map pgm/png not readable')
        self.resolution = float(cfg['resolution'])
        self.ox, self.oy, self.oyaw = map(float, cfg['origin'])
        if self.resolution <= 0:
            raise ValueError('map resolution must be positive')
        raw = self.image.astype('float32') / 255.0
        occ_prob = raw if int(cfg.get('negate', 0)) else (1.0 - raw)
        free = (occ_prob < float(cfg.get('free_thresh', 0.196))).astype('uint8')
        self.clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5) * self.resolution
        self.passable = self.clearance >= clearance_m
        self.h, self.w = self.image.shape
        acell = self.cell(A)
        bcell = self.cell(B)
        route = self._a_star(acell, bcell)
        mask = np.zeros((self.h, self.w), dtype='uint8')
        for row, col in route:
            mask[row, col] = 255
        # Dilating makes a local route region, not a strict physical stop boundary.
        radius = max(1, math.ceil(corridor_half_width_m / self.resolution))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius + 1, 2*radius + 1))
        self.zone = cv2.dilate(mask, kernel) > 0
        self.allowed_free = self.passable & self.zone
        num, labels = cv2.connectedComponents(self.allowed_free.astype('uint8'), connectivity=4)
        self.labels = labels
        self.base_component = int(labels[bcell])
        if self.base_component == 0 or int(labels[acell]) != self.base_component:
            raise ValueError('A and B not connected in allowed corridor')
        # 지도 전체에서 (clearance 확보 + B 와 연결된) 주행 가능 영역 - 구역 제한 없이 접근할 때 사용
        _, self.all_labels = cv2.connectedComponents(self.passable.astype('uint8'), connectivity=4)
        self.all_component = int(self.all_labels[bcell])

    def cell(self, xy):
        x, y = map(float, xy)
        dx, dy = x-self.ox, y-self.oy
        c, s = math.cos(self.oyaw), math.sin(self.oyaw)
        local_x, local_y = c*dx+s*dy, -s*dx+c*dy
        col = math.floor(local_x/self.resolution)
        row = self.image.shape[0] - 1 - math.floor(local_y/self.resolution)
        if row < 0 or row >= self.image.shape[0] or col < 0 or col >= self.image.shape[1]:
            return None
        return row, col

    def target_in_zone(self, xy):
        p = self.cell(xy)
        return p is not None and bool(self.zone[p])

    def safe_goal(self, xy):
        p = self.cell(xy)
        return p is not None and bool(self.allowed_free[p]) and int(self.labels[p]) == self.base_component

    def safe_goal_anywhere(self, xy):
        """Free + clearance-screened + connected to B anywhere on the map (no A-B corridor limit)."""
        p = self.cell(xy)
        return p is not None and bool(self.passable[p]) and int(self.all_labels[p]) == self.all_component

    def _a_star(self, start, end):
        if start is None or end is None or not self.passable[start] or not self.passable[end]:
            raise ValueError('A or B not navigable at configured map clearance')
        queue = [(0, 0, start)]
        cost = {start: 0}
        parent = {}
        count = 0
        while queue:
            f, _, cur = heapq.heappop(queue)
            if cur == end:
                route = [cur]
                while cur != start:
                    cur = parent[cur]
                    route.append(cur)
                return route[::-1]
            for dr, dc in ((0, 1),(0, -1),(1, 0),(-1, 0)):
                nxt = cur[0]+dr, cur[1]+dc
                if not (0 <= nxt[0] < self.h and 0 <= nxt[1] < self.w and self.passable[nxt]):
                    continue
                new_g = cost[cur] + 1
                if new_g >= cost.get(nxt, float('inf')):
                    continue
                cost[nxt] = new_g
                parent[nxt] = cur
                count += 1
                heuristic = abs(nxt[0]-end[0]) + abs(nxt[1]-end[1])
                heapq.heappush(queue, (new_g+heuristic, count, nxt))
        raise ValueError('No A-B path with requested map clearance')


def find_approach_goal(robot_xy, target_xy, guard, standoff_m=0.8, restrict_zone=True):
    """Try straight-ahead goal, then nearby angular alternatives around target.

    Return None when robot is already close enough. Raise ValueError when no
    reachable, clearance-screened alternative exists.
    restrict_zone=False: target/goal may be anywhere on the map (still free +
    clearance-screened + connected), not only inside the A-B corridor.
    """
    goal = standoff_goal(robot_xy, target_xy, standoff_m)
    if goal is None:
        return None
    if restrict_zone and not guard.target_in_zone(target_xy):
        raise ValueError('Target outside A-B mission zone')
    is_safe = guard.safe_goal if restrict_zone else guard.safe_goal_anywhere
    if is_safe(goal[:2]):
        return goal
    tx, ty = target_xy
    angle = math.atan2(goal[1]-ty, goal[0]-tx)
    for radial_scale in (1.0, 1.25, 1.5):
        for offset_deg in (15, -15, 30, -30, 45, -45, 60, -60, 90, -90):
            test_angle = angle + math.radians(offset_deg)
            radius = standoff_m * radial_scale
            x = tx + radius * math.cos(test_angle)
            y = ty + radius * math.sin(test_angle)
            if is_safe((x, y)):
                return x, y, math.atan2(ty-y, tx-x)
    raise ValueError('No clear, connected standoff goal' + (' within A-B zone' if restrict_zone else ''))
