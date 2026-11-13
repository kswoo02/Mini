#!/usr/bin/env python3
"""minimap.pgm-based TurtleBot4 ROS 2 Jazzy Nav2 waypoint route.

Default action is PREVIEW ONLY, no robot commands are sent.
Actual execution needs --execute, a running Nav2 action server and live map->base_link TF.

Dependencies: python3-numpy, python3-opencv, python3-yaml;
when executing: rclpy, tf2_ros, nav2_msgs, action_msgs.
"""
import argparse
import heapq
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

from amr_ab_control.paths import MAP_YAML

# A/B goals derived from uploaded map at 0.05 m/pixel. NOT live-tested.
# World coordinates are expressed in the uploaded map's 'map' frame.
WAYPOINTS = [
    ("A_upper_link", -2.15, 2.20),  # 기존 1.40 -> 위쪽 방 안쪽으로 0.8m
    ("B_center", -1.00, -0.50),
]

# Screening clearance, not a replacement for Nav2 robot footprint + costmaps.
MIN_CLEARANCE_M = 0.35


class MapData:
    def __init__(self, path):
        self.yaml_file = Path(path).expanduser().resolve()
        with self.yaml_file.open(encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp)
        self.image_file = (self.yaml_file.parent / cfg["image"]).resolve()
        self.image = cv2.imread(str(self.image_file), cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Cannot load map image: {self.image_file}")
        self.height, self.width = self.image.shape
        self.res = float(cfg["resolution"])
        self.origin_x, self.origin_y, self.origin_yaw = map(float, cfg["origin"])
        self.free_thresh = float(cfg.get("free_thresh", 0.196))
        self.negate = int(cfg.get("negate", 0))
        if self.res <= 0:
            raise ValueError("resolution must be positive")
        # Map-server trinary convention: white=free when negate=0.
        grayscale = self.image.astype(np.float32) / 255.0
        occupancy_probability = grayscale if self.negate else (1.0 - grayscale)
        self.known_free = occupancy_probability < self.free_thresh
        self.clearance = cv2.distanceTransform(
            self.known_free.astype(np.uint8), cv2.DIST_L2, 5
        ) * self.res
        self.passable = self.clearance >= MIN_CLEARANCE_M
        _, self.components = cv2.connectedComponents(
            self.passable.astype(np.uint8), connectivity=4
        )

    def world_to_cell(self, x, y):
        dx, dy = x - self.origin_x, y - self.origin_y
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        col = math.floor(local_x / self.res)
        row = self.height - 1 - math.floor(local_y / self.res)
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            raise ValueError(f"({x:.2f}, {y:.2f}) outside the map")
        return row, col

    def validate_route(self, route):
        previous_component = None
        output = []
        for name, x, y in route:
            row, col = self.world_to_cell(x, y)
            gap = float(self.clearance[row, col])
            component = int(self.components[row, col])
            if not bool(self.passable[row, col]):
                raise ValueError(
                    f"{name} ({x:.2f}, {y:.2f}) is not screened safe: "
                    f"clearance {gap:.2f}m < {MIN_CLEARANCE_M:.2f}m"
                )
            if previous_component is not None and component != previous_component:
                raise ValueError(f"{name} is not 4-connected at {MIN_CLEARANCE_M}m clearance")
            output.append((name, x, y, row, col, gap))
            previous_component = component
        return output


def route_grid_path(map_data, start, goal):
    """A* over 4-connected cells screened to have >= MIN_CLEARANCE_M clearance.

    This is an occupancy-map feasibility preview, NOT a Nav2 global plan.
    """
    start = tuple(start)
    goal = tuple(goal)
    h, w = map_data.image.shape
    queue = [(0.0, 0, start)]
    came_from = {}
    distances = {start: 0.0}
    serial = 0
    while queue:
        _, _, curr = heapq.heappop(queue)
        if curr == goal:
            path = [curr]
            while curr != start:
                curr = came_from[curr]
                path.append(curr)
            return list(reversed(path))
        old_dist = distances[curr]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = curr[0] + dr, curr[1] + dc
            if not (0 <= nr < h and 0 <= nc < w) or not map_data.passable[nr, nc]:
                continue
            new_dist = old_dist + 1.0
            next_cell = (nr, nc)
            if new_dist < distances.get(next_cell, float("inf")):
                distances[next_cell] = new_dist
                came_from[next_cell] = curr
                serial += 1
                heuristic = abs(goal[0] - nr) + abs(goal[1] - nc)
                heapq.heappush(queue, (new_dist + heuristic, serial, next_cell))
    raise ValueError(f"No clearance-screened grid route from {start} to {goal}")


def write_preview(map_data, route, destination):
    # Grid A* path is a conservative 2D map preview, NOT the Nav2 planner output.
    canvas = cv2.cvtColor(map_data.image, cv2.COLOR_GRAY2BGR)
    scale = 5
    canvas = cv2.resize(canvas, (map_data.width * scale, map_data.height * scale), interpolation=cv2.INTER_NEAREST)
    for index in range(1, len(route)):
        prev, curr = route[index - 1], route[index]
        path = route_grid_path(map_data, (prev[3], prev[4]), (curr[3], curr[4]))
        pix = np.array([[(col * scale + scale // 2, row * scale + scale // 2)
                         for row, col in path]], dtype=np.int32)
        cv2.polylines(canvas, pix, False, (255, 150, 0), 2)

    for index, (name, x, y, row, col, gap) in enumerate(route):
        point = (col * scale + scale // 2, row * scale + scale // 2)
        cv2.circle(canvas, point, 9, (20, 150, 20), -1)
        cv2.circle(canvas, point, 9, (255, 255, 255), 2)
        cv2.putText(canvas, name.split('_')[0], (point[0] + 11, point[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 40, 190), 2)
    out = Path(destination).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), canvas):
        raise OSError(f"Couldn't write {out}")
    return out


def execute_nav2(route, map_data, action_name, map_frame, base_frames, namespace="/robot3"):
    # Imported only for --execute, so map validation works without ROS installed.
    import rclpy
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.duration import Duration
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener, TransformException

    # TurtleBot4 TF is published on <ns>/tf, not /tf -> remap the TF listener.
    ns = "/" + namespace.strip("/") if namespace.strip("/") else ""
    rclpy.init(args=["--ros-args", "-r", f"/tf:={ns}/tf", "-r", f"/tf_static:={ns}/tf_static"])
    node = rclpy.create_node("minimap_patrol")
    client = ActionClient(node, NavigateToPose, action_name)
    buffer = Buffer()
    listener = TransformListener(buffer, node, spin_thread=False)
    current_handle = None
    try:
        # Reject a non-localized robot: map -> base_link TF must exist and
        # its current position must be at a known-free map pixel.
        print(f"Checking localization TF {map_frame} -> one of {base_frames} ...")
        tf = None
        deadline = time.monotonic() + 8.0
        while rclpy.ok() and time.monotonic() < deadline and tf is None:
            rclpy.spin_once(node, timeout_sec=0.2)
            for base in base_frames:
                try:
                    tf = buffer.lookup_transform(map_frame, base, Time())
                    break
                except TransformException:
                    continue
        if tf is None:
            raise RuntimeError("Map-to-base_link TF missing. Set initial pose in RViz and verify frame names.")
        x = float(tf.transform.translation.x)
        y = float(tf.transform.translation.y)
        r, c = map_data.world_to_cell(x, y)
        if not map_data.known_free[r, c]:
            raise RuntimeError(f"Robot pose ({x:.2f},{y:.2f}) is not inside known free map space")
        print(f"Current robot pose in {map_frame}: ({x:.2f}, {y:.2f})")

        print(f"Waiting for Nav2 action server: {action_name}")
        if not client.wait_for_server(timeout_sec=6.0):
            raise RuntimeError(f"No Nav2 server at {action_name}. Check namespaces and Nav2 startup.")

        for idx, (name, x, y, row, col, clearance) in enumerate(route):
            next_x, next_y = ((route[idx + 1][1], route[idx + 1][2])
                              if idx + 1 < len(route)
                              else (route[idx - 1][1], route[idx - 1][2])
                              if idx > 0 else (x + 1.0, y))
            yaw = math.atan2(next_y - y, next_x - x)
            msg = PoseStamped()
            msg.header.frame_id = map_frame
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.orientation.w = math.cos(yaw / 2.0)
            goal = NavigateToPose.Goal()
            goal.pose = msg
            print(f"[{idx+1}/{len(route)}] Navigate to {name}: ({x:.2f},{y:.2f})")
            send_future = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(node, send_future)
            current_handle = send_future.result()
            if current_handle is None or not current_handle.accepted:
                raise RuntimeError(f"Nav2 rejected goal: {name}")
            result_future = current_handle.get_result_async()
            rclpy.spin_until_future_complete(node, result_future)
            result = result_future.result()
            if result is None or result.status != GoalStatus.STATUS_SUCCEEDED:
                status = result.status if result is not None else "no result"
                raise RuntimeError(f"Nav2 did not reach {name}, result status={status}")
            current_handle = None
        print("Route completed. Docking is NOT triggered automatically.")
    except KeyboardInterrupt:
        print("Interrupted. Cancelling active Nav2 goal...")
        if current_handle is not None:
            try:
                cancel = current_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(node, cancel, timeout_sec=3)
            except Exception as exc:
                print(f"Cancellation request failed: {exc}")
        raise
    finally:
        del listener
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Robot3 minimap A-B route: preview-only unless --execute")
    parser.add_argument("--map", default=str(MAP_YAML))
    parser.add_argument("--preview", default="minimap_waypoints.png", help="PNG preview output path")
    parser.add_argument("--execute", action="store_true", help="Send goals to a physical Nav2 robot")
    parser.add_argument("--reverse", action="store_true", help="Reverse route order (B to A)")
    parser.add_argument("--start-at", type=int, default=1, help="1-based route waypoint to start from")
    parser.add_argument("--nav-action", default="/robot3/navigate_to_pose",
                        help="Use /navigate_to_pose if Nav2 is not namespaced")
    parser.add_argument("--namespace", default="/robot3", help="Robot namespace for TF topics ('' if none)")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frames", default="base_link,robot3/base_link",
                        help="Comma-separated candidate robot TF frame names")
    args = parser.parse_args()
    if args.start_at < 1 or args.start_at > len(WAYPOINTS):
        parser.error(f"--start-at must be 1..{len(WAYPOINTS)}")
    map_data = MapData(args.map)
    checked_all = map_data.validate_route(WAYPOINTS)
    ordered = list(reversed(checked_all)) if args.reverse else checked_all
    route = ordered[args.start_at - 1:]
    out = write_preview(map_data, checked_all, args.preview)
    print(f"Map: {map_data.width}x{map_data.height} px, "
          f"resolution {map_data.res:.3f}m, origin "
          f"({map_data.origin_x:.3f},{map_data.origin_y:.3f},{map_data.origin_yaw:.3f})")
    print(f"Route: {len(route)} goals, image={out}")
    for name, x, y, row, col, gap in route:
        print(f"  {name:18s}: x={x:6.2f} y={y:6.2f} "
              f"map-pixel=({col:3d},{row:3d}) obstacle clearance≥~{gap:.2f}m")
    if not args.execute:
        print("PREVIEW ONLY. No Nav2 goals sent. Use --execute only after RViz initial-pose, map->base_link TF and costmap checks.")
        return 0
    execute_nav2(route, map_data, args.nav_action, args.map_frame,
                 [s.strip() for s in args.base_frames.split(',') if s.strip()], args.namespace)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
