# amr_ab_control: TurtleBot4 A/B 순찰 + 차량 접근 (ROS 2 Jazzy)

웹캠 car 감지(START) → Undock → **A(위쪽 방)로 이동 → 제자리에서 회전하며 찾기** → OAK-D YOLO로 car 감지 → 0.8 m 앞까지 접근 → **정지 중에는 차를 화면 가운데에 유지**, 차가 움직이면 따라감 → **화면에서 사라지면 나간 쪽으로 바로 회전** → 한 바퀴 돌아도 없으면 A로 돌아가 다시 회전.

> A/B 순찰 버전은 `amr_ab_controller_patrol`로 남겨 두었습니다(git 태그 `patrol-v1`).

## 패키지 구성

```
amr_ab_control/
├── amr_ab_control/
│   ├── amr_ab_controller.py   메인 컨트롤러 (START → 차 위치 근처 탐색 → 감지 → 접근 → follow)
│   ├── docking.py             └ 도크 상태, Undock, 초기 위치·AMCL 확인
│   ├── navigation.py          └ Nav2 이동·취소, 탐색 지점 이동, 제자리 회전, 접근
│   ├── perception.py          └ RGB/depth 동기화, YOLO 감지·추적, 감지 화면
│   ├── amr_ab_controller_patrol.py  백업: A/B 순찰 버전
│   ├── approach_core.py       A/B 좌표, 접근 목표 계산, 지도 안전 검사 (ROS 비의존)
│   ├── ab_nav_only.py         A → B 이동만 따로 시험 (기본은 프리뷰)
│   ├── webcam_event.py        웹캠 car 감지 → START 이벤트
│   ├── turtle_cam_view.py     터틀봇 카메라 화면 (YOLO 박스, 거리)
│   ├── bag_to_mp4.py          bag 안의 카메라 영상을 mp4로 변환
│   └── paths.py               지도, 모델 경로
├── maps/minimap.{yaml,pgm}    순찰 지도
├── models/
│   ├── oakd_best1.pt              로봇 OAK-D용 (기본값)
│   ├── webcam_car_dummy_aug.pt    웹캠 START용 (기본값)
│   └── nano_colab.pt              Colab에서 학습한 nano (웹캠용 후보)
├── scripts/record_test.sh     테스트 bag 녹화
├── test/test_approach_core.py 오프라인 유닛 테스트 11개
└── docs/ab_preview.png        A/B 경로 지도 프리뷰
```

순찰 좌표(`map` frame): **A = (-2.15, 2.20)**(위쪽 방 안쪽), **B = (-1.00, -0.50)**

## 빌드

```bash
cd ~/rokey_ws
colcon build --symlink-install --packages-select amr_ab_control
source install/setup.bash
```

`--symlink-install`로 빌드했기 때문에 `.py` 파일을 고친 뒤에는 **다시 빌드할 필요가 없습니다.** 파일을 새로 추가했거나 `setup.py`를 고쳤을 때만 다시 빌드하세요.

pip로 따로 설치해야 하는 것: `ultralytics`, `lap`(추적용)

## 실행 (터미널마다 순서대로)

모든 터미널에서 먼저 `source ~/rokey_ws/install/setup.bash`를 실행하세요.

| # | 명령 | 다음 단계로 넘어가는 시점 |
|---|---|---|
| ① | `ros2 launch turtlebot4_navigation localization.launch.py namespace:=/robot3 map:=$(ros2 pkg prefix amr_ab_control)/share/amr_ab_control/maps/minimap.yaml` | `Managed nodes are active` |
| ② | `ros2 launch turtlebot4_navigation nav2.launch.py namespace:=/robot3` | `Managed nodes are active` |
| ③ | `ros2 launch turtlebot4_viz view_navigation.launch.py namespace:=/robot3` | 2D Pose Estimate를 찍은 뒤 |
| ④ | `ros2 run amr_ab_control amr_ab_controller --execute` | `controller ready`가 뜨고 10초 뒤 |
| ⑤ | `ros2 run amr_ab_control turtle_cam_view` | (선택) 카메라 화면 |
| ⑥ | `ros2 run amr_ab_control record_test.sh <메모>` | (선택) 녹화, Ctrl+C로 저장 |
| ⑦ | `ros2 run amr_ab_control webcam_event` | `ROBOT: IDLE`이 보이면 차를 웹캠 앞에 |

- **도크에서 출발할 때:** ③에서 2D Pose Estimate를 **도크 위 로봇 위치와 방향 그대로** 찍고 START를 보내세요. 컨트롤러가 Undock을 하고, **Undock 명령 이후**에 들어온 AMCL 위치와 map TF를 확인(최대 20초, `post_undock_localize_s`)한 뒤 A로 출발합니다. robot3는 도크 위에서도 라이다가 켜져 있습니다(11/13 확인). `power_saver`가 켜진 로봇이면 도크 위에서 스캔이 안 보일 수 있는데, 그래도 동작합니다.
- `--execute`가 없으면 dry-run입니다. 이때는 Undock 명령과 Nav2 목표를 보내지 않습니다.
- 웹캠 없이 START만 보내려면: `ros2 topic pub --once /robot3/amr/event std_msgs/msg/String "{data: START}"`
- 정지: `ros2 topic pub --once /robot3/amr/event std_msgs/msg/String "{data: STOP}"`. **STOP은 작업 취소일 뿐, 비상정지가 아닙니다.**
- 상태 보기: `ros2 topic echo /robot3/amr/status`, `ros2 topic echo /robot3/amr/target_info`

### 회전 탐색 위치 바꾸기

- 기본은 **A (-2.15, 2.20), 위쪽 방**입니다. 실행 중에 RViz 위쪽 툴바의 **Publish Point**를 누르고 지도를 클릭하면 바뀝니다.
- 처음부터 정하려면: `--ros-args -p search_x:=-1.0 -p search_y:=-0.5`
- A로 가는 도중에 본 차는 무시합니다. 가는 도중에도 접근하게 하려면 `-p detect_on_way:=true`
- 회전은 Nav2가 아니라 `cmd_vel`로 직접 합니다. 속도는 `scan_speed`(0.5 rad/s, 한 바퀴 약 13초)입니다.

### 모델이나 파라미터 바꾸기

```bash
# OAK-D 모델 변경
ros2 run amr_ab_control amr_ab_controller --execute --ros-args -p model_path:=/경로/모델.pt
# 웹캠 모델 변경 (예: Colab nano)
ros2 run amr_ab_control webcam_event --model $(ros2 pkg prefix amr_ab_control)/share/amr_ab_control/models/nano_colab.pt
```

주요 파라미터(`amr_ab_controller`): `min_confidence`(0.7), `standoff_m`(0.80), `search_x`/`search_y`(A), `detect_on_way`(false), `scan_speed`(0.5), `track_gain`(1.5), `track_deadband_deg`(6), `lost_turn_after_s`(1.0), `replan_distance_m`(0.45)

### 영상 변환

```bash
ros2 run amr_ab_control bag_to_mp4 ~/rokey_ws/bags/test_YYYYMMDD_HHMMSS_메모
```

bag 저장 위치: `~/rokey_ws/bags/`

## 시험

```bash
cd ~/rokey_ws/src/amr_ab_control && python3 -m pytest -q test
ros2 run amr_ab_control ab_nav_only --preview /tmp/ab_preview.png   # 지도 경로 프리뷰 (로봇은 움직이지 않음)
```

## 남은 과제 (2026-11-12 실기 기준)

1. ~~정지 거리 보정~~: `standoff_m`을 0.80에서 0.55로 낮췄습니다(11/13). 실기에서 실제로 약 0.80 m에서 멈추는지 확인이 필요합니다.
2. 지도 밖이나 미지 영역에 있는 대상은 무시하기
3. 로봇이 회전하는 동안 대상 좌표가 흔들리는 문제 확인
4. 도크에서 출발하는 Undock 시연. 도크 출발 흐름은 11/13에 고쳤고 시뮬레이션으로 확인했습니다. 실기 확인이 남았습니다.

## 주의

- 안전 구역 검사는 접근 목표점만 확인합니다. Nav2가 실제로 지나가는 경로까지는 보장하지 않습니다.
- RGB와 depth 정렬은 robot3의 OAK-D(704x704, RGB 정렬 depth) 기준으로 확인했습니다. 다른 로봇에서 쓰면 다시 확인해야 합니다.
