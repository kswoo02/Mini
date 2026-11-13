# Mini Project: TurtleBot4 차량 탐색·접근 AMR

TurtleBot4(ROS 2 Jazzy)가 웹캠으로 START 신호를 받으면 도크에서 나와 지정 위치로 이동하고, OAK-D 카메라와 YOLOv8로 `car`를 찾아 0.8 m 앞까지 접근한 뒤 따라가는 프로젝트입니다. `dummy`는 접근 대상에서 제외합니다.

프로젝트 노트(Notion): https://app.notion.com/p/C-3-PROJECT-414ce25f3edf809c82a2fd13798d6a38?source=copy_link

## 폴더 구성

```
Mini-main/
├── amr_ab_control/          ROS 2 패키지 (메인 코드, 최신본)
│   ├── amr_ab_control/      노드 소스 (컨트롤러, 웹캠 이벤트, 카메라 뷰 등)
│   ├── maps/                순찰 지도 (minimap.yaml / .pgm)
│   ├── models/              YOLO 가중치 (oakd_best1, webcam_car_dummy_aug, nano_colab)
│   ├── scripts/             record_test.sh (bag 녹화)
│   ├── test/                오프라인 유닛 테스트
│   └── README_KO.md         패키지 상세 설명 (실행 순서, 파라미터)
├── yolo/                    단독 실행용 YOLO + depth 거리 측정 노드
│   ├── yolo_node.py         초기 버전
│   ├── yolo_detector.py     개선 버전 (docs/yolo_node_개선점.md 반영)
│   └── models/best.pt
├── dataset/                 Roboflow YOLOv8 데이터셋 (car / dummy)
├── docs/                    시스템 구성도, 흐름도(.drawio), 코드 개선 노트
└── archive/                 이전 버전 압축본 (B/C 순찰 버전 등)
```

## 시스템 흐름

1. `webcam_event`: 웹캠에서 car를 감지하면 `/robot3/amr/event`에 `START` 발행
2. `amr_ab_controller`: Undock → AMCL 위치 확인 → 탐색 위치 A로 이동
3. A에서 제자리 회전하며 OAK-D + YOLO로 car 탐색
4. 감지하면 depth로 거리를 구해 0.8 m 앞까지 Nav2로 접근
5. 차를 화면 가운데에 유지하며 추적. 놓치면 사라진 방향으로 회전, 끝내 못 찾으면 A로 돌아가 다시 탐색

구성도와 흐름도는 `docs/AMR.drawio`, `docs/AMR_flowchart.drawio`([draw.io](https://app.diagrams.net)에서 열기)에 있습니다.

## 요구 사항

- Ubuntu 24.04 + ROS 2 Jazzy, TurtleBot4 패키지 (`turtlebot4_navigation`, `turtlebot4_viz`)
- Python 패키지: `pip install -r requirements.txt` (ultralytics, lap)

## 빌드

저장소를 워크스페이스 `src` 안에 그대로 clone하면 됩니다. colcon이 하위 폴더의 `amr_ab_control/package.xml`을 자동으로 찾습니다.

```bash
mkdir -p ~/rokey_ws/src && cd ~/rokey_ws/src
git clone <이 저장소 URL> mini
pip install -r mini/requirements.txt
cd ~/rokey_ws
rosdep install --from-paths src --ignore-src -y   # ROS 의존성 (선택)
colcon build --symlink-install --packages-select amr_ab_control
source install/setup.bash
```

지도와 모델은 패키지에 포함되어 함께 설치되므로 경로를 따로 설정할 필요가 없습니다.

### 다른 로봇에서 실행할 때

기본 네임스페이스는 `/robot3`입니다. 다른 로봇이면 노드마다 `--namespace`를 넘깁니다.

```bash
ros2 run amr_ab_control amr_ab_controller --execute --namespace /robot1
ros2 run amr_ab_control webcam_event --namespace /robot1
ros2 run amr_ab_control turtle_cam_view --namespace /robot1
NS=/robot1 ros2 run amr_ab_control record_test.sh
```

## 실행

터미널마다 `source ~/rokey_ws/install/setup.bash` 후 순서대로 실행합니다.

```bash
# 1. Localization
ros2 launch turtlebot4_navigation localization.launch.py namespace:=/robot3 \
  map:=$(ros2 pkg prefix amr_ab_control)/share/amr_ab_control/maps/minimap.yaml
# 2. Nav2
ros2 launch turtlebot4_navigation nav2.launch.py namespace:=/robot3
# 3. RViz (2D Pose Estimate 지정)
ros2 launch turtlebot4_viz view_navigation.launch.py namespace:=/robot3
# 4. 메인 컨트롤러 (--execute 없으면 dry-run)
ros2 run amr_ab_control amr_ab_controller --execute
# 5. (선택) 카메라 화면
ros2 run amr_ab_control turtle_cam_view
# 6. 웹캠 START 감지
ros2 run amr_ab_control webcam_event
```

웹캠 없이 시작/정지만 하려면:

```bash
ros2 topic pub --once /robot3/amr/event std_msgs/msg/String "{data: START}"
ros2 topic pub --once /robot3/amr/event std_msgs/msg/String "{data: STOP}"   # 작업 취소일 뿐, 비상정지 아님
```

파라미터, 탐색 위치 변경, bag→mp4 변환 등은 [`amr_ab_control/README_KO.md`](amr_ab_control/README_KO.md)를 참고하세요.

### 단독 YOLO 노드

```bash
python3 yolo/yolo_detector.py                         # 기본 모델: yolo/models/best.pt
YOLO_MODEL=/경로/모델.pt python3 yolo/yolo_detector.py   # 다른 모델
```

구독 토픽은 `/robot3/oakd/...`로 고정되어 있습니다. 다른 로봇이면 파일 위쪽의 `RGB_TOPIC`, `DEPTH_TOPIC`을 바꿔 주세요.

## 테스트

```bash
cd amr_ab_control && python3 -m pytest -q test   # 11개 오프라인 테스트
```

## 남은 과제

- 실기에서 정지 거리(`standoff_m`) 확인
- 지도 밖이나 미지 영역의 대상 무시
- RGB/depth 시간 동기화 (`message_filters.ApproximateTimeSynchronizer`)
- 도크 출발(Undock) 실기 시연
- 구역 통제용 Nav2 Keepout Filter (미구현)
