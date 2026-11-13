#!/usr/bin/env bash
# 테스트 1회분 rosbag 녹화 (영상 + 상태 + 위치 + 경로).  Ctrl+C 로 종료/저장.
#   ros2 run amr_ab_control record_test.sh [메모]      예) ros2 run amr_ab_control record_test.sh dummy_test
# 저장: ${BAG_DIR:-~/rokey_ws/bags}/test_<날짜시간>_<메모>/
# 다른 로봇/경로:  NS=/robot1 BAG_DIR=~/bags ros2 run amr_ab_control record_test.sh
#
# 영상은 이 PC 에서 만든 로컬 토픽(YOLO 박스가 그려진 화면)만 녹화 -> WiFi 추가 부담 없음
#   /robot3/amr/camera_view/compressed  : 터틀봇 OAK-D + YOLO (amr_ab_controller.py)
#   /robot3/amr/webcam_view/compressed  : 외장 웹캠 + YOLO   (webcam_event.py)
NS=${NS:-/robot3}
BAG_DIR=${BAG_DIR:-$HOME/rokey_ws/bags}
NAME="test_$(date +%Y%m%d_%H%M%S)${1:+_$1}"
OUT="$BAG_DIR/$NAME"
mkdir -p "$BAG_DIR"
echo "녹화 시작 -> $OUT   (Ctrl+C 로 종료)"
exec ros2 bag record -s mcap -o "$OUT" \
  $NS/amr/camera_view/compressed \
  $NS/amr/webcam_view/compressed \
  $NS/amr/status $NS/amr/event $NS/amr/target_info \
  $NS/amcl_pose $NS/tf $NS/tf_static $NS/map \
  $NS/plan $NS/cmd_vel $NS/odom $NS/scan \
  $NS/dock_status $NS/battery_state \
  /rosout
