# AMR System Requirements 및 Test Plan (시험용 초안)

| ID | Requirement | 구현 파일 / 항목 | 확인 방법 | 현재 상태 |
|---|---|---|---|---|
| SR-01 | START 이벤트 수신 후 동작 시작 | amr_bc_controller.py 이벤트 subscriber | /robot3/amr/event 발행 후 상태 확인 | 작성, 실기 미검증 |
| SR-02 | Dock 상태 확인 후 Undock 액션 | DockStatus + Undock | 도킹 상태/언도킹 결과 기록 | 작성, 실기 미검증 |
| SR-03 | 실제 출발 좌표로 Initial Pose 설정 | /initialpose + AMCL/TF | RViz pose/map TF 확인 | 작성, 실기 미검증 |
| SR-04 | B(-2.15,1.40), C(-1.00,-0.50) 이동 | Nav2 NavigateToPose | RViz 경로와 주행 결과 | 맵 사전 검사 통과, 실기 미검증 |
| SR-05 | car 대상 정보 획득 | YOLO track, CameraInfo, Depth, TF2 | /robot3/amr/target_info 확인 | 작성, 실기 미검증 |
| SR-06 | dummy는 접근 제외 | eligible_target_class | 단위 시험 및 실제 YOLO 시험 | 오프라인 시험 통과 |
| SR-07 | 목표 안전거리 0.8m 접근 | find_approach_goal + Nav2 | 실기에서 거리계측/정지 오차 기록 | 기하학 시험 통과, 실기 미검증 |
| SR-08 | 유효하지 않은 깊이/맵 밖 목표 거부 | robust_depth_m + MapGuard | 오프라인 테스트, 시뮬레이션 | 오프라인 시험 통과 |
| SR-09 | B-C 외 구역 엄격 통제 | Nav2 Keepout Filter 필요 | footprint를 포함한 경로 점검 | **미구현** |
| SR-10 | STOP 이벤트로 Nav2 취소 | amr_bc_controller.py | 실제 이동 중 STOP 및 상태 확인 | 작성, 실기 미검증 |

## 단계별 실험 순서

1. PC에서 `python3 -m unittest discover -s tests -v` (11개 로직 시험)
2. `python3 bc_nav_only.py`로 맵 B/C 경유지 검사
3. RViz에서 AMCL localization/map->base_link 및 global/local costmap 확인
4. 충분한 안전거리에서 Nav2 단일 B 목표 주행, 그다음 B->C
5. DockStatus/Undock 액션과 실제 dock 후 초기 위치 설정 실험
6. OAK-D RGB/Depth 시간 동기화 및 **공간 정렬**, CameraInfo/TF 검증
7. car/dummy 정적 물체 오검출률·거리 오차 측정
8. 접근 기능을 켜기 전 실제 물체 안전거리와 Nav2 footprint/costmap/Keepout 검사
9. 이동하는 car 접근·재계획 시험 (안전관리자 확보)
10. 실험 로그(시작/종료 시간, 실패 이유, 목표 오차, 최소 장애물 거리)로 요구사항 갱신
