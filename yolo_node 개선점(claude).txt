1. 콜백 기아 — 가장 큰 문제 (294~301줄)

rclpy.spin_once는 준비된 작업을 한 개만 처리하고 돌아옵니다. 그런데 구독이 RGB와 depth 둘이라 초당 20개의 콜백이 들어옵니다. 계산해보면:

추론 시간	초당 루프	필요	결과
200ms (CPU)	4.8	20	1/4만 처리
100ms	9.1	20	밀림
25ms (GPU)	28.6	20	따라감

device='cpu'로 고정돼 있으니(200줄) CPU 판입니다. 큐가 계속 밀리고, QoS depth=5라 오래된 것부터 버려집니다. 살아남은 프레임도 몇 초 전 것이라 로봇이 움직이는 중이면 박스와 거리가 서로 다른 시점의 장면이 됩니다.

수정본은 spin_once(0.05)로 기다린 뒤 spin_once(0.0)을 여러 번 돌려 큐를 비우고, 새 프레임이 있을 때만 detect()를 부릅니다.

2. 중심 5×5의 함정 (162~169줄)

박스 중심 한 점은 차체 틈이나 유리로 배경이 비칠 수 있습니다. 1.2m 차량 박스 중앙에 4m 배경이 보이는 상황을 만들어 실험했습니다:

중심 5x5 중앙값      : 4.00 m   ← 배경을 집었다
박스 중앙 40% 중앙값 : 1.20 m   ← 차체를 집었다

수정본은 CENTER_FRACTION = 0.4로 박스 중앙 40% 영역을 봅니다.

3. 창 먹통 (192~193줄)
python
if self.frame is None or self.depth_image is None:
    return

여기서 return하면 cv2.waitKey(276줄)에 도달하지 못해 창이 응답 불가가 됩니다. 어제 진단한 그 문제가 그대로 있습니다. 게다가 depth가 안 오면 YOLO 검출조차 안 보입니다 — 지금 depth가 83 Mbps라 자주 끊기는데, 그때마다 화면이 통째로 죽습니다.

수정본은 RGB만 있어도 검출을 띄우고, 프레임이 없을 때도 show()로 창을 갱신합니다. HUD에 rgb 0.1s depth 2.4s처럼 마지막 수신 후 경과 시간을 띄워서 어느 쪽이 끊겼는지 바로 보입니다.

4. 죽은 코드 (93~101줄)
python
depth_image = cv2.imdecode(...) if msg.encoding in ['jpeg','jpg'] else None

sensor_msgs/Image를 구독하고 있으므로(52줄) encoding은 절대 'jpeg'가 되지 않습니다. 'jpeg'는 CompressedImage의 format 필드 값입니다. 이 분기는 한 번도 실행되지 않습니다.

5. msg.step 무시 (110, 116줄)

.reshape(msg.height, msg.width)는 행 패딩이 있으면 깨집니다. 968 step에 960 width인 메시지로 테스트해보니 수정본은 올바르게 잘라냈습니다. OAK-D는 보통 패딩이 없지만 잠복 버그입니다. mono16도 빠져 있어서 추가했습니다.

6. 그 밖에
device='cpu' 고정(200줄) — GPU가 있으면 10~50배 손해입니다. DEVICE = None로 자동 선택하게 바꿨습니다
하드코딩된 경로(23줄) — 상수로 올렸습니다. 어제 이야기한 ROS 패키지로 올리실 거면 --pkg로 찾는 방식이 낫습니다
raise KeyboardInterrupt(279줄) — 동작은 하지만 should_exit 플래그가 의도가 분명합니다
Z vs 직선거리 — depth는 광축 방향 Z 성분입니다. 화면 가장자리 물체는 실제 직선거리가 10% 정도 더 멉니다. 주행 제어용이라면 Z가 오히려 맞으니 그대로 뒀습니다
아직 안 고친 것 — 시간 동기화

RGB와 depth를 각각 "가장 최근 것"으로 쓰고 있어서, 둘의 촬영 시각이 다를 수 있습니다. 로봇이 정지 상태면 문제없지만 주행 중이면 거리값이 어긋납니다. 제대로 하려면 message_filters.ApproximateTimeSynchronizer로 두 토픽을 짝지어야 합니다. 지금 단계에서 필요하실지 모르겠어서 남겨뒀는데, 주행 제어에 쓰실 거면 그때 붙이시면 됩니다.
