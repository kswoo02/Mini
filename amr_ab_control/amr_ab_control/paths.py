"""패키지 리소스(지도, 모델) 경로.

설치된 share/amr_ab_control 를 우선 사용하고, 없으면 소스 트리(src/amr_ab_control) 를 사용.
"""
from pathlib import Path

PKG = 'amr_ab_control'


def share_dir() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory(PKG))
    except Exception:
        return Path(__file__).resolve().parent.parent  # 소스 트리에서 직접 실행/테스트할 때


MAP_YAML = share_dir() / 'maps' / 'minimap.yaml'
OAKD_MODEL = share_dir() / 'models' / 'oakd_best1.pt'              # 로봇 OAK-D 감지 (turtle_car_dummy_v8n best1)
WEBCAM_MODEL = share_dir() / 'models' / 'webcam_car_dummy_aug.pt'  # 웹캠 START 감지 (car_dummy_aug best)
NANO_COLAB_MODEL = share_dir() / 'models' / 'nano_colab.pt'        # Colab 학습 nano (웹캠용 후보)
