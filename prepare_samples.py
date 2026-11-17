"""샘플 사진 준비 — 인터넷에서 받지 않고, 이미 설치된 ultralytics 패키지 안의 예제 2장을 복사한다.

zidane.jpg 는 실제 인물이 나온 보도 사진이라 저장소에는 올리지 않았다(저작권·초상권 배려).
이 스크립트를 실행하면 내 컴퓨터의 data/samples 에 복사되어 검증을 똑같이 재현할 수 있다.
"""
import shutil
from pathlib import Path

import ultralytics

src = Path(ultralytics.__file__).parent / "assets"
dst = Path(__file__).resolve().parent / "data" / "samples"
dst.mkdir(parents=True, exist_ok=True)
for name in ["bus.jpg", "zidane.jpg"]:
    shutil.copy2(src / name, dst / name)
    print("복사:", name)
