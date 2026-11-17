"""
모델 파일을 '공식 출처에서 직접' 받고, 지문(SHA256)을 공식 값과 대조하는 스크립트.

- 출처: Ultralytics 공식 자료 저장소 (github.com/ultralytics/assets, 릴리스 v8.4.0)
- 아래 지문은 2026-11-17 에 공식 릴리스 목록에서 조회한 값이다.
- 지문이 다르면 파일을 지우고 멈춘다. (.pt 파일은 여는 순간 코드가 실행될 수 있는 형식이라
  반드시 대조한 뒤에만 연다.)

실행:
    python download_models.py            # 기본 묶음
    python download_models.py --with-text-encoder   # 글자 해석 부품(242MB)까지
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"

# 파일 이름: (공식 SHA256, 쓰임새)
PINNED = {
    "yolo26n.pt": ("9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef", "기준선 detection (기존 서비스와 같은 모델)"),
    "yolo26n-seg.pt": ("361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f", "기준선 segmentation (기존 서비스와 같은 모델)"),
    "yolo26s.pt": ("646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b", "더 큰 YOLO (small)"),
    "yolo26m.pt": ("401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7", "더 큰 YOLO (medium)"),
    "sam2.1_t.pt": ("3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3", "SAM 2.1 tiny - 마스크 다시 따기"),
    "yoloe-26s-seg.pt": ("48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654", "YOLOE - 글자로 지목해서 찾기"),
}
TEXT_ENCODER = {
    "mobileclip2_b.ts": ("35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f", "YOLOE 글자 해석 부품 (MobileCLIP2 텍스트 인코더)"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, expected: str) -> dict:
    dst = MODELS / name
    if dst.exists() and sha256(dst) == expected:
        print(f"[이미 있음·지문 일치] {name}")
        return {"file": name, "bytes": dst.stat().st_size, "sha256": expected, "status": "이미 있음, 지문 일치"}
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[받는 중] {name} ...", flush=True)
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "model-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    got = sha256(tmp)
    if got != expected:
        tmp.unlink()
        print(f"[중단] {name}: 지문이 공식 값과 다릅니다. 파일을 지웠습니다.\n  기대 {expected}\n  실제 {got}")
        sys.exit(2)
    tmp.replace(dst)
    print(f"[완료·지문 일치] {name} ({dst.stat().st_size / 1e6:.1f} MB)")
    return {"file": name, "bytes": dst.stat().st_size, "sha256": got, "status": "내려받음, 지문 일치"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-text-encoder", action="store_true", help="YOLOE 글자 해석 부품(242MB)도 받기")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    todo = dict(PINNED)
    if args.with_text_encoder:
        todo.update(TEXT_ENCODER)
    record = []
    for name, (digest, use) in todo.items():
        row = fetch(name, digest)
        row["use"] = use
        row["source"] = BASE + name
        record.append(row)
    # 기존 기록에 합치기 (파일 이름 기준)
    out = MODELS / "MODELS.json"
    old = {r["file"]: r for r in json.loads(out.read_text(encoding="utf-8"))} if out.exists() else {}
    old.update({r["file"]: r for r in record})
    out.write_text(json.dumps(list(old.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"기록: models/MODELS.json ({len(old)}개)")


if __name__ == "__main__":
    main()
