"""
YOLO 보완 PoC — 기존 YOLO 한 번으로 끝내지 않고, 약점마다 다른 AI를 덧붙인다.

흐름 (사진 1장):
  ① 기준선   yolo26n / yolo26n-seg (640)          = 기존 서비스가 하던 일
  ② 큰 YOLO  yolo26m (640)                        -> 기준선이 놓친 물체 후보
  ③ 글자 지목 YOLOE + 글자 목록(prompts)            -> 이름을 지목해서 다시 찾기 (clip 패키지가 있을 때만)
  ④ SAM      기준선 박스를 SAM 에 넘겨 윤곽 다시 따기 -> 마스크 구멍 메우기
  ⑤ 검토 목록 ①과 ②③④가 서로 다른 곳만 잘라서 report.html 로 모음 -> 사람이 볼 곳을 줄임

실행 예:
    python poc.py                         # data/samples 안의 모든 사진
    python poc.py --images data/samples/bus.jpg
    python poc.py --prompts "cell phone" "tie"

안전: 실행 중 인터넷에서 무언가를 몰래 설치하지 않도록 YOLO_AUTOINSTALL 을 끈다.
"""
from __future__ import annotations

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # 숨은 자동 설치 금지
os.environ.setdefault("YOLO_OFFLINE", "True")        # 실행 중 인터넷 접속 안 함 (모델은 download_models.py 로 미리 받음)

import argparse
import html
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SAMPLES = ROOT / "data" / "samples"
DEFAULT_PROMPTS = ["person", "bus", "cell phone", "tie", "handbag", "backpack", "cup", "bottle"]

BASE_CONF = 0.25      # 기존 서비스와 같은 임계값
HELPER_CONF = 0.10    # 보조 모델은 낮게: 어차피 사람이 확인할 "후보"이므로 넓게 모은다
IMGSZ = 640

COLORS = {"baseline": (255, 160, 0), "bigger": (0, 200, 255), "text": (255, 0, 255), "sam": (0, 220, 0),
          "gt": (255, 255, 255)}


# ---------- 파일 (한글 경로 안전) ----------
def imread(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지를 열 수 없습니다: {path}")
    return img


def imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise ValueError(f"이미지를 저장할 수 없습니다: {path}")
    buf.tofile(str(path))


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---------- 모델 ----------
class Models:
    """필요할 때 한 번만 읽는다."""

    def __init__(self):
        self._cache = {}
        self.text_ready = False
        self.text_skip_reason = ""

    def yolo(self, name: str):
        if name not in self._cache:
            from ultralytics import YOLO
            path = MODELS / name
            if not path.exists():
                raise FileNotFoundError(f"모델이 없습니다: models/{name}  ->  먼저 python download_models.py 를 실행하세요")
            self._cache[name] = YOLO(str(path))
        return self._cache[name]

    def sam(self):
        if "sam" not in self._cache:
            from ultralytics import SAM
            self._cache["sam"] = SAM(str(MODELS / "sam2.1_t.pt"))
        return self._cache["sam"]

    def yoloe(self, prompts: list[str]):
        """글자 지목 모델. clip 패키지나 글자 부품이 없으면 None (그 단계는 건너뜀으로 기록)."""
        if "yoloe" in self._cache:
            return self._cache["yoloe"]
        model = None
        if importlib.util.find_spec("clip") is None:
            self.text_skip_reason = "clip 패키지가 설치되지 않음 (README의 선택 설치 참고)"
        elif not (MODELS / "mobileclip2_b.ts").exists() or not (MODELS / "yoloe-26s-seg.pt").exists():
            self.text_skip_reason = "YOLOE 모델 또는 글자 부품 파일이 없음 (python download_models.py --with-text-encoder)"
        else:
            from ultralytics import YOLOE
            cwd = os.getcwd()
            os.chdir(MODELS)                      # 글자 부품(mobileclip2_b.ts)을 이 폴더에서 찾게 함
            try:
                model = YOLOE("yoloe-26s-seg.pt")
                model.set_classes(prompts, model.get_text_pe(prompts))
                self.text_ready = True
            finally:
                os.chdir(cwd)
        self._cache["yoloe"] = model
        return model


def detect(model, img, conf: float, imgsz: int = IMGSZ, source: str = "") -> tuple[list[dict], float]:
    t = time.perf_counter()
    r = model.predict(img, conf=conf, imgsz=imgsz, device="cpu", verbose=False)[0]
    secs = time.perf_counter() - t
    dets = []
    for b in r.boxes:
        cid = int(b.cls.item())
        dets.append({"class_name": r.names[cid], "score": round(float(b.conf.item()), 3),
                     "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()], "source": source})
    return dets, secs


def segment_yolo(model, img, conf: float) -> tuple[list[dict], list[np.ndarray], float]:
    t = time.perf_counter()
    r = model.predict(img, conf=conf, imgsz=IMGSZ, device="cpu", verbose=False, retina_masks=True)[0]
    secs = time.perf_counter() - t
    insts, masks = [], []
    if r.masks is not None:
        data = r.masks.data.cpu().numpy() > 0.5
        for i, b in enumerate(r.boxes):
            cid = int(b.cls.item())
            insts.append({"class_name": r.names[cid], "score": round(float(b.conf.item()), 3),
                          "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()]})
            masks.append(data[i])
    return insts, masks, secs


def segment_sam(sam, img, boxes: list[list[float]]) -> tuple[list[np.ndarray], float]:
    if not boxes:
        return [], 0.0
    t = time.perf_counter()
    r = sam.predict(img, bboxes=boxes, device="cpu", verbose=False)[0]
    secs = time.perf_counter() - t
    data = r.masks.data.cpu().numpy() > 0.5
    return [data[i] for i in range(len(boxes))], secs


# ---------- 그리기 ----------
def draw(img, dets, color, thick=2, tag=True):
    out = img.copy()
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        if tag:
            text = f'{d["class_name"]} {d.get("score", 0):.2f}'
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yt = max(0, y1 - th - base - 2)
            cv2.rectangle(out, (x1, yt), (x1 + tw + 4, yt + th + base + 2), color, -1)
            cv2.putText(out, text, (x1 + 2, yt + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def tint(img, mask, color, alpha=0.5):
    out = img.copy()
    out[mask] = (img[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def crop_around(img, box, pad=0.6, min_side=220):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    side_x, side_y = max(bw * (1 + 2 * pad), min_side), max(bh * (1 + 2 * pad), min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - side_x / 2)), int(max(0, cy - side_y / 2))
    c, d = int(min(w, cx + side_x / 2)), int(min(h, cy + side_y / 2))
    crop = img[b:d, a:c]
    scale = 320 / max(crop.shape[:2])
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


# ---------- 검토 후보 만들기 ----------
def unmatched(cands: list[dict], refs: list[dict], thr=0.5) -> list[dict]:
    """refs 에 같은 클래스·같은 자리(IoU>=thr)가 없는 cands."""
    return [c for c in cands
            if not any(r["class_name"] == c["class_name"] and iou(r["xyxy"], c["xyxy"]) >= thr for r in refs)]


def dedupe(cands: list[dict], thr=0.6) -> list[dict]:
    """여러 보조 모델이 같은 곳을 가리키면 하나로 합치고, 누가 가리켰는지만 모은다."""
    out: list[dict] = []
    for c in sorted(cands, key=lambda d: -d["score"]):
        for o in out:
            if o["class_name"] == c["class_name"] and iou(o["xyxy"], c["xyxy"]) >= thr:
                o["source"] = "+".join(sorted(set(o["source"].split("+")) | {c["source"]}))
                break
        else:
            out.append(dict(c))
    return out


def process(img_path: Path, models: Models, prompts: list[str], out_root: Path, use_sam=True) -> dict:
    img = imread(img_path)
    h, w = img.shape[:2]
    out = out_root / img_path.stem
    out.mkdir(parents=True, exist_ok=True)
    timing = {}

    # ① 기준선 (기존 방식)
    base, timing["1_baseline_det"] = detect(models.yolo("yolo26n.pt"), img, BASE_CONF, source="baseline")
    seg_insts, seg_masks, timing["1_baseline_seg"] = segment_yolo(models.yolo("yolo26n-seg.pt"), img, BASE_CONF)

    # ② 큰 YOLO
    bigger, timing["2_bigger_yolo"] = detect(models.yolo("yolo26m.pt"), img, HELPER_CONF, source="bigger")

    # ③ 글자 지목 (선택)
    text_dets: list[dict] = []
    yoloe = models.yoloe(prompts)
    if yoloe is not None:
        text_dets, timing["3_text_yoloe"] = detect(yoloe, img, HELPER_CONF, source="text")

    # ④ SAM: 기준선 박스를 그대로 넘겨 윤곽만 다시 딴다
    sam_masks: list[np.ndarray] = []
    mask_rows = []
    if use_sam:
        sam_masks, timing["4_sam"] = segment_sam(models.sam(), img, [d["xyxy"] for d in base])
        for d, sm in zip(base, sam_masks):
            # 같은 물체의 기존 seg 마스크 찾기 (seg 모델과 det 모델은 번호가 다르므로 좌표로 맞춤)
            best, best_iou = None, 0.0
            for k, s in enumerate(seg_insts):
                v = iou(s["xyxy"], d["xyxy"])
                if s["class_name"] == d["class_name"] and v > best_iou:
                    best, best_iou = k, v
            ym = seg_masks[best] if best is not None and best_iou >= 0.5 else np.zeros((h, w), bool)
            added, removed = sm & ~ym, ym & ~sm
            mask_rows.append({"class_name": d["class_name"], "xyxy": d["xyxy"],
                              "yolo_mask_px": int(ym.sum()), "sam_mask_px": int(sm.sum()),
                              "added_by_sam_px": int(added.sum()), "removed_by_sam_px": int(removed.sum()),
                              "_added": added})

    # ⑤ 검토 후보
    cands = []
    helpers = dedupe(unmatched(bigger, base) + unmatched(text_dets, base))
    for c in helpers:
        cands.append({"kind": "missed?", "why": f"보조 모델({c['source']})만 찾음 - 기준선이 놓쳤을 수 있음", **c})
    if bigger:
        for c in unmatched(base, bigger):
            cands.append({"kind": "false?", "why": "기준선만 찾음 - 큰 모델은 못 봄, 오탐일 수 있음", **c})
    for m in mask_rows:
        added = m.pop("_added")
        n, lab, stats, _ = cv2.connectedComponentsWithStats(added.astype(np.uint8), 8)
        for k in sorted(range(1, n), key=lambda k: -stats[k, cv2.CC_STAT_AREA])[:2]:
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area >= max(400, 0.01 * max(1, m["sam_mask_px"])):
                x, y, bw, bh = [int(stats[k, j]) for j in range(4)]
                cands.append({"kind": "mask-gap?", "class_name": m["class_name"], "score": 0.0, "source": "sam",
                              "why": f"SAM이 {area:,}px 를 더 칠함 - 기존 마스크의 구멍일 수 있음",
                              "xyxy": [x, y, x + bw, y + bh], "_region": lab == k})

    # ---- 저장: 그림 ----
    imwrite(out / "original.jpg", img)
    base_img = draw(img, base, COLORS["baseline"])
    imwrite(out / "1_baseline.jpg", base_img)
    helper_img = draw(draw(base_img, [c for c in helpers if "bigger" in c["source"]], COLORS["bigger"], 3),
                      [c for c in helpers if c["source"] == "text"], COLORS["text"], 3)
    imwrite(out / "2_3_helpers.jpg", helper_img)
    if seg_masks:
        ym_all = np.any(np.stack(seg_masks), axis=0)
        imwrite(out / "1_baseline_mask.jpg", tint(img, ym_all, COLORS["baseline"]))
    if sam_masks:
        sm_all = np.any(np.stack(sam_masks), axis=0)
        imwrite(out / "4_sam_mask.jpg", tint(img, sm_all, COLORS["sam"]))

    for i, c in enumerate(cands):
        view = img
        if "_region" in c:
            view = tint(img, c.pop("_region"), COLORS["sam"], 0.55)
        color = {"missed?": COLORS["bigger"], "false?": (0, 0, 255), "mask-gap?": COLORS["sam"]}[c["kind"]]
        marked = view.copy()
        x1, y1, x2, y2 = [int(round(v)) for v in c["xyxy"]]
        cv2.rectangle(marked, (x1, y1), (x2, y2), color, 2)
        c["id"] = i
        c["crop"] = f"candidates/c{i:02d}.jpg"
        imwrite(out / c["crop"], crop_around(marked, c["xyxy"]))

    result = {
        "image": img_path.name, "width": w, "height": h,
        "settings": {"baseline_conf": BASE_CONF, "helper_conf": HELPER_CONF, "imgsz": IMGSZ, "device": "cpu"},
        "baseline": base, "bigger_yolo": bigger,
        "text_prompts": prompts if yoloe is not None else None,
        "text_stage": "실행함" if yoloe is not None else f"건너뜀: {models.text_skip_reason}",
        "text_yoloe": text_dets, "masks": mask_rows, "review_candidates": cands,
        "timing_sec": {k: round(v, 2) for k, v in timing.items()}, "total_sec": round(sum(timing.values()), 2),
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out, result)
    return result


def write_report(out: Path, r: dict) -> None:
    e = html.escape
    kinds = {"missed?": "놓쳤을 수 있음", "false?": "오탐일 수 있음", "mask-gap?": "마스크 구멍일 수 있음"}
    rows = "".join(
        f'<tr><td>{c["id"]}</td><td><b>{kinds[c["kind"]]}</b><br>{e(c["class_name"])}'
        f'{" " + format(c["score"], ".2f") if c["score"] else ""}</td>'
        f'<td>{e(c["why"])}<br><small>좌표 {[int(v) for v in c["xyxy"]]}</small></td>'
        f'<td><img src="{c["crop"]}"></td><td class="chk">□ 맞음 &nbsp; □ 아님</td></tr>'
        for c in r["review_candidates"])
    mask_rows = "".join(
        f'<tr><td>{e(m["class_name"])}</td><td>{m["yolo_mask_px"]:,}</td><td>{m["sam_mask_px"]:,}</td>'
        f'<td>+{m["added_by_sam_px"]:,}</td><td>-{m["removed_by_sam_px"]:,}</td></tr>' for m in r["masks"])
    figs = [("original.jpg", "원본"), ("1_baseline.jpg", "① 기준선 (기존 방식)"),
            ("2_3_helpers.jpg", "②③ 보조 모델이 추가로 가리킨 곳 (굵은 박스)"),
            ("1_baseline_mask.jpg", "① 기존 마스크"), ("4_sam_mask.jpg", "④ SAM 마스크")]
    fig_html = "".join(f'<figure><img src="{f}"><figcaption>{t}</figcaption></figure>' for f, t in figs if (out / f).exists())
    timing = " · ".join(f"{k} {v}s" for k, v in r["timing_sec"].items())
    page = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>검토 목록 - {e(r["image"])}</title>
<style>body{{font-family:'Malgun Gothic',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
figure{{display:inline-block;margin:6px;vertical-align:top}}figure img{{height:300px}}figcaption{{font-size:13px;text-align:center}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{border:1px solid #ccc;padding:6px 8px;vertical-align:top}}
th{{background:#f3f3f3}}td img{{max-height:200px}}.chk{{white-space:nowrap}}small{{color:#666}}</style>
<h1>검토 목록 — {e(r["image"])}</h1>
<p>사진 전체를 훑는 대신 <b>아래 {len(r["review_candidates"])}곳만</b> 확인하면 됩니다. 모델끼리 결과가 다른 곳만 모았습니다.
AI의 제안일 뿐이므로 <b>사람이 맞음/아님을 표시</b>해야 합니다.</p>
<p><small>처리 시간(CPU): 합계 {r["total_sec"]}s — {timing}<br>글자 지목 단계: {e(r["text_stage"])}</small></p>
{fig_html}
<h2>확인할 곳</h2>
<table><tr><th>#</th><th>종류</th><th>이유</th><th>잘라 본 그림</th><th>사람 확인</th></tr>{rows or '<tr><td colspan=5>모델끼리 다른 곳이 없습니다.</td></tr>'}</table>
<h2>마스크 비교 (기존 YOLO-seg vs SAM)</h2>
<table><tr><th>물체</th><th>기존 마스크(px)</th><th>SAM 마스크(px)</th><th>SAM이 더 칠함</th><th>SAM이 뺌</th></tr>{mask_rows}</table>
</html>"""
    (out / "report.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="YOLO 보완 PoC")
    ap.add_argument("--images", nargs="*", help="사진 경로 (생략하면 data/samples 전체)")
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS, help="글자로 지목할 대상 (영어)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "poc"))
    ap.add_argument("--no-sam", action="store_true")
    args = ap.parse_args()

    paths = [Path(p) for p in args.images] if args.images else sorted(
        p for p in SAMPLES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    models = Models()
    summary = []
    for p in paths:
        if not p.exists():
            print(f"[건너뜀] 파일이 없습니다: {p}")
            continue
        r = process(p, models, args.prompts, Path(args.out), use_sam=not args.no_sam)
        kinds = [c["kind"] for c in r["review_candidates"]]
        print(f"[{p.name}] 기준선 {len(r['baseline'])}개 / 검토 후보 {len(kinds)}곳 "
              f"(놓침? {kinds.count('missed?')}, 오탐? {kinds.count('false?')}, 마스크? {kinds.count('mask-gap?')}) "
              f"/ {r['total_sec']}s / 글자 지목: {r['text_stage']}")
        summary.append({"image": p.name, "baseline": len(r["baseline"]), "candidates": len(kinds),
                        "total_sec": r["total_sec"]})
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {Path(args.out)}  (사진마다 report.html 을 열어 보세요)")


if __name__ == "__main__":
    main()
