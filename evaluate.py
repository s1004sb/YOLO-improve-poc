"""
개선 효과 검증 — 기존 방식(yolo26n, 640, conf 0.25)과 보완 방식을 정답 목록으로 나란히 비교한다.

  A. 탐지: 정답 물체 중 몇 개를 찾았나 / 잘못 찾은 것은 몇 개인가 (방식별)
  B. 마스크: 손으로 표시한 '가려진 틈' 영역을 마스크가 얼마나 덮었나 (기존 seg vs SAM)
  C. 검토 목록: 기준선이 놓친 정답이 자동 검토 목록 안에 들어 있나, 볼 곳이 몇 곳인가

실행:  python evaluate.py      (먼저 python poc.py 를 실행해 outputs/poc 를 만들어 둔다)
결과:  outputs/eval/eval.json, outputs/eval/*.jpg, 그리고 표를 화면에 출력
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import poc
from poc import ROOT, SAMPLES, Models, detect, imread, imwrite, iou, segment_sam, segment_yolo, tint

GT = json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))["images"]
OUT = ROOT / "outputs" / "eval"

# (표에 쓸 이름, 모델 파일, imgsz, conf)
METHODS = [
    ("기존: yolo26n 640 conf0.25", "yolo26n.pt", 640, 0.25),
    ("yolo26n 1280 conf0.25", "yolo26n.pt", 1280, 0.25),
    ("yolo26s 640 conf0.25", "yolo26s.pt", 640, 0.25),
    ("yolo26m 640 conf0.25", "yolo26m.pt", 640, 0.25),
    ("yolo26m 640 conf0.10", "yolo26m.pt", 640, 0.10),
    ("yolo26m 1280 conf0.10", "yolo26m.pt", 1280, 0.10),
]


def is_hit(gt: dict, pred: dict) -> bool:
    if gt["class"] != pred["class_name"]:
        return False
    if iou(gt["xyxy"], pred["xyxy"]) >= 0.5:
        return True
    if gt.get("small"):                       # 작은 물체: 예측 중심이 정답 박스(여유 50%) 안이고, 예측이 터무니없이 크지 않을 것
        x1, y1, x2, y2 = gt["xyxy"]
        px, py = (x2 - x1) * 0.5, (y2 - y1) * 0.5
        cx, cy = (pred["xyxy"][0] + pred["xyxy"][2]) / 2, (pred["xyxy"][1] + pred["xyxy"][3]) / 2
        p_area = (pred["xyxy"][2] - pred["xyxy"][0]) * (pred["xyxy"][3] - pred["xyxy"][1])
        g_area = (x2 - x1) * (y2 - y1)
        return x1 - px <= cx <= x2 + px and y1 - py <= cy <= y2 + py and p_area <= 6 * g_area
    return False


def score(gts: list[dict], preds: list[dict]) -> dict:
    """점수 높은 예측부터 정답에 하나씩 짝지음. 짝이 없는 예측은 잘못 찾은 것(중복 포함)."""
    used, found, false = set(), [], []
    for p in sorted(preds, key=lambda d: -d["score"]):
        k = next((i for i, g in enumerate(gts) if i not in used and is_hit(g, p)), None)
        if k is None:
            false.append(p)
        else:
            used.add(k)
            found.append(k)
    return {"found": sorted(found), "missed": [i for i in range(len(gts)) if i not in used], "false": false}


def label(g: dict) -> str:
    return g["class"] + (f' ({g["note"]})' if g.get("note") else "")


def save_small_zooms(images: dict) -> list[dict]:
    """작은 정답 물체(small=true)를 5배 확대해 저장하고, 크기를 숫자로 남긴다. (왜 놓쳤는지 살펴보는 근거)"""
    rows = []
    for n, img in images.items():
        h, w = img.shape[:2]
        for g in GT[n]["objects"]:
            if not g.get("small"):
                continue
            x1, y1, x2, y2 = g["xyxy"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            a, b, c, d = max(0, cx - 50), max(0, cy - 50), min(w, cx + 50), min(h, cy + 50)
            crop = img[b:d, a:c].copy()
            cv2.rectangle(crop, (x1 - a, y1 - b), (x2 - a, y2 - b), (0, 255, 255), 1)
            crop = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            name = f'zoom_{Path(n).stem}_{g["class"].replace(" ", "_")}.jpg'
            imwrite(OUT / name, crop)
            scale = poc.IMGSZ / max(h, w)                      # 모델에 들어갈 때 줄어드는 비율
            rows.append({"image": n, "class": g["class"], "file": name,
                         "size_px": [x2 - x1, y2 - y1],
                         "share_of_image": round((x2 - x1) * (y2 - y1) / (h * w), 5),
                         "size_at_model_input_640": [round((x2 - x1) * scale, 1), round((y2 - y1) * scale, 1)]})
    return rows


def main():
    models = Models()
    OUT.mkdir(parents=True, exist_ok=True)
    images = {n: imread(SAMPLES / n) for n in GT if (SAMPLES / n).exists()}
    n_gt = sum(len(GT[n]["objects"]) for n in images)
    report = {"images": list(images), "gt_objects": n_gt, "detection": [], "mask": [], "review": []}

    # 정답을 그려서 사람이 확인할 수 있게 저장
    for n, img in images.items():
        g_img = poc.draw(img, [{"class_name": g["class"], "score": 0, "xyxy": g["xyxy"]} for g in GT[n]["objects"]],
                         (255, 255, 255), 2, tag=False)
        for reg in GT[n].get("mask_regions", []):
            cv2.polylines(g_img, [np.array(reg["polygon"], np.int32)], True, (0, 255, 255), 2)
        imwrite(OUT / f"gt_{Path(n).stem}.jpg", g_img)

    report["small_objects"] = save_small_zooms(images)

    # ---------- A. 탐지 ----------
    for name, weights, imgsz, conf in METHODS:
        m = models.yolo(weights)
        for img in images.values():                      # 준비 운동(첫 실행은 느려서 시간 재기에 넣지 않음)
            m.predict(img, conf=conf, imgsz=imgsz, device="cpu", verbose=False)
        found = false = 0
        secs, missed_names, false_names = [], [], []
        for n, img in images.items():
            preds, t = detect(m, img, conf, imgsz)
            s = score(GT[n]["objects"], preds)
            found += len(s["found"])
            false += len(s["false"])
            secs.append(t)
            missed_names += [f"{n}: {label(GT[n]['objects'][i])}" for i in s["missed"]]
            false_names += [f"{n}: {p['class_name']} {p['score']:.2f} {[int(v) for v in p['xyxy']]}" for p in s["false"]]
        report["detection"].append({"method": name, "found": found, "total": n_gt, "false": false,
                                    "sec_per_image": round(float(np.mean(secs)), 2),
                                    "missed": missed_names, "false_list": false_names})

    # 글자 지목(YOLOE) — 설치되어 있을 때만
    prompts = sorted({g["class"] for n in images for g in GT[n]["objects"]})
    yoloe = models.yoloe(prompts)
    if yoloe is None:
        report["text_stage"] = f"건너뜀: {models.text_skip_reason}"
    else:
        report["text_stage"] = f"실행함, 지목한 글자: {prompts}"
        for conf in (0.25, 0.10):
            for imgsz in (640, 1280):
                found = false = 0
                secs, missed_names, false_names = [], [], []
                for n, img in images.items():
                    preds, t = detect(yoloe, img, conf, imgsz)
                    s = score(GT[n]["objects"], preds)
                    found += len(s["found"]); false += len(s["false"]); secs.append(t)
                    missed_names += [f"{n}: {label(GT[n]['objects'][i])}" for i in s["missed"]]
                    false_names += [f"{n}: {p['class_name']} {p['score']:.2f} {[int(v) for v in p['xyxy']]}" for p in s["false"]]
                report["detection"].append({"method": f"YOLOE(글자 지목) {imgsz} conf{conf:.2f}", "found": found,
                                            "total": n_gt, "false": false, "sec_per_image": round(float(np.mean(secs)), 2),
                                            "missed": missed_names, "false_list": false_names})

    # ---------- B. 마스크 ----------
    for n, img in images.items():
        regs = GT[n].get("mask_regions", [])
        if not regs:
            continue
        h, w = img.shape[:2]
        base, _ = detect(models.yolo("yolo26n.pt"), img, poc.BASE_CONF)
        insts, ymasks, t_y = segment_yolo(models.yolo("yolo26n-seg.pt"), img, poc.BASE_CONF)
        smasks, t_s = segment_sam(models.sam(), img, [d["xyxy"] for d in base])
        for reg in regs:
            region = np.zeros((h, w), np.uint8)
            cv2.fillPoly(region, [np.array(reg["polygon"], np.int32)], 1)
            region = region.astype(bool)
            ym = np.any(np.stack([m for i, m in zip(insts, ymasks) if i["class_name"] == reg["target"]] or
                                 [np.zeros((h, w), bool)]), axis=0)
            sm = np.any(np.stack([m for d, m in zip(base, smasks) if d["class_name"] == reg["target"]] or
                                 [np.zeros((h, w), bool)]), axis=0)
            row = {"image": n, "region": reg["name"], "region_px": int(region.sum()),
                   "yolo_seg_cover": round(float((ym & region).sum() / region.sum()), 3),
                   "sam_cover": round(float((sm & region).sum() / region.sum()), 3),
                   "yolo_seg_sec": round(t_y, 2), "sam_sec": round(t_s, 2)}
            report["mask"].append(row)
            xs, ys = zip(*reg["polygon"])
            a, b, c, d = max(0, min(xs) - 90), max(0, min(ys) - 60), min(w, max(xs) + 110), min(h, max(ys) + 60)
            tiles = []
            for mask, color, title in [(None, None, "original"), (ym, poc.COLORS["baseline"], "YOLO-seg (before)"),
                                       (sm, poc.COLORS["sam"], "SAM (after)")]:
                v = img if mask is None else tint(img, mask, color, 0.55)
                v = v.copy()
                cv2.polylines(v, [np.array(reg["polygon"], np.int32)], True, (0, 255, 255), 2)
                v = cv2.resize(v[b:d, a:c], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                cv2.putText(v, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(v, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                tiles.append(v)
            imwrite(OUT / f"mask_compare_{Path(n).stem}.jpg", np.hstack(tiles))

    # ---------- C. 검토 목록 ----------
    for n in images:
        rp = ROOT / "outputs" / "poc" / Path(n).stem / "result.json"
        if not rp.exists():
            report["review"].append({"image": n, "note": "outputs/poc 결과 없음 - 먼저 python poc.py 실행"})
            continue
        r = json.loads(rp.read_text(encoding="utf-8"))
        gts = GT[n]["objects"]
        base_missed = score(gts, r["baseline"])["missed"]
        cands = [c for c in r["review_candidates"] if c["kind"] == "missed?"]
        caught = [i for i in base_missed if any(is_hit(gts[i], c) for c in cands)]
        real = sum(1 for c in cands if any(is_hit(g, c) for g in gts))
        report["review"].append({
            "image": n, "candidates_total": len(r["review_candidates"]),
            "missed_candidates": len(cands), "missed_candidates_that_are_real": real,
            "baseline_missed": [label(gts[i]) for i in base_missed],
            "baseline_missed_caught_in_list": [label(gts[i]) for i in caught],
            "mask_gap_candidates": sum(c["kind"] == "mask-gap?" for c in r["review_candidates"]),
            "pipeline_total_sec": r["total_sec"], "text_stage": r["text_stage"]})

    (OUT / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n== A. 탐지 (정답 {n_gt}개, 사진 {len(images)}장) ==")
    for d in report["detection"]:
        print(f'  {d["method"]:<32} 찾음 {d["found"]}/{d["total"]}  잘못 찾음 {d["false"]}  {d["sec_per_image"]}s/장')
        for x in d["missed"]:
            print(f"      놓침: {x}")
        for x in d["false_list"]:
            print(f"      잘못: {x}")
    print(f'  글자 지목 단계: {report["text_stage"]}')
    print("\n== B. 마스크 ==")
    for m in report["mask"]:
        print(f'  {m["image"]} [{m["region"]}] 기존 {m["yolo_seg_cover"]:.1%} -> SAM {m["sam_cover"]:.1%} '
              f'(영역 {m["region_px"]}px, 기존 {m["yolo_seg_sec"]}s / SAM {m["sam_sec"]}s)')
    print("\n== C. 검토 목록 ==")
    for r in report["review"]:
        print(" ", json.dumps(r, ensure_ascii=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
