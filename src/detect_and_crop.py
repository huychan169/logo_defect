import os
import sys
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "best.pt"

IMAGES_DIR       = PROJECT_ROOT / "data" / "images_raw"
CROPS_DIR        = PROJECT_ROOT / "data" / "crops"
OUT_VIS_DIR      = PROJECT_ROOT / "outputs" / "vis"
OUT_DETECT_DIR   = PROJECT_ROOT / "outputs" / "detect"  
OUT_JSON_DIR     = PROJECT_ROOT / "outputs" / "json"     

CONF_THRES = 0.55
IOU_THRES  = 0.45


def ensure_dirs():
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DETECT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

def load_model():
    print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def list_images(folder: Path):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]



def process_single_image(model, image_path: Path):
    print(f"\n[INFO] Processing: {image_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None

    h, w = img_bgr.shape[:2]

    results = model.predict(
        source=img_bgr,
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False
    )

    r = results[0]
    boxes = r.boxes

    if len(boxes) == 0:
        print("[INFO] No detection found.")

        vis_path = OUT_VIS_DIR / f"{image_path.stem}_detected.jpg"
        cv2.imwrite(str(vis_path), r.plot())


        out_json = OUT_DETECT_DIR / f"{image_path.stem}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "file": str(image_path),
                "detections": []
            }, f, indent=2)

        return None


    detect_list = []

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())


        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)

        crop = img_bgr[y1:y2, x1:x2]
        crop_name = f"{image_path.stem}_crop{i}_{conf:.2f}.png"
        crop_path = CROPS_DIR / crop_name
        cv2.imwrite(str(crop_path), crop)

        detect_list.append({
            "cls": cls_id,
            "conf": conf,
            "bbox": [x1, y1, x2, y2],
            "crop": str(crop_path.relative_to(PROJECT_ROOT))
        })

    vis_path = OUT_VIS_DIR / f"{image_path.stem}_detected.jpg"
    cv2.imwrite(str(vis_path), r.plot())

    out_json = OUT_DETECT_DIR / f"{image_path.stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "file": str(image_path),
            "detections": detect_list
        }, f, indent=2)

    print(f"[DONE] → {len(detect_list)} crops saved.")
    return detect_list


def process_folder(model):
    imgs = list_images(IMAGES_DIR)
    if not imgs:
        print(f"[WARN] No images found in: {IMAGES_DIR}")
        return

    for img in imgs:
        process_single_image(model, img)


def main():
    ensure_dirs()
    model = load_model()

    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        process_single_image(model, img_path)
    else:
        process_folder(model)


if __name__ == "__main__":
    main()
