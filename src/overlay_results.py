import json
import glob
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT     = Path(__file__).resolve().parent.parent
DETECT_JSON_DIR  = PROJECT_ROOT / "outputs" / "detect"  
MATCH_JSON_DIR   = PROJECT_ROOT / "outputs" / "json"    
OUT_IMG_DIR      = PROJECT_ROOT / "outputs" / "vis_full"


def ensure_dirs():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_match_results():
    """
    Đọc toàn bộ *_match.json từ match_gallery
    => map: crop_path (POSIX) -> {pred_brand, similarity, decision}
    """
    match_map = {}

    pattern = str(MATCH_JSON_DIR / "*_match.json")
    for jpath in glob.glob(pattern):
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        crop_rel = data.get("crop") or data.get("crop_path")
        if not crop_rel:
            continue

        crop_key = str(Path(crop_rel).as_posix())
        match_map[crop_key] = {
            "pred_brand": data.get("pred_brand"),
            "similarity": float(data.get("similarity") or 0.0),
            "decision": data.get("decision"),
        }

    print(f"[INFO] Loaded {len(match_map)} matched crops from gallery.")
    return match_map


def draw_label(draw, bbox, text, color):
    x1, y1, x2, y2 = bbox


    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    try:
        text_w, text_h = draw.textsize(text, font=font)
    except AttributeError:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top

    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    bg = [x1, max(0, y1 - text_h - 4), x1 + text_w + 8, y1]
    draw.rectangle(bg, fill=color)
    draw.text((bg[0] + 4, bg[1] + 2), text, fill="white", font=font)


def main():
    ensure_dirs()
    match_map = load_match_results()

    detect_files = [Path(p) for p in glob.glob(str(DETECT_JSON_DIR / "*.json"))]
    if not detect_files:
        print(f"[WARN] No detect JSON found in {DETECT_JSON_DIR}")
        return

    for det_path in detect_files:
        with open(det_path, "r", encoding="utf-8") as f:
            det_data = json.load(f)

        img_path = Path(det_data["file"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path

        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        print(f"[INFO] Visualizing: {img_path.name}")
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for det in det_data.get("detections", []):
            bbox = det["bbox"]
            crop_rel = det.get("crop")
            if not crop_rel:
                continue

            crop_key = str(Path(crop_rel).as_posix())
            mi = match_map.get(crop_key)

            if mi is None:
                label = "logo_unverified"
                color = "yellow"
            else:
                decision   = (mi.get("decision") or "unknown").lower()
                pred_brand = mi.get("pred_brand") or "brand"
                sim        = mi.get("similarity") or 0.0

                if "fake" in decision:
                    color = "red"
                    label = f"{pred_brand}_fake ({sim:.2f})"
                elif "real" in decision:
                    color = "lime"
                    label = f"{pred_brand}_real ({sim:.2f})"
                else:
                    color = "yellow"
                    label = f"{pred_brand}_unknown ({sim:.2f})"

            draw_label(draw, bbox, label, color)

        out_name = img_path.stem + "_vis.png"
        out_path = OUT_IMG_DIR / out_name
        img.save(out_path)
        print(f"[DONE] Saved: {out_path}")

    print("\n[ALL DONE] Full-image visualization complete.")


if __name__ == "__main__":
    main()
