import json
import glob
from pathlib import Path

import faiss
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH   = PROJECT_ROOT / "outputs" / "gallery.index"
META_PATH    = PROJECT_ROOT / "outputs" / "gallery_meta.json"
CROP_DIR     = PROJECT_ROOT / "data" / "crops"
OUT_JSON_DIR = PROJECT_ROOT / "outputs" / "json"

THRESH_REAL = 0.6
THRESH_FAKE = 0.5


def ensure_dirs():
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)


def load_clip_and_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise SystemExit(
            f"Gallery index/meta not found. Run build_gallery.py first.\n"
            f"Expected: {INDEX_PATH}, {META_PATH}"
        )

    print("[INFO] Loading CLIP model (openai/clip-vit-base-patch32) ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"[INFO] Loading FAISS index from {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        metas = json.load(f)

    return model, processor, index, metas


def list_crops():
    files = glob.glob(str(CROP_DIR / "*.png")) + \
            glob.glob(str(CROP_DIR / "*.jpg")) + \
            glob.glob(str(CROP_DIR / "*.jpeg"))
    return [Path(p) for p in files]


def main():
    ensure_dirs()
    model, processor, index, metas = load_clip_and_index()

    crops = list_crops()
    if not crops:
        print(f"[WARN] No crops found in {CROP_DIR}")
        return

    for p in crops:
        print(f"[INFO] Matching crop: {p.name}")
        img = Image.open(p).convert("RGB")

        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)

        feat_np = feat.cpu().numpy().astype("float32")
        D, I = index.search(feat_np, k=1)

        similarity = float(D[0][0])
        gallery_idx = int(I[0][0])
        pred_brand = metas[gallery_idx]["brand"]

        if similarity >= THRESH_REAL:
            decision = f"{pred_brand}_real"
        elif similarity <= THRESH_FAKE:
            decision = f"{pred_brand}_fake"
        else:
            decision = "unknown"

        result = {
            "crop": str(p.relative_to(PROJECT_ROOT)),  
            "pred_brand": pred_brand,
            "gallery_item": metas[gallery_idx]["path"],
            "similarity": similarity,
            "decision": decision
        }

        out_json = OUT_JSON_DIR / f"{p.stem}_match.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(result)

    print("\n[DONE] Matching finished.")


if __name__ == "__main__":
    main()
