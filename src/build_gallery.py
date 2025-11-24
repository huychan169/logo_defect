import json
from pathlib import Path

import faiss
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GALLERY_ROOT = PROJECT_ROOT / "data" / "gallery"

INDEX_PATH   = PROJECT_ROOT / "outputs" / "gallery.index"
META_PATH    = PROJECT_ROOT / "outputs" / "gallery_meta.json"


def list_gallery_images():
    """
    Quét đệ quy toàn bộ data/gallery và lấy:
      - brand  = tên folder level 1 (adidas / nike / puma / ...)
      - variant = tên folder level 2 nếu có (real / fake / ...), nếu không có thì 'unknown'
    Ví dụ path:
      data/gallery/adidas/real/img1.jpg -> brand='adidas', variant='real'
      data/gallery/adidas/fake/img2.jpg -> brand='adidas', variant='fake'
      data/gallery/nike/img3.jpg        -> brand='nike',   variant='unknown'
    """
    image_infos = []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if not GALLERY_ROOT.exists():
        return image_infos

    for brand_dir in GALLERY_ROOT.iterdir():
        if not brand_dir.is_dir():
            continue
        brand = brand_dir.name  # adidas, nike, puma...

        # duyệt tất cả file con (đệ quy)
        for path in brand_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in exts:
                continue

            # xác định variant: real/fake/... nếu có
            rel = path.relative_to(GALLERY_ROOT)  # adidas/real/img1.jpg
            parts = rel.parts  # ('adidas','real','img1.jpg') hoặc ('nike','img3.jpg',)

            if len(parts) >= 3:
                variant = parts[1]  # adidas / **real** / img1.jpg
            else:
                variant = "unknown"

            image_infos.append({
                "path": str(path.relative_to(PROJECT_ROOT)),  # lưu path tương đối project
                "brand": brand,
                "variant": variant,
            })

    return image_infos


def main():
    print("[INFO] Loading CLIP model (openai/clip-vit-base-patch32) ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_infos = list_gallery_images()
    if not image_infos:
        print(f"No gallery images found in {GALLERY_ROOT}")
        return

    print(f"[INFO] Found {len(image_infos)} gallery images.")

    feats = []
    metas = []

    for info in image_infos:
        img_path = PROJECT_ROOT / info["path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {img_path}: {e}")
            continue

        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)

        feats.append(feat.cpu().numpy())
        metas.append(info)

    if not feats:
        print("[WARN] No valid gallery features extracted.")
        return

    feats_np = np.concatenate(feats, axis=0).astype("float32")
    d = feats_np.shape[1]

    print(f"[INFO] Building FAISS index with dim={d}, n={feats_np.shape[0]} vectors")
    index = faiss.IndexFlatIP(d)
    index.add(feats_np)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved FAISS index to {INDEX_PATH}")
    print(f"[DONE] Saved gallery meta to {META_PATH}")


if __name__ == "__main__":
    main()
