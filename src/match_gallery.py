import os, json, glob
import numpy as np
from PIL import Image
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INDEX_PATH = 'outputs/gallery.index'
META_PATH  = 'outputs/gallery_meta.json'
CROP_DIR   = 'data/crops'
OUT_JSON   = 'outputs/json'

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, 'r', encoding='utf-8') as f:
    metas = json.load(f)

THRESH = 0.28  # điều chỉnh sau thực nghiệm

for p in glob.glob(os.path.join(CROP_DIR, '*')):
    img = Image.open(p).convert('RGB')
    inputs = processor(images=img, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    q = feat.cpu().numpy().astype('float32')
    D, I = index.search(q, 1)
    score = float(D[0][0])
    midx = int(I[0][0])
    pred_brand = metas[midx]['brand']
    result = {
        'crop': p,
        'pred_brand': pred_brand,
        'gallery_item': metas[midx]['path'],
        'similarity': score,
        'decision': pred_brand if score >= THRESH else 'unknown'
    }
    out_path = os.path.join(OUT_JSON, os.path.splitext(os.path.basename(p))[0] + '_match.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(result)
