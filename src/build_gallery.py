import os, json, glob
import numpy as np
from PIL import Image
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GALLERY_DIR = 'data/gallery'
OUT_INDEX = 'outputs/gallery.index'
OUT_META = 'outputs/gallery_meta.json'

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

embeds, metas = [], []
for brand in sorted(os.listdir(GALLERY_DIR)):
    bdir = os.path.join(GALLERY_DIR, brand)
    if not os.path.isdir(bdir):
        continue
    for p in glob.glob(os.path.join(bdir, '*')):
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            continue
        inputs = processor(images=img, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        embeds.append(feat.squeeze(0).cpu().numpy())
        metas.append({'brand': brand, 'path': p})

if not embeds:
    raise SystemExit('No gallery images found.')

embeds = np.stack(embeds).astype('float32')
index = faiss.IndexFlatIP(embeds.shape[1])
index.add(embeds)
faiss.write_index(index, OUT_INDEX)
with open(OUT_META, 'w', encoding='utf-8') as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)
print(f'Saved {len(embeds)} entries to {OUT_INDEX}')
