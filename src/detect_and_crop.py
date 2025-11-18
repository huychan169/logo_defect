import os, json, cv2, uuid
from PIL import Image
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = 'google/owlv2-base-patch16-ensemble'

processor = Owlv2Processor.from_pretrained(MODEL_ID)
model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)

PROMPTS = [
    ["logo nike"],
    ["logo yourbrand"],
]

CONF_THRESH = 0.25
NMS_IOU = 0.5

IN_DIR = 'data/images_raw'
OUT_CROP = 'data/crops'
OUT_VIS = 'outputs/vis'
OUT_JSON = 'outputs/json'
os.makedirs(OUT_CROP, exist_ok=True)
os.makedirs(OUT_VIS, exist_ok=True)
os.makedirs(OUT_JSON, exist_ok=True)

def nms(boxes, scores, iou_thr=0.5):
    import numpy as np
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(float)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = (ovr <= iou_thr).nonzero()[0]
        order = order[inds+1]
    return keep

for fname in os.listdir(IN_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    path = os.path.join(IN_DIR, fname)
    image = Image.open(path).convert('RGB')
    inputs = processor(text=PROMPTS, images=image, return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=[image.size[::-1]])[0]

    boxes = results['boxes'].cpu().numpy()
    scores = results['scores'].cpu().numpy()
    labels = results['labels'].cpu().numpy()

    import numpy as np
    keep = np.where(scores >= CONF_THRESH)[0]
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = nms(boxes, scores, NMS_IOU)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    img_bgr = cv2.imread(path)
    H, W = img_bgr.shape[:2]
    annos = []
    for b, s, lab in zip(boxes, scores, labels):
        x1,y1,x2,y2 = [int(max(0, v)) for v in b]
        x1,y1,x2,y2 = max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)
        crop = img_bgr[y1:y2, x1:x2]
        cid = str(uuid.uuid4())[:8]
        crop_path = os.path.join(OUT_CROP, f'{os.path.splitext(fname)[0]}_{cid}.png')
        cv2.imwrite(crop_path, crop)
        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_bgr, f"{PROMPTS[lab][0]}:{s:.2f}", (x1, max(10,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        annos.append({
            'bbox':[x1,y1,x2,y2], 'score': float(s), 'label_idx': int(lab),
            'label': PROMPTS[lab][0], 'crop': crop_path
        })
    cv2.imwrite(os.path.join(OUT_VIS, fname), img_bgr)
    with open(os.path.join(OUT_JSON, os.path.splitext(fname)[0]+'.json'), 'w', encoding='utf-8') as f:
        json.dump({'file': path, 'detections': annos}, f, ensure_ascii=False, indent=2)
    print(f'Processed {fname}: {len(annos)} detections')
