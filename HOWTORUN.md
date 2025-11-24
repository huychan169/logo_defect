# Logo Authenticity Pipeline (YOLOv8 + CLIP + FAISS)

This project performs **logo detection, logo cropping, brand matching, and authenticity estimation** using:

- **YOLOv8** → detect logos in full images
- **CLIP** → encode crops and gallery samples
- **FAISS** → fast similarity search
- **Overlay tools** → draw REAL / FAKE / UNKNOWN labels on original images

---

## 1. Project Structure

```
project/
│
├── best.pt                     # YOLO detection model
├── requirements.txt
├── README.md
│
├── src/
│   ├── detect_and_crop.py      # YOLO detect → crop → detection JSON
│   ├── build_gallery.py        # Build CLIP embeddings + FAISS index
│   ├── match_gallery.py        # Match crop with gallery (real/fake)
│   └── overlay_results.py      # Draw results on original images
│
├── data/
│   ├── images_raw/             # Input images to process
│   ├── crops/                  # Cropped logo patches
│   │
│   └── gallery/                # Reference samples (per brand)
│       ├── adidas/
│       │   ├── real/*.jpg
│       │   └── fake/*.jpg
│       ├── nike/
│       │   ├── real/*.jpg
│       │   └── fake/*.jpg
│       └── puma/
│           ├── real/*.jpg
│           └── fake/*.jpg
│
└── outputs/
    ├── detect/                 # YOLO JSON metadata
    ├── vis/                    # YOLO visualization (bbox)
    ├── json/                   # CLIP match results
    └── vis_full/               # Full visualization (REAL/FAKE on original image)
```

---

## 2. Installation

### Create environment

```bash
conda create -n trk python=3.9 -y
conda activate trk
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Data Preparation

### 3.1 Input images

Place all raw images here:

```
data/images_raw/
```

### 3.2 YOLO model

Place your trained YOLO logo-detector here:

```
best.pt
```

### 3.3 Gallery structure (REAL/FAKE reference samples)

```
data/gallery/
  adidas/
    real/*.jpg
    fake/*.jpg
  nike/
    real/*.jpg
    fake/*.jpg
  puma/
    real/*.jpg
    fake/*.jpg
```

---

## 4. Pipeline Overview

### **Step 1 — Detect logos & crop them**

Process entire folder:

```bash
python src/detect_and_crop.py
```

Or process one image:

```bash
python src/detect_and_crop.py data/images_raw/img_001.jpg
```

Output:

- Crops → `data/crops/`
- YOLO-vis → `outputs/vis/`
- YOLO JSON → `outputs/detect/`

---

### **Step 2 — Build CLIP gallery index**

```bash
python src/build_gallery.py
```

This generates:

```
outputs/gallery.index
outputs/gallery_meta.json
```

Used later for matching.

---

### **Step 3 — Match cropped logos against gallery**

```bash
python src/match_gallery.py
```

Outputs:

```
outputs/json/<crop>_match.json
```

Each file contains:

- detected brand
- matched gallery item
- similarity score
- final decision (brand_real, brand_fake, unknown)

---

### **Step 4 — Draw REAL / FAKE / UNKNOWN on original images**

```bash
python src/overlay_results.py
```

Outputs rendered images to:

```
outputs/vis_full/
```

Colors:

- **Green** → brand_real
- **Red** → brand_fake
- **Yellow** → unknown

---

## 5. Full Pipeline (recommended order)

```bash
python src/detect_and_crop.py
python src/build_gallery.py
python src/match_gallery.py
python src/overlay_results.py
```

---

## 6. Adjusting Real/Fake Thresholds

Inside `match_gallery.py`:

```python
THRESH_REAL = 0.50
THRESH_FAKE = 0.35
```

Interpretation:

| similarity    | decision   |
| ------------- | ---------- |
| ≥ THRESH_REAL | brand_real |
| ≤ THRESH_FAKE | brand_fake |
| otherwise     | unknown    |

Tune these values depending on your gallery diversity.
