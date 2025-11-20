# AeroEyes Drone Detection & Tracking System (Zalo AI Challenge 2025)

The pipeline includes three main stages: Detection → Refinement → Tracking.

## 1. System Overview

---

### 1.1 Detection

-   Apply **CLAHE** and **horizontal-flip TTA**.
-   Run ensemble of: YOLOv8s-p2, YOLOv11s-p2 and D-FINE (DETR-like)
-   Merge outputs using **Weighted Boxes Fusion (WBF)**.

### 1.2 Refinement

-   Filter boxes by **min/max area**.
-   Extract features with **MobileNetV3-Small**.
-   Compute cosine similarity with reference images in `box_grabcut/`.
-   Select the box with the highest combined score.

### 1.3 Tracking

-   Track the final selected box using an **Kalman Filter**.

## 2. Environment Setup

### 2.1 Create Python 3.10.19 Virtual Environment

From the project root:

```bash
# Check you have Python 3.10.19 available
python3 --version     # or python --version

# Create virtual environment
python3.10 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate
```

### 2.2 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Dataset Preparation

-   Convert Raw Data → YOLO Format (`create_dataset.py`)

data/
└── yolo_dataset/
├── images/
│ ├── train/
│ ├── val/
│ └── test/
├── labels/
│ ├── train/
│ ├── val/
│ └── test/
└── dataset.yaml

-   D-FINE model requires COCO-style annotations. Follow the structure used in the official repo:
    data/
    └── coco/
    ├── images/
    │ ├── train/
    │ └── val/
    └── annotations/
    ├── instances_train.json
    └── instances_val.json

### 3.3 Reference Image Processing

-   Use `reference_image_processor.py` to create cleaned reference crops (e.g. via GrabCut)

### 3.3 Training

-   Training code and experiments are organized in Jupyter notebooks under: `/notebooks`
-   `cd D-FINE` for training D-Fine model

### Inference

```bash
python predict.py
```
