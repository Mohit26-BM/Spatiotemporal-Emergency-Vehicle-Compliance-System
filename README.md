# Spatiotemporal Emergency Vehicle Compliance System (SCAS)

An AI-powered traffic enforcement pipeline that detects emergency vehicles, tracks all surrounding traffic, and flags vehicles that fail to yield — using spatiotemporal rules to eliminate false positives. Validated on real overhead and dashcam footage.


![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Stars](https://img.shields.io/github/stars/YourUsername/Emergency-Vehicle-Compliance-System)
![Forks](https://img.shields.io/github/forks/YourUsername/Emergency-Vehicle-Compliance-System)
![Issues](https://img.shields.io/github/issues/YourUsername/Emergency-Vehicle-Compliance-System)
![Last Commit](https://img.shields.io/github/last-commit/YourUsername/Emergency-Vehicle-Compliance-System)

## How It Works

```text
Video Input → YOLOv8n Detection → Ensemble Classification → DeepSORT Tracking → Compliance Engine → SQLite + Evidence
```

| Stage          | Component               | Detail                                                  |
| -------------- | ----------------------- | ------------------------------------------------------- |
| Detection      | YOLOv8n                 | Bounding boxes per frame                                |
| Classification | YOLOv8n-cls + ResNet18  | Emergency vs non-emergency (ensemble)                   |
| Tracking       | DeepSORT                | Persistent track IDs, majority-vote label stabilisation |
| Compliance     | Rule engine             | Proximity radius, reaction window, yield distance       |
| Logging        | SQLite 3NF + JPEG crops | Audit-ready violation records                           |

---

## Model Performance

Trained with 5-Fold Stratified Cross-Validation on the [Kaggle emergency vehicle dataset](https://www.kaggle.com/datasets/parthplc/emergency-vs-nonemergency-vehicle-classification).

| Metric                          | Value                              |
| ------------------------------- | ---------------------------------- |
| CV Mean Accuracy                | **95.62%** ± 0.26%                 |
| Best Fold (Fold 3) Val Accuracy | **95.90%**                         |
| Held-Out Test Accuracy          | **95.25%**                         |
| F1 Score                        | **0.95**                           |
| Ensemble                        | YOLOv8n-cls (50%) + ResNet18 (50%) |

Key training choices: differential learning rates, label smoothing (ε=0.1), AdamW with cosine annealing, early stopping (patience=10).

---

## Real-World Results

Tested on two real traffic videos — one overhead intersection camera and one dashcam.

| Video              | Resolution | FPS | Frames    | EV Detection | Violations | Unique Blockers |
| ------------------ | ---------- | --- | --------- | ------------ | ---------- | --------------- |
| Video 3 — Overhead | 828×720    | 60  | 756       | 87.6%        | 10         | 7               |
| Video 4 — Dashcam  | 1280×720   | 30  | 524       | 94.3%        | 4          | 3               |
| **Total**          |            |     | **1,280** | **90.3%**    | **14**     | **10**          |

**98% reduction** in false positives vs a distance-only baseline (300–500 violations/video → 4–10).

Violation types: Approaching (79%), Slow Yield (14%), Stationary (7%).

---

## Compliance Engine Parameters

| Parameter         | Default   | Meaning                                                |
| ----------------- | --------- | ------------------------------------------------------ |
| `proximity_r`     | 180 px    | Zone around EV where compliance is checked             |
| `reaction_w`      | 30 frames | Sliding window for yield calculation (~1 sec @ 30 FPS) |
| `yield_threshold` | 40 px     | Minimum distance increase to count as a yield          |

Camera-aware: uses Euclidean distance for overhead cameras, vertical distance for dashcams.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Mohit26-BM/Spatiotemporal-Emergency-Vehicle-Compliance-System.git
cd Spatiotemporal-Emergency-Vehicle-Compliance-System

# Install dependencies
pip install -r frontend/requirements.txt
pip install ultralytics deep-sort-realtime torch torchvision opencv-python kagglehub scikit-learn

# Run the dashboard
streamlit run frontend/app.py
```

> **Model weights** are not included in this repo due to file size. Download from the training notebook output or re-run the K-Fold training pipeline in `notebooks/Notebook_Final.ipynb`.

---

## Project Structure

```text
├── frontend/
│   ├── app.py                      # Streamlit entry point
│   └── pages_modules/
│       ├── home.py                 # Overview page
│       ├── dashboard.py            # Model metrics + charts
│       ├── video_analysis.py       # Input/output video viewer
│       └── compliance_report.py    # Violation log + filters
├── notebooks/
│   └── Notebook_Final.ipynb        # Training + inference notebook
├── outputs/
│   ├── plots/                      # Confusion matrix, K-Fold chart, cv_summary.json
│   ├── screenshots/                # EV detection + violation screenshots
│   └── videos/                     # Input and annotated output videos
├── docs/                           # Detailed notebook documentation (3 parts)
└── Research/                       # Capstone report + 13 academic papers
```

---

## Tech Stack

Python · PyTorch · Ultralytics YOLOv8 · OpenCV · DeepSORT · SQLite · Streamlit
