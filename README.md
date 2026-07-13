

# 🚗 YOLOPv2 Professional ADAS Perception System

This project transforms the standard YOLOPv2 (You Only Look Once for Panoptic Driving) model into a **Modern Autonomous Driving Perception Dashboard**. 

By building a modular intelligent layer on top of the existing YOLOPv2 inference outputs, we generate a real-time Advanced Driver Assistance System (ADAS) without modifying or retraining the neural network. The final output resembles a professional automotive perception system suitable for engineering portfolios, interviews, and university projects.

![ADAS Demo](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2011.8+-red)

## ✨ Key Features

- **Lane Departure Warning (LDW):** Computes vehicle offset and triggers visual warnings with drift direction.
- **Driving Corridor Generation:** Dynamically fills the safe driving path green (or red if departing).
- **Lane Curvature Estimation:** Classifies road type (Straight, Gentle/Sharp Left/Right) using robust polynomial fitting.
- **Steering Suggestion:** Calculates pure-pursuit-based steering angle and draws a directional arrow.
- **Vehicle-in-Lane Analysis:** Isolates vehicles in the current lane and labels them red.
- **Collision Risk Estimation:** Classifies risk (SAFE, CAUTION, HIGH RISK) based on bounding box distance and lane occupancy.
- **Drivable Area Analytics:** Calculates the percentage of visible drivable area.
- **Lane Confidence Score:** Evaluates lane tracking quality based on pixel continuity and histogram peaks.
- **Real-Time ADAS Dashboard:** A clean, automotive-style sidebar displaying FPS, inference time, road type, steering, and collision metrics.
- **Trajectory & Distance Bands:** Projects 5m, 10m, 20m, and 30m distance markers and draws a smooth predicted trajectory.

---

## 📂 Folder Structure

Ensure your project is organized exactly like this:

```text
YOLOPv2-ADAS/
├── data/
│   ├── weights/
│   │   └── yolopv2.pt          # Your original YOLOPv2 TorchScript model
│   └── road.mp4                # Your test video/images
│
├── utils/
│   └── utils.py                 # Original YOLOPv2 utilities (untouched)
│
├── adas/                        # 🧠 NEW: ADAS Intelligence Package
│   ├── __init__.py
│   ├── config.py                # Tunable parameters (camera, thresholds, colors)
│   ├── geometry.py              # IPM, distance mapping, polyfit math
│   ├── lane_tracker.py          # Lane detection, LDW, curvature, confidence
│   ├── steering.py              # Pure-pursuit steering angle estimation
│   ├── object_analysis.py       # Vehicle-in-lane, collision risk
│   ├── drivable_area.py         # Drivable area %
│   └── visualizer.py            # Professional dashboard & overlay rendering
│
├── demo.py                      # Original YOLOPv2 demo (untouched)
├── demo_adas.py                 # 🚀 NEW: ADAS entry point script
└── README.md
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- NVIDIA GPU (e.g., RTX 4050/3060/4090) with latest drivers.
- Python 3.8+ 
- CUDA Toolkit 11.8 (or compatible with your GPU)

### 2. Install PyTorch with CUDA (Crucial for 20+ FPS)
If you install standard PyTorch, it will default to CPU, giving you only 2 FPS. You **must** install the CUDA version. Open your terminal and run:

```bash
# Uninstall any existing CPU-only PyTorch
pip uninstall torch torchvision torchaudio

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install OpenCV and Numpy
```bash
pip install opencv-python numpy
```

---

## 🚀 How to Run

To run the ADAS perception system, use the `demo_adas.py` script. 

### Basic Command (Video)
```bash
python demo_adas.py --source data/road.mp4 --device 0
```

### ⚡ High FPS Command (20+ FPS)
To achieve maximum real-time performance on an RTX 4050, lower the inference image size. This cuts inference time drastically without hurting lane detection:
```bash
python demo_adas.py --source data/road.mp4 --device 0 --img-size 384
```

### Other Options
- **Webcam:** `python demo_adas.py --source 0 --device 0`
- **Image:** `python demo_adas.py --source data/example.jpg --device 0`
- **Save Video:** Add `--nosave` flag to disable saving, or leave it off to automatically save to `runs/detect/adas/`.

---

## ⚙️ How It Works (Architecture)

The system strictly adheres to the rule: **Do not touch the neural network.**

1. **Inference:** `demo_adas.py` loads the original `yolopv2.pt` model and runs forward passes to extract:
   - Object Detections (Cars, Trucks, Buses)
   - Drivable Area Mask (Binary)
   - Lane Line Mask (Binary)
2. **Geometry (`adas/geometry.py`):** Converts 2D image pixels into 3D world coordinates using a pinhole camera model (flat ground assumption) to estimate distances and project 5m/10m/20m bands.
3. **Lane Tracking (`adas/lane_tracker.py`):** Uses a sliding window algorithm over the lane mask to fit 2nd-degree polynomials. It includes heavy temporal smoothing (EMA) to prevent jitter and calculates a `safe_y_top` boundary to prevent the green corridor from spilling at the horizon.
4. **Object Analysis (`adas/object_analysis.py`):** Filters COCO classes {2, 5, 7} and uses ray-casting to check if a vehicle's bottom-center point falls inside the lane polygon. Calculates collision risk based on inverse distance.
5. **Visualization (`adas/visualizer.py`):** Blends the raw model mask with the dynamic ADAS overlays, applying alpha blending and anti-aliased rendering for a clean, modern UI.

## 🔧 Tuning Parameters
All constants are centralized in **`adas/config.py`**. 
If the lane width feels too narrow, or the collision risk triggers too early, you can easily adjust:
- `CAMERA_HEIGHT` & `FOCAL_LENGTH` for distance accuracy.
- `LDW_THRESHOLD` for lane departure sensitivity.
- `SAFE_DIST` & `CAUTION_DIST` for collision warnings.
