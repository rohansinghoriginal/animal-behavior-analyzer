# 🦒 Wildlife Behavior Analyzer

<p align="center">
  <img src="docs/images/banner.png" alt="Wildlife Behavior Analyzer" width="100%"/>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/rohansingh0/wildlife-behavior-analyzer">
    <img src="https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow?style=for-the-badge" alt="HuggingFace Demo"/>
  </a>
  <a href="https://www.kaggle.com/code/rohansingh0/kabr-behavior-analysis">
    <img src="https://img.shields.io/badge/Kaggle-Notebook-blue?style=for-the-badge&logo=kaggle" alt="Kaggle Notebook"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-green?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-red?style=for-the-badge" alt="License"/>
</p>

---

## 📌 Overview

An **AI-powered wildlife behavior recognition system** that analyzes drone footage of animals and classifies their behavior in real time — per individual animal, per frame.

Built as a college project using the **KABR dataset** (WACV 2024) and **X3D-KABR** pretrained model. The system detects each animal using YOLOv8, extracts a mini-scene around it, and classifies behavior using a 3D convolutional video model.

### Detected Behaviors

| Label | Description |
|---|---|
| 🚶 Walk | Moving at walking pace |
| 🏃 Trot | Moving at trot pace |
| 💨 Run | Moving at canter or gallop |
| 🌿 Graze | Eating grass or vegetation |
| 🌳 Browse | Eating from trees or bushes |
| 👀 Head Up | Looking around / observing surroundings |
| 🧴 Auto-Groom | Grooming itself (licking, scratching) |

---

## 🏗️ Architecture

```
Drone Video Input
       │
       ▼
┌─────────────────┐
│   YOLOv8 (nano) │  ← Detects every animal per frame
│   COCO classes  │     (accepts all animal classes —
│   14–25         │      handles drone-view misclassification)
└────────┬────────┘
         │  bounding box per animal
         ▼
┌─────────────────────┐
│  Mini-scene buffer  │  ← 16-frame sliding window
│  per animal (deque) │     centered on each detection
└────────┬────────────┘
         │  (16, 300, 300, 3) clip
         ▼
┌─────────────────────┐
│   X3D-KABR Model    │  ← X3D-L pretrained on KABR dataset
│   (WACV 2024)       │     100% weights loaded via
│   5.4M parameters   │     custom SlowFast→pytorchvideo
│                     │     key remapping
└────────┬────────────┘
         │  8-class sigmoid output
         ▼
┌─────────────────────┐
│  Behavior Label +   │  ← Per-animal annotation overlay
│  Confidence Score   │     + time budget chart
│  + Timeline Plot    │     + behavior timeline
└─────────────────────┘
```

---

## 🗂️ Repository Structure

```
wildlife-behavior-analyzer/
│
├── demo/                          # Streamlit web app (HuggingFace Spaces)
│   ├── app.py                     # Main Streamlit application
│   ├── requirements.txt           # Python dependencies
│   ├── packages.txt               # System-level apt packages
│   └── README.md                  # HuggingFace Space description
│
├── notebooks/                     # Kaggle notebooks
│   └── kabr_behavior_analysis.ipynb   # Full training + inference pipeline
│
├── docs/
│   └── images/                    # README assets
│
├── .gitignore
├── LICENSE
└── README.md                      # This file
```

> **Note on external assets:**
> - Model weights live on [HuggingFace](https://huggingface.co/imageomics/x3d-kabr-kinetics) — downloaded at runtime, not stored here
> - Training data lives on [HuggingFace Datasets](https://huggingface.co/datasets/imageomics/KABR) — streamed at runtime
> - Kaggle notebook runs on [Kaggle](https://kaggle.com) with GPU — not duplicated here

---

## 🚀 Live Demo

**Try it now:** [huggingface.co/spaces/rohansingh0/wildlife-behavior-analyzer](https://huggingface.co/spaces/rohansingh0/wildlife-behavior-analyzer)

Upload a short drone wildlife video (≤30 seconds) and get:
- Annotated video with per-animal behavior labels
- Time budget pie chart
- Behavior timeline plot

---

## 🧠 Model Details

### X3D-KABR

| Property | Value |
|---|---|
| Architecture | X3D-L (3D CNN) |
| Parameters | 5.4M |
| Training data | KABR dataset (WACV 2024) |
| Input | 16 frames × 300×300px |
| Output | 8-class sigmoid (multi-label) |
| Published accuracy | 65.1% mAP (giraffes) |

The checkpoint uses **SlowFast framework naming** (`s1.pathway0_stem...`) while inference uses **pytorchvideo** (`blocks.0.conv...`). This repo includes a complete key remapping function achieving **100% weight coverage** without requiring the SlowFast framework (which is incompatible with Python 3.12+).

### YOLOv8

- Model: `yolov8n.pt` (nano — fast, CPU-friendly)
- Detects all COCO animal classes (IDs 14–25)
- Drone-view footage causes species misclassification (giraffe detected as bear) — handled by overriding with user-specified species name while keeping correct bounding boxes

---

## 📓 Kaggle Notebook

The full pipeline notebook is at [`notebooks/kabr_behavior_analysis.ipynb`](notebooks/kabr_behavior_analysis.ipynb).

**To run on Kaggle:**
1. Go to [kaggle.com](https://kaggle.com) → Create new notebook
2. Upload `kabr_behavior_analysis.ipynb`
3. Enable **GPU T4×2** accelerator
4. Add your drone video as a dataset
5. Update `VIDEO_PATH` and `SPECIES_NAME` in Step 7
6. Run all cells

**Requirements on Kaggle (auto-installed by notebook):**
```
pytorchvideo  huggingface_hub  ultralytics  fvcore
```

---

## 🖥️ Local Setup

> ⚠️ Requires Python 3.10 or 3.11 (not 3.12 — SlowFast incompatibility).  
> Minimum 8GB RAM. GPU optional but recommended.

```bash
# Clone
git clone https://github.com/rohansingh0/wildlife-behavior-analyzer.git
cd wildlife-behavior-analyzer

# Create environment
conda create -n wildlife python=3.10 -y
conda activate wildlife

# Install dependencies
pip install -r demo/requirements.txt

# Run the app
streamlit run demo/app.py
```

---

## 📊 Dataset

**KABR — Kenyan Animal Behavior Recognition**

| Property | Value |
|---|---|
| Source | Mpala Research Centre, Kenya |
| Collection | DJI Mavic 2S drone, 5.4K resolution |
| Duration | 10+ hours annotated video |
| Species | Giraffe, Plains Zebra, Grevy's Zebra |
| Frames | 1,139,893 |
| Behaviors | 8 classes |
| License | CC0 (public domain) |

> Kholiavchenko et al. *KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition from Drone Videos.* WACV 2024.

---

## 🔬 Key Technical Challenges Solved

### 1. SlowFast → pytorchvideo Weight Remapping
The X3D-KABR checkpoint uses SlowFast's internal naming convention. Since SlowFast is broken on Python 3.12+, we built a complete regex-based key remapping function that translates all 1,141 weight tensors to pytorchvideo format — achieving 100% coverage with zero architecture changes.

### 2. Domain Gap in Object Detection
YOLOv8 misclassifies animals from drone altitude (giraffe → bear, zebra → horse). Fixed by accepting all COCO animal class detections (IDs 14–25) and overriding the display label with the user-specified species name. Bounding box accuracy is unaffected.

### 3. Memory Management on CPU
X3D inference on CPU is memory-intensive. Mitigations applied:
- Frame downscaling to max 640px wide
- Periodic buffer clearing every 150 frames
- Timeline list capped at 1,000 entries
- Fast Mode: 8-frame clips instead of 16
- `torch.set_num_threads(2)` to prevent thread overload

---

## 📈 Sample Results

```
Video: giraffe_drone.mp4 (30s, 1280×720, 60fps)
Processing time: ~4 minutes (CPU)

Behavior Summary:
  Graze        ████████████████████  82.3%  (2181 detections)
  Trot         ██                    12.8%  (339  detections)
  Run                                 3.5%  (93   detections)
  Walk                                1.3%  (34   detections)
  Auto-Groom                          0.1%  (2    detections)
```

---

## 🗺️ Roadmap

- [ ] Add animal tracking (SORT/DeepSORT) for consistent per-individual IDs
- [ ] Support zebra and elephant species presets
- [ ] Optimize with ONNX export for faster CPU inference
- [ ] Add multi-video batch processing
- [ ] Build time budget comparison across multiple videos

---

## 📚 References

```bibtex
@inproceedings{kholiavchenko2024kabr,
  title={KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition from Drone Videos},
  author={Kholiavchenko, Maksim and Kline, Jenna and others},
  booktitle={WACV Workshops},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The KABR dataset and X3D-KABR model are CC0 / MIT licensed respectively.

---

## 👤 Author

**Rohan Singh** — Engineering Student  
[GitHub](https://github.com/rohansingh0) · [HuggingFace](https://huggingface.co/rohansingh0)

---

<p align="center">
  Built with ❤️ using X3D-KABR · YOLOv8 · pytorchvideo · Streamlit
</p>
