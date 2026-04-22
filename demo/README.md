---
title: Wildlife Behavior Analyzer
emoji: 🦒
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
short_description: AI-based animal behavior recognition from drone footage
---

# 🦒 Wildlife Behavior Analyzer

AI-powered animal behavior recognition from drone wildlife footage.

## What it does

Upload a drone video of wildlife and the app will:
- **Detect** each animal in every frame using YOLOv8
- **Classify** behavior per animal using the X3D-KABR model
- **Visualize** a behavior timeline and time budget chart
- **Export** the annotated video with per-animal behavior labels

## Behaviors detected

| Label | Description |
|---|---|
| Walk | Animal moving at walking pace |
| Trot | Animal moving at trot pace |
| Run | Animal moving at canter or gallop |
| Graze | Animal eating grass or vegetation |
| Browse | Animal eating from trees or bushes |
| Head Up | Animal looking around / observing |
| Auto-Groom | Animal grooming itself |

## Model

This app uses **X3D-KABR** — an X3D video model trained on the KABR dataset:

> Kholiavchenko et al. *KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition from Drone Videos.* WACV 2024.

The KABR dataset contains 10+ hours of drone footage of giraffes, plains zebras, and Grevy's zebras at the Mpala Research Centre in Kenya.

## Best results

- Drone/aerial footage at 10–50m altitude
- Large ungulates: giraffe, zebra, deer, horse
- Good lighting conditions
- Animals clearly visible (not too far)

## Processing time

- ~2–4 minutes for a 30-second video
- ~5–8 minutes for a 1-minute video

This is a research-grade model running on CPU — processing time is expected.

## Credits

- **X3D-KABR model**: [imageomics/x3d-kabr-kinetics](https://huggingface.co/imageomics/x3d-kabr-kinetics)
- **KABR Dataset**: [imageomics/KABR](https://huggingface.co/datasets/imageomics/KABR)
- **YOLOv8**: Ultralytics
- **Built by**: rohansingh0
