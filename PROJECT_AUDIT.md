# PROJECT AUDIT: SMARTSPARK AI (PARKSENSE AI)
**Audit Date:** November 2026  
**Auditor:** Senior AI/ML & Computer Vision Specialist  
**Repository Path:** `c:\Desktop\PROJECTS\ParkSense-AI`  
**Initial Total Size:** 1,661.3 MB (1.66 GB) — 67,821 files  

---

## 1. Executive Summary

An in-depth architectural and code audit of the **SmartPark AI** (formerly ParkSense AI) repository was conducted. The project aims to provide an intelligent parking occupancy detection system combining computer vision object detection (YOLO) and convolutional image patch classification (ResNet18).

While the original codebase establishes a baseline proof-of-concept, significant architectural shortcomings, algorithmic limitations, duplicated storage, and lack of integration were uncovered. Specifically:
1. **Repository Bloat:** The repository contains a 1,569 MB virtual environment (`.venv`) and duplicate datasets (`CNRPark-Patches-150x150` and `dataset`), with no `.gitignore` file.
2. **Matching Inaccuracies:** The slot occupancy engine relies exclusively on strict vehicle center-point containment within bounding rectangles. On the reference image (`parking.jpg`), 10 vehicles were detected but only 2 slots were registered as OCCUPIED (80% missed due to center-point edge effects).
3. **Disconnected Models:** YOLO vehicle detection and ResNet18 patch classification exist as isolated scripts. ResNet18 weights (`parking_classifier.pth`) were neither persisted nor integrated into an end-to-end decision engine.
4. **Inefficient Inference:** Tiling a low-resolution image (415×612 px) into 306×208 sub-windows introduced redundant model forward passes, edge slicing artifacts, and slow NMS.
5. **Missing Production Layers:** Lacks automated tests, structured logging, centralized configuration, video streaming pipeline, and a production-grade user dashboard.

---

## 2. Directory Structure & File Audit

### 2.1 File & Directory Breakdown

| Path | Size (MB) | File Count | Purpose / Status |
| :--- | :---: | :---: | :--- |
| `.venv/` | 1,569.42 MB | ~55,200 | Python 3.11 virtual environment. **Critical bloat:** Uncommitted dependency folder that must be ignored via `.gitignore`. |
| `CNRPark-Patches-150x150/` | 33.17 MB | 12,584 | Original patch dataset from CNRPark (Cameras A & B; classes `busy` and `free`). |
| `dataset/` | 33.17 MB | 12,584 | Duplicate 80/20 train/val split created by `src/prepare_dataset.py`. |
| `runs/` | 0.59 MB | 3 | Output runs from Ultralytics YOLO testing (`predict`, `predict-2`). |
| `src/` | 0.01 MB | 5 | Source Python scripts. |
| `yolo26s.pt` | 20.42 MB | 1 | YOLOv8s pretrained weights (COCO 80-class). |
| `yolo26n.pt` | 5.54 MB | 1 | YOLOv8n pretrained weights (COCO 80-class). |
| `parking.jpg` | 0.05 MB | 1 | Test parking lot scene (415 × 612 × 3 BGR image). |
| `bus.jpg` | 0.14 MB | 1 | Sample ultralytics image. |
| `parking_slots.json` | 1.51 KB | 1 | 22 parking slot 2D bounding boxes `[x1, y1, x2, y2]`. |
| `Model architecture ResNet18.txt` | 0.37 KB | 1 | Text diagram of ResNet18 pipeline. |

---

## 3. Analysis of Existing Codebase

### 3.1 `src/main.py`
- **Functionality:** Runs YOLO (`yolo26n.pt`) with `conf=0.15` and `imgsz=1280` on `parking.jpg`. Prints detected objects to stdout.
- **Issues:**
  - Hardcoded paths and parameters.
  - Upscaling a 415×612 image to `imgsz=1280` creates excessive memory footprint and interpolation blur with no accuracy benefit.
  - No slot mapping or occupancy inference.

### 3.2 `src/define_slots.py`
- **Functionality:** OpenCV GUI window allowing interactive mouse clicks to draw bounding boxes and save them into `parking_slots.json`.
- **Issues:**
  - Does not support editing existing slots (only append or pop last).
  - Bounding boxes are axis-aligned rectangles; real parking bays often have perspective perspective/skew angles.
  - Lacks slot labeling / numbering management or slot validation.

### 3.3 `src/occupancy.py`
- **Functionality:** Implements sliding-tile vehicle detection using `yolo26s.pt` with a tile size of 306×208 and overlap of 60 px. Applies custom IoU-based NMS, then checks if vehicle bounding box center is inside slot bounding box.
- **Critical Issues Identified:**
  1. **Tiling Inefficiency:** On a 415×612 image, slicing into 306×208 tiles produces multiple overlapping windows, increasing inference latency by 4×–6× over full-image single-shot inference.
  2. **Severe Center-Point Fallacy:** `vehicle_in_slot` calculates `(vx1+vx2)//2, (vy1+vy2)//2` and tests strict containment within slot `[sx1, sy1, sx2, sy2]`. If a car is parked diagonally or slightly overlaps the slot boundary, the center point falls 1 px outside, marking the slot FREE despite being 90% blocked.
  3. **Multi-Allocation Conflict:** No one-to-one or Hungarian assignment. A single oversized vehicle bounding box could trigger multiple slots, or multiple vehicles could compete without resolution.
  4. **Lack of ResNet Integration:** ResNet18 is completely absent from this script.
  5. **Static Execution Only:** No CLI arguments, no video support, no API, no return values for downstream consumers.

### 3.4 `src/prepare_dataset.py`
- **Functionality:** Reads images from `CNRPark-Patches-150x150/A` and `B`, splits them 80/20 randomly with fixed seed `42`, and copies files into `dataset/train` and `dataset/val`.
- **Issues:**
  - Duplicates 33 MB of disk space by copying files rather than using PyTorch symlinks, dataset splits, or subset samplers.
  - Lacks a separate held-out `test` split for unbiased multi-model benchmarking.

### 3.5 `src/train_classifier.py`
- **Functionality:** Trains a PyTorch ResNet18 classifier on `dataset/` (binary classification: `busy` vs `free`).
- **Issues:**
  - `parking_classifier.pth` was not generated or saved in repository.
  - Frozen backbone only (`fc` layer trained), no fine-tuning comparison.
  - No early stopping, learning rate scheduler, or validation checkpointing (saves only at epoch 5).
  - No evaluation metrics beyond training/validation accuracy (missing Precision, Recall, F1-score, Confusion Matrix).
  - Hardcoded batch size (32), epochs (5), and learning rate (0.001).

---

## 4. Model Architecture & Weights Review

### 4.1 YOLO Detector
- **Weights:** `yolo26s.pt` (20.4 MB, 11.2M parameters) and `yolo26n.pt` (5.5 MB, 3.2M parameters).
- **Classes:** 80 COCO classes. Relevant filter: `car`, `truck`, `bus`, `motorcycle`.
- **Strengths:** Fast, reliable vehicle bounding box proposals on full image.
- **Weaknesses:** Cannot classify empty slots directly; requires spatial mapping to parking lot coordinates.

### 4.2 ResNet18 Patch Classifier
- **Architecture:** Torchvision ResNet18 with pretrained ImageNet weights, modified head `Linear(512, 2)`.
- **Input:** 224×224 RGB image patch cropped from parking slot coordinates.
- **Output:** Logits for `['busy', 'free']`.
- **Strengths:** Directly inspects slot textures (asphalt vs car roof/hood), robust to occlusion when properly trained.
- **Weaknesses:** Independent patch crops lack global context; sensitive to shadow and perspective distortion without proper augmentation.

---

## 5. Storage & Repository Bloat Analysis

- **Total disk space:** 1,661 MB.
- **Root causes:**
  1. `.venv` contains PyTorch CPU, Torchvision, Ultralytics, OpenCV, and site packages totaling 1,569 MB. This should never be committed to git.
  2. Duplicate dataset in `dataset/` (33.17 MB) when `CNRPark-Patches-150x150/` (33.17 MB) is already present.
  3. YOLO weights (`yolo26s.pt` at 20.4 MB and `yolo26n.pt` at 5.5 MB) should be handled via Git LFS or external download script if tracking via git.

---

## 6. Actionable Upgrade Roadmap

| Phase | Module | Key Deliverables |
| :---: | :--- | :--- |
| **Phase 2** | Computer Vision Engine | Modular pipeline: `preprocessing`, `detection` (YOLO + vectorized NMS), `classification` (ResNet18 fine-tuning/inference), `occupancy` (Intersection over Union, center containment, bbox overlap, 1-to-1 Hungarian/greedy matching, UNCERTAIN thresholding), Hybrid fusion engine. |
| **Phase 3** | Training & Evaluation | Train and save `parking_classifier.pth` with early stopping, scheduler, checkpointing; comprehensive evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix) across YOLO-only, ResNet18-only, and Hybrid models on held-out test set. |
| **Phase 4** | Streamlit Dashboard | 6-page interactive UI (Dark navy, glass panels): Overview, Live Parking Monitor, Image Analysis, Video Analysis, Model Performance, Parking Analytics. |
| **Phase 5** | Testing & Reliability | Pytest test suite covering IoU math, slot matching edge cases, empty/full lots, invalid inputs, missing weights. |
| **Phase 6** | Documentation & Cleanup | Complete professional documentation (`README.md`, `ARCHITECTURE.md`, `MODEL_DOCUMENTATION.md`, `DATASET_DOCUMENTATION.md`, `BENCHMARKS.md`, `CONTRIBUTING.md`), `.gitignore`, size audit. |
| **Phase 7** | Deployment & MLOps | Dockerfile, Streamlit config, GitHub Actions CI workflow, requirements.txt. |
| **Phase 8** | Portfolio & Interview Prep | ATS-optimized resume bullet points, 20 deep-dive technical interview questions with model answers. |

---
**Audit Approved for Execution.**
