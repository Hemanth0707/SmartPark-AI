# SmartPark AI — Resume Optimization & Technical Interview Prep

---

## 1. Professional Project Description

**SmartPark AI** is an enterprise-grade, end-to-end computer vision and spatial intelligence platform designed for automated parking occupancy monitoring, vehicle localization, and urban telematics. Addressing the common pitfalls of single-modality vision systems—such as boundary false-negatives, multi-stall vehicle double allocation, and severe environmental domain shifts—SmartPark AI integrates a **dual-stage Bayesian hybrid perception pipeline**:
- **Stage 1 (Object Detection):** YOLOv8s vehicle localization combined with vectorized Non-Maximum Suppression (NMS) and sliding-window aerial tiling.
- **Stage 2 (Spatial Optimization):** 1-to-1 minimum-cost bipartite matching via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), mathematically guaranteeing that no single detected vehicle can falsely claim multiple parking bays.
- **Stage 3 (Surface Classification):** A fine-tuned ResNet18 classifier evaluating cropped slot ROI patches for surface texture discrimination (asphalt vs vehicular metal/windshield).
- **Stage 4 (Decision Consensus):** Bayesian fusion with uncertainty quantification, distinguishing confirmed `FREE`, confirmed `OCCUPIED`, and `UNCERTAIN` boundary edge-cases.
- **Stage 5 (Telematics Dashboard):** A 6-page interactive Streamlit dashboard featuring live CCTV HUD overlay, temporal video subsampling (reducing redundant inference by 93%), real-time slot patch inspection, and CSV/JSON reporting.

---

## 2. ATS-Friendly Professional Resume Bullet Points

- **Bullet Point 1 (Computer Vision & Dual-Stage Deep Learning):**  
  *Architected an end-to-end computer vision parking occupancy pipeline in Python, coupling YOLOv8s vehicle detection with fine-tuned ResNet18 patch classification (achieving 89.75% zero-leakage cross-camera test accuracy and 99.86% unseen-slot validation accuracy across the CNRPark benchmark).*

- **Bullet Point 2 (Spatial Optimization & Algorithmic Engineering):**  
  *Eliminated vehicle double-allocation and center-point boundary failure modes by implementing 1-to-1 minimum-cost bipartite matching (Hungarian algorithm) and custom IoU/slot intersection coverage geometry, resolving multi-stall overlaps and edge-case occlusions.*

- **Bullet Point 3 (Bayesian Fusion, Video Processing & Full-Stack Telematics):**  
  *Engineered a Bayesian decision consensus engine with uncertainty quantification and built a 6-page interactive Streamlit telemetry dashboard with temporal video subsampling (cutting inference overhead by 93%), verified by a 23-test automated pytest suite, Dockerized, and integrated with GitHub Actions CI.*

---

## 3. Technical Architecture & Dataflow

```
Input Frame (Image / Video Stream)
              │
              ▼
   Image Preprocessing & Bounds Clipping
              │
      ┌───────┴────────────────────────┐
      ▼                                ▼
[Stage 1: YOLOv8s Detection]   [Stage 2: ROI Extraction]
  - Vectorized NMS (IoU 0.45)    - 224x224 Patch Normalization
  - Overlap de-duplication       - Domain-invariant Augmentation
  - Vehicle Bounding Boxes                     │
      │                                        ▼
      ▼                                [Stage 3: ResNet18]
[Hungarian 1-to-1 Matching]      - Fine-tuned Layer4 + FC Head
  - Linear Sum Assignment        - P(Busy) vs P(Free) Probabilities
  - Slot Intersection Coverage                 │
      │                                        │
      └──────────────────┬─────────────────────┘
                         ▼
           [Stage 4: Bayesian Decision Engine]
             - Probabilistic Consensus Fusion
             - Uncertainty Boundary Quantification
                         │
                         ▼
        [Stage 5: Visualization & Analytics HUD]
          - FREE / OCCUPIED / UNCERTAIN States
          - Real-Time Streamlit Glassmorphic Dashboard
          - Temporal Video Subsampling & Telemetry Export
```

---

## 4. Key Algorithms & Core Purpose

| Algorithm / Module | Mathematical / Architectural Formulation | System Purpose |
| :--- | :--- | :--- |
| **YOLOv8s** | One-stage CNN anchor-free object detector with CSPDarknet backbone | Detects vehicles (`car`, `truck`, `bus`, `motorcycle`) across wide-angle surveillance scenes. |
| **Vectorized NMS** | $\text{IoU} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)} > 0.45$ suppression | Eliminates duplicate multi-scale vehicle proposals across sliding tiles. |
| **Hungarian Bipartite Matching** | $\min \sum_{i} \sum_{j} C_{i,j} X_{i,j}$ via `linear_sum_assignment` | Enforces strictly 1-to-1 vehicle-to-slot assignment, preventing double-allocation. |
| **Slot Intersection Coverage** | $\text{Coverage} = \frac{\text{Area}(V \cap S)}{\text{Area}(S)}$ | Resolves vehicle aspect-ratio distortion where large vans cover compact stalls. |
| **ResNet18 Fine-Tuning** | Residual skip connections ($F(x) + x$) with unchoked `layer4` + linear head | Discriminates fine-grained surface textures (asphalt road stripes vs car roofs). |
| **Bayesian Fusion** | $P(\text{Occ} \mid Y, R) = \frac{P(Y \mid \text{Occ}) P(R \mid \text{Occ}) P_0}{\sum P}$ | Fuses spatial overlap and patch classification into an ambiguity-aware decision. |
| **Temporal Subsampling** | Evaluation stride $k$ (e.g. 1 frame every 15 frames) | Eliminates 93% of redundant inference load on static parking CCTV streams. |

---

## 5. Model Performance & Evaluation Conditions

Rigorous machine learning engineering requires distinguishing **leaked random validation** from **true zero-leakage generalization**:

| Evaluation Protocol | Train Split | Validation / Test Split | Accuracy | Macro F1 | Key Scientific Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Split (Leaked Baseline)** | 80% shuffled patches (Cam A + B) | 20% shuffled patches | **98.61%** | 0.9860 | Inflated by temporal correlation (5-minute video frames) and identical stall geometry shared across splits. |
| **Intra-Scene Slot-Disjoint Split** | 39 slots of Camera A (`id % 4 != 0`) | 12 unseen slots of Camera A (`id % 4 == 0`) | **99.86%** | **0.9986** | Proves the classifier learns the scene texture, stall lines, and lighting of the training environment near-perfectly. |
| **Strict Zero-Leakage Cross-Camera** | Camera A (all slots) | **Camera B (Untouched 2,400 test set)** | **89.75%** | **0.8975** | True generalization performance across an unseen physical location, angle, and day without data contamination. |

### Verified Cross-Camera Confusion Matrix (Camera B, 2,400 samples):
- **True Busy, Predicted Busy:** 1,059
- **True Busy, Predicted Free (False Negatives):** 141 (concentrated in slots occluded by tree foliage, e.g. Slot 18)
- **True Free, Predicted Free:** 1,095
- **True Free, Predicted Busy (False Positives):** 105 (concentrated in dark asphalt slots under tree canopy shadows, e.g. Slot 32)
- **Macro Precision:** 89.79% | **Macro Recall:** 89.75% | **Weighted F1:** 89.75%

---

## 6. Limitations & Future Roadmap

1. **Environmental Domain Gap (Canopy Shadows & Foliage):**  
   *Current Limitation:* Training solely on Camera A exposes the model to domain shift when testing on Camera B, where dense tree canopies cast dark shadows on empty asphalt and tree leaves occlude parked cars.  
   *Planned Solution:* Deploy Domain Adversarial Training (DANN) or contrastive self-supervised pre-training (SimCLR/MoCo) across unlabelled multi-camera surveillance feeds.
2. **Static Camera Assumption:**  
   *Current Limitation:* Bounding coordinates for parking slots are calibrated to fixed-angle CCTV cameras.  
   *Planned Solution:* Implement zero-shot parking bay segmentation using Segment Anything (SAM) or Mask R-CNN to auto-discover slot geometries upon camera deployment.
3. **Edge Optimization:**  
   *Current Limitation:* CPU inference executes in ~512ms (ResNet18) and ~875ms (YOLOv8s).  
   *Planned Solution:* Quantize weights to INT8 via TensorRT or ONNX Runtime for 15–30 FPS real-time performance on edge hardware (NVIDIA Jetson Orin).

---

## 7. Top 10 Technical Interview Questions & Answers

### Q1: Why combine YOLOv8 with ResNet18 instead of using YOLO alone?
**Answer:** YOLO is an object detector optimized to output vehicle bounding box proposals and confidence scores from general scene context. However, it does not explicitly understand whether a pre-demarcated 2D geometric slot is legally occupied, especially when vehicles park slightly across painted lines. Conversely, ResNet18 operates directly on cropped slot surface patches to classify surface texture (asphalt vs vehicular roof/metal). Combining both via Bayesian fusion provides dual confirmation: YOLO confirms vehicle presence and dimensions, while ResNet18 validates patch texture. When signals disagree, the system tags the slot as `UNCERTAIN` instead of making a high-confidence mistake.

### Q2: What was the limitation of the original center-point containment method?
**Answer:** The original implementation calculated `(vx1+vx2)//2, (vy1+vy2)//2` and checked `sx1 <= cx <= sx2 and sy1 <= cy <= sy2`. If a vehicle parked at a slight angle or was 1 pixel over the boundary, its geometric center fell outside the bounding box, causing the slot to register as `FREE` despite being 90% blocked. On our reference scene, this caused an 80% false-negative rate (10 vehicles detected, only 2 slots marked occupied). Replacing it with intersection coverage $\frac{\text{Area}(V \cap S)}{\text{Area}(S)}$ and IoU resolved all occupied stalls cleanly.

### Q3: How do you prevent a single large vehicle (like a truck or bus) from occupying multiple adjacent parking spaces in software?
**Answer:** We formulate vehicle-to-slot matching as a global minimum-cost bipartite assignment problem solved via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). Given $M$ vehicles and $N$ slots, we construct an affinity matrix based on spatial coverage and IoU. Minimizing negative affinity ensures strict 1-to-1 assignment: each detected vehicle can claim at most one parking stall (the one with highest affinity), preventing double-counting.

### Q4: Why did fine-tuning ResNet18 improve accuracy from 76.2% to 89.75% over a frozen backbone?
**Answer:** A frozen backbone relies solely on generic ImageNet features (edges, textures, everyday objects) and only trains the final linear classification layer. By unfreezing `layer4` (the deepest residual block), the convolutional filters adapt directly to domain-specific visual features of the CNRPark benchmark—specifically parking asphalt textures, painted white lines, vehicular windshield reflections, and complex cast shadows.

### Q5: How did you identify and address data leakage during evaluation?
**Answer:** When analyzing the CNRPark-Patches dataset, we discovered that consecutive images were sampled at 5-minute intervals from the same surveillance camera. A random 80/20 train/validation split leaked near-identical video frames and identical parking stall geometry into both splits, artificially inflating validation accuracy to 98.61%. To establish genuine generalization, we designed two zero-leakage splits: (1) an intra-camera split where 25% of slot IDs were held out completely, achieving 99.86% validation accuracy, and (2) a strict cross-camera test set (Camera B) which remained untouched during all training and hyperparameter tuning, achieving a true cross-camera accuracy of 89.75%.

### Q6: How does temporal subsampling in the video pipeline optimize CPU/GPU utilization?
**Answer:** CCTV video streams typically capture 25–30 frames per second. However, vehicles do not park or depart in 33 milliseconds. Running full YOLO and ResNet inference on every frame wastes computational resources. By introducing a configurable frame stride (e.g., evaluating 1 frame every 15 frames), we reduce inference load by ~93% while still capturing state changes with sub-second responsiveness (every 0.5s at 30 FPS).

### Q7: What is the difference between IoU and Slot Intersection Coverage?
**Answer:** $\text{IoU} = \frac{\text{Area}(V \cap S)}{\text{Area}(V \cup S)}$, whereas $\text{Coverage} = \frac{\text{Area}(V \cap S)}{\text{Area}(S)}$. If a large SUV or delivery van parks in a compact parking stall, the vehicle area is much larger than the slot. IoU will be artificially penalized (e.g., 0.35) even though the vehicle covers 100% of the stall. Using slot intersection coverage ensures the stall is recognized as 100% occupied.

### Q8: How are `UNCERTAIN` states determined?
**Answer:** A stall is flagged as `UNCERTAIN` in two scenarios:
1. **Spatial Ambiguity:** Vehicle overlap is between 12% and 25% (marginal encroachment or adjacent vehicle parked on the line).
2. **Model Divergence:** YOLO detects a vehicle with moderate overlap, but ResNet predicts $P(\text{busy}) < 0.40$, or vice versa.

### Q9: How does the system handle an invalid or corrupted image input?
**Answer:** In `clip_box_to_image` and `extract_slot_patch`, we validate that `image is not None` and `image.size > 0`, raising a `ValueError`. If bounding box coordinates exceed image dimensions, `clip_box_to_image` clips coordinates to `[0..W, 0..H]` and enforces minimum dimensions of 1×1 pixel to avoid OpenCV assertion crashes.

### Q10: How did you test the system for production reliability?
**Answer:** We engineered an automated test suite with 23 pytest test cases covering:
- Mathematical edge cases: identical boxes, inverted coordinates, zero-overlap boxes.
- Vectorized NMS suppression and deduplication.
- 1-to-1 Hungarian matching and multi-slot overlap prevention.
- Empty lots and fully occupied lots.
- Dashboard export serialization (CSV/JSON).
- Error handling for missing weights and invalid inputs.
- Imbalanced classification metrics.
