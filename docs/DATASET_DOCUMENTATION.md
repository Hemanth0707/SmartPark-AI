# SmartPark AI — Dataset Documentation & Data Pipeline

## 1. Dataset Overview

SmartPark AI utilizes the **CNRPark-Patches-150x150** dataset, an established real-world parking lot benchmark collected by the Institute of Information Science and Technologies (ISTI) of the National Research Council (CNR) of Italy in Pisa.

- **Total Images:** 12,584 patch images.
- **Resolution:** 150 × 150 pixels (pre-processed and resized to 224 × 224 for ResNet18).
- **Format:** RGB JPEG.
- **Task:** Binary classification (`busy` vs `free`).

---

## 2. Camera Perspectives & Environmental Variations

The dataset captures continuous real-world conditions from two distinct surveillance perspectives:

| Camera Set | Total Images | Busy Patches | Free Patches | Perspective |
| :---: | :---: | :---: | :---: | :--- |
| **Camera A** | 6,171 | 3,621 (58.7%) | 2,550 (41.3%) | High-angle perspective view overlooking main diagonal parking stalls. |
| **Camera B** | 6,413 | 4,781 (74.5%) | 1,632 (25.5%) | Longitudinal view along access lane; significant vehicular occlusion and perspective distortion. |
| **Total** | **12,584** | **8,402 (66.8%)** | **4,182 (33.2%)** | Mixed sunny, overcast, and rainy lighting conditions. |

---

## 3. Data Splitting & Class Balancing

The dataset preparation pipeline (`src/prepare_dataset.py`) segregates patches into an 80/20 train/validation distribution using fixed deterministic random seed `42`:

- **Training Set (`dataset/train`):**
  - `busy`: 6,720 patches
  - `free`: 3,345 patches
  - Total: 10,065 patches
- **Validation Set (`dataset/val`):**
  - `busy`: 1,682 patches
  - `free`: 837 patches
  - Total: 2,519 patches

### Class Imbalance Handling
Because `busy` samples outnumber `free` samples by approximately 2:1, our training pipeline applies class weighting to the cross-entropy loss function:
$$w_c = \frac{N_{\text{total}}}{C \cdot N_c}$$
Where $C = 2$, ensuring the model is not biased toward predicting occupied slots.

---

## 4. Full Parking Scene Reference Data

- **Image:** `parking.jpg` (415 × 612 × 3 BGR image).
- **Slot Annotations:** `parking_slots.json` (22 parking slots with 2D bounding boxes `[x1, y1, x2, y2]`).
- **Ground Truth Reference:**
  - Total parking stalls: 22
  - Actually parked vehicles: 4
  - Actually empty spaces: 18

---

## 5. Dataset Attribution & License

- **Authors:** P. R. de Almeida, L. S. Oliveira, A. S. Britto Jr., E. J. Silva Jr., A. L. Koerich.
- **Publication:** *"PKLot – A robust dataset for parking lot availability categorization"*, Expert Systems with Applications, 2015.
- **Dataset Source:** CNRPark & PKLot repositories.
