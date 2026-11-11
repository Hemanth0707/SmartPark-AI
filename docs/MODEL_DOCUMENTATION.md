# SmartPark AI — Model Architecture & Decision Theory Documentation

## 1. Perception Models Overview

SmartPark AI utilizes a heterogeneous multi-modal computer vision architecture combining real-time object detection proposals with fine-grained visual patch classification.

| Model | Base Architecture | Parameters | Input Resolution | Primary Task |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s** | CSPDarknet + PAN-FPN | 11.2M | 640×640 px | Vehicle Proposal Generation & Bounding Box Localization |
| **ResNet18** | Residual CNN (18 Layers) | 11.1M | 224×224 px | Binary Slot Patch Classification (`busy` vs `free`) |
| **Decision Engine** | Bipartite Matching + Bayesian Fusion | — | Tabular Geometries | Final Occupancy Resolution & Uncertainty Flagging |

---

## 2. YOLOv8 Detection Architecture

### 2.1 Network Topology
1. **Backbone (CSPDarknet53):** Employs Cross-Stage Partial connections with C2f (Cross-Stage Partial Bottleneck with 2 convolutions) blocks to extract multi-scale contextual features while preserving gradient flow.
2. **Neck (Path Aggregation Network - PAN):** Combines top-down and bottom-up feature pyramids to fuse high-level semantic abstractions with low-level spatial detail.
3. **Head (Decoupled Anchor-Free):** Separates classification and bounding-box regression branches, avoiding anchor tuning biases.

### 2.2 Vehicle Filtering & Non-Maximum Suppression
- Filters raw detections to vehicle categories: `{"car", "truck", "bus", "motorcycle"}`.
- Vectorized NMS suppresses redundant proposals across sliding-window boundaries:
$$\text{IoU}(B_1, B_2) = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)} \le \theta_{\text{NMS}} = 0.45$$

---

## 3. ResNet18 Slot Patch Classifier

### 3.1 Residual Learning Formulation
ResNet18 solves the vanishing gradient degradation problem through identity shortcut mappings:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
Where $\mathbf{x}$ and $\mathbf{y}$ are input and output vectors, and $\mathcal{F}$ is the residual mapping to be learned.

### 3.2 Transfer Learning & Fine-Tuning Regimes
We evaluated two training paradigms on the CNRPark benchmark:
1. **Frozen Backbone:** Feature extractor layers (conv1 through layer4) remain frozen with ImageNet pretrained weights. Only the final linear classification layer $W_{\text{FC}} \in \mathbb{R}^{512 \times 2}$ is optimized.
2. **Fine-Tuning:** Unfreezes residual stage 4 (`layer4`) alongside the classification head, allowing deeper convolutional filters to adapt to parking slot asphalt texture, ambient weather conditions, and partial vehicular shadows.

---

## 4. Spatial Matching & 1-to-1 Bipartite Allocation

To resolve spatial ambiguity and prevent a single vehicle from claiming multiple adjacent slots, we model the matching as a minimum-cost bipartite matching problem.

### 4.1 Affinity Formulation
For slot $S_i$ and detected vehicle $V_j$, the affinity score is defined as:
$$\text{Affinity}(S_i, V_j) = 0.65 \cdot \text{Coverage}(S_i, V_j) + 0.25 \cdot \text{IoU}(S_i, V_j) + 0.10 \cdot \mathbb{I}(\text{Center}(V_j) \in S_i)$$
Where $\text{Coverage}(S_i, V_j) = \frac{\text{Area}(S_i \cap V_j)}{\text{Area}(S_i)}$.

### 4.2 Hungarian Bipartite Assignment
The cost matrix is constructed as $C_{ij} = -\text{Affinity}(S_i, V_j)$. We solve:
$$\min_{\pi} \sum_{i} C_{i, \pi(i)}$$
Using `scipy.optimize.linear_sum_assignment`. This guarantees globally optimal 1-to-1 pairing without greedy race conditions.

---

## 5. Bayesian Hybrid Decision Fusion

When both YOLO and ResNet18 signals are available, the final occupancy state is resolved using the following decision matrix:

| YOLO Status | YOLO Coverage | ResNet $P(\text{busy})$ | Consensus State | Final Confidence |
| :---: | :---: | :---: | :---: | :---: |
| `OCCUPIED` | $\ge 0.30$ | $\ge 0.55$ | **OCCUPIED** | $\min(1.0, \frac{C_{\text{yolo}} + C_{\text{resnet}}}{2} \cdot 1.1)$ |
| `FREE` | $< 0.15$ | $< 0.40$ | **FREE** | $\min(1.0, \frac{C_{\text{yolo}} + (1 - P_{\text{busy}})}{2} \cdot 1.1)$ |
| `OCCUPIED` | $\ge 0.45$ | $< 0.40$ | **OCCUPIED** (YOLO Override) | $C_{\text{yolo}} \cdot 0.75$ |
| `OCCUPIED` | $0.15 \le \text{cov} < 0.45$ | $< 0.40$ | **UNCERTAIN** | $0.50$ |
| `FREE` | $< 0.15$ | $\ge 0.65$ | **OCCUPIED** (ResNet Override) | $C_{\text{resnet}} \cdot 0.80$ |
| Any | $0.12 \le \text{cov} < 0.25$ | $0.40 \le P \le 0.55$ | **UNCERTAIN** | $0.50$ |
