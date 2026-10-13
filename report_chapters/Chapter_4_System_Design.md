# CHAPTER 4
# SYSTEM DESIGN AND ARCHITECTURE

## 4.1 Overview of System Architecture

The course provides an introduction to the architecture of a system. The Federated Hospital Collaborative Platform is divided into two broad subsystems: The Federated Training Subsystem is run sequentially on Kaggle GPU infrastructure, and the Web Application Subsystem serves as the RESTful web interface (deployed as a local Django + Streamlit stack). They execute a full federated learning lifecycle, from the raw data all the way through to insights, and from training to deployment in the clinic.

**The system consists of six modules interconnected to each other:**

1. **Data Pipeline Module** — Splits and partitions the CheXpert data, then loads the training, validation, and federated subsets of clients, using complex Albumentations augmentations.
2. **Model Module** — Defines a custom 512-dimensional Swin Transformer V2 multi-label classification head and sets up the Focal Loss function.
3. **Federated Learning Module** — Orchestrates the FedAvg training loop across three synthetic patients to represent hospital clients over 15 rounds of communication.
4. **Evaluation Module** — Globally and per-node AUROC evaluation (per class) after each round, with checkpointing.
5. **Backend API Module** — A Django REST Framework server that exposes 15+ endpoints for authentication, training workflow, inference, and management.
6. **Frontend Module** — A Streamlit multi-page application offering role-specific solutions, providing Super Admins, Hospital Clients and Doctors with dashboards to start with.

---

## 4.2 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  KAGGLE TRAINING ENVIRONMENT                │
│                                                             │
│  CheXpert Dataset ──► Data Pipeline ──► Federated Clients  │
│                                    A / B / C (Swin Trans.) │
│                                         │                   │
│                                    FedAvg Server            │
│                                         │                   │
│                                   Global Model              │
│                               (backup_round_N.pth)          │
└───────────────────────────────────────┬─────────────────────┘
                                        │
                              Model weights transferred
                                        │
┌───────────────────────────────────────▼─────────────────────┐
│                   WEB APPLICATION LAYER                      │
│                                                             │
│  ┌──────────────────┐         ┌──────────────────────────┐  │
│  │  Django Backend  │ ◄─────► │  Streamlit Frontend      │  │
│  │  (REST API)      │         │  (Multi-page App)        │  │
│  │                  │         │                          │  │
│  │  • JWT Auth      │         │  • Super Admin Dashboard │  │
│  │  • Training API  │         │  • Hospital Dashboard    │  │
│  │  • Inference API │         │  • Live Diagnostics      │  │
│  │  • Audit Logs    │         │  • Analytics Charts      │  │
│  │  • SQLite DB     │         │                          │  │
│  └──────────────────┘         └──────────────────────────┘  │
│                                                             │
│  Users: Super Admin │ Hospital Clients │ Doctors            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4.3 Data Pipeline Design

### 4.3.1 CSV Parsing and Path Engineering

The images are included in the CheXpert dataset in a CSV manifest (train.csv) that includes image paths with reference to the Stanford file system. In the Kaggle deployment environment, images live on different paths and/or at different directory levels. All paths in the string are replaced by the replacement string in the pipeline to match the CSV-specified paths, which means that any CSV-specified paths are matched to the actual local file system. During the training process, the CheXpertDataset class can find all of the images.

### 4.3.2 Patient-Level Grouped Data Splitting

One of the basic design principles is to avoid data leakage, which is the unintentional transmission of data so there is no issue with images from the same patient being used for the train and validation sets. Since patients often have multiple CXR studies in CheXpert, a naive image-level random split may result in different studies of the same patient appearing on both sides of the split, inflating validation performance.

The pipeline creates a `patient_id` field from each image path and performs **GroupShuffleSplit** using scikit-learn with groups as patient IDs. This guarantees a patient-disjoint split: all images from a given patient are only present in either the training or validation set, but not both.

The split ratio is 90% training / 10% global validation:
- **Training set:** 171,927 images (~58,716 unique patients)
- **Validation set:** 19,100 images (~6,524 unique patients)
- **Patient overlap:** 0 (verified by assertion)

### 4.3.3 Federated Data Partitioning Strategy

After splitting at the patient level, three federated training sets are created for clients:

1. Get rid of any duplicate patient IDs in train_df.
2. Shuffle the patient list randomly (to make it reproducible, use a fixed seed).
3. Split the shuffled patient list into 3 parts with `np.array_split`.
4. Create three DataFrames based on the `train_df` for each patient segment.

| Client | Hospital | Approx. Images |
|--------|----------|---------------|
| Client 0 | Hospital A | 56,648 |
| Client 1 | Hospital B | 57,610 |
| Client 2 | Hospital C | 57,669 |

### 4.3.4 Data Augmentation and Preprocessing Design

These are very difficult to train with the transformer architecture being this large and to avoid overfitting, the pipeline uses the **Albumentations** library to perform complex spatial and pixel level transformations:

**Training Transform:**
```
Resize(256×256)      → Scale to model input size
HorizontalFlip       → p=0.5
ShiftScaleRotate     → Shift ±10%, Scale ±10%, Rotate ±10° (p=0.5)
OneOf                → CLAHE or RandomBrightnessContrast (p=0.5)
GaussNoise           → Inject variance to simulate machine noise (p=0.2)
CoarseDropout        → Random rectangular block dropping (max 8 holes)
Normalize            → ImageNet statistics
ToTensorV2           → Convert to PyTorch float tensor [0,1]
```

**Validation Transform:** Identical except all random augmentations (flips, rotations, noise, dropout) are excluded so that a deterministic evaluation is made.

**Inference Transform (Web Application):** The validation transform is the same as the inference transform (Web application). After square padding, resizing to 256x256, conversion to tensor, and normalization, it was ensured that the training and deployment pipelines are the same for the preprocessing pipeline.

---

## 4.4 Model Architecture Design

### 4.4.1 Swin Transformer V2 — Internal Structure

The system utilizes the `swinv2_tiny_window8_256` architecture. Unlike CNNs, the Swin Transformer partitions the input image into non-overlapping patches, passing them through hierarchical transformer blocks using Shifted Window based Self-Attention.

```
Input (3×256×256)
    │
    ▼
Patch Partition (4×4 patches) → Linear Embedding
    │
    ▼
Stage 1: Swin Transformer Block (Window Attention + Shifted Window Attention)
    │
Patch Merging (downsampling)
    │
    ▼
Stage 2: Swin Transformer Block
    │
Patch Merging
    │
    ▼
Stage 3: Swin Transformer Block
    │
Patch Merging
    │
    ▼
Stage 4: Swin Transformer Block
    │
    ▼
LayerNorm + Global Average Pooling  →  Feature vector
    │
    ▼
Custom Classification Head          →  14 logits
    │
    ▼
Sigmoid (element-wise)              →  14 probabilities ∈ [0,1]
```

### 4.4.2 Shifted Window Attention and Post-Norm

The standard Vision Transformer computes self-attention globally, resulting in quadratic complexity. Swin Transformer V2 computes self-attention locally within 8x8 windows. In the next layer, these windows are shifted by 4 pixels, allowing the network to build cross-window connections. 

Additionally, Swin V2 utilizes **Post-Normalization** (placing LayerNorm after the attention blocks) and **Scaled Cosine Attention** instead of dot-product attention, which significantly stabilizes training for large medical datasets.

### 4.4.3 Custom Classification Head for 14 Labels

The base `timm` model outputs a feature vector from its global average pooling layer. The default 1000-class ImageNet head is replaced with a custom sequential block optimized for multi-label classification:

```python
self.head = nn.Sequential(
    nn.Linear(self.feature_dim, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, Config.NUM_CLASSES)
)
```

This 512-dimensional bottleneck with Batch Normalization and Dropout provides regularization before the final 14-node output layer, helping prevent the transformer from overfitting to the training set.

### 4.4.4 Sigmoid Activation for Multi-Label Output

Unlike single-label classification (which uses Softmax), multi-label classification requires independent probability estimates for each class. The sigmoid function is applied element-wise to the 14 output logits, yielding 14 independent probabilities in [0,1]. During training, this is handled via Focal Loss. During inference, sigmoid is applied explicitly to obtain the output probabilities.

---

## 4.5 Federated Learning Architecture

### 4.5.1 Server–Client Communication Model

The federated system follows a synchronous star topology: a central server communicates with all clients simultaneously, waiting for all clients to complete local training before performing aggregation.

```
                    ┌─────────────────┐
                    │  Central Server │
                    │  (FedAvg)       │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
     ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
     │  Hospital A │  │  Hospital B │  │  Hospital C │
     │  Client 0   │  │  Client 1   │  │  Client 2   │
     │  56,648 img │  │  57,610 img │  │  57,669 img │
     └─────────────┘  └─────────────┘  └─────────────┘
```

### 4.5.2 FedAvg Aggregation Design

The aggregation function iterates over all model parameters layer-by-layer. Weights are computed based on each client's global validation AUROC performance:

```
For each parameter key in model.state_dict():
    weighted_sum = 0
    For each client k:
        weighted_sum += W_k[key] × client_weight_k
    W_global[key] = weighted_sum
```

This handles all parameter types — transformer weights, linear layers, and layer norm parameters — uniformly through PyTorch's state dictionary interface.

### 4.5.3 Round-Based Training Protocol

The training runs for 15 rounds using the AdamW optimizer with weight decay (0.01) and a decaying learning rate schedule:

| Rounds | Action | Optimizer | Learning Rate |
|--------|--------|-----------|---------------|
| 1–5 | Local train → FedAvg → Validate | AdamW | 1e-4 |
| 6–10 | Repeat with updated global weights | AdamW | 5e-5 |
| 11–15 | Repeat with updated global weights | AdamW | 2e-5 |
| Best | Save highest AUROC model | — | — |

---

## 4.6 Loss Function Design

### 4.6.1 Focal Loss for Class Imbalance

While the Support Devices appear in 117,000 images, Chest CTs are distributed across all other classes, with only 6,100 patients having the diagnosis Pleural Other. Under these circumstances, Standard Binary Cross-Entropy fails because its training is dominated by the overwhelming number of easy negatives, causing conditions to be overpowered by the loss gradient.

This is addressed by the pipeline using **Focal Loss** where alpha = 0.25 and gamma = 2.0:

```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```

Where:
- `p_t` is the estimated probability of the model for the correct class.
- `α_t` (0.25) is the balancing factor for positive/negative examples.
- `γ` (2.0) is the focusing parameter.

### 4.6.2 The Modulating Factor

The important modulating factor is `(1 - p_t)^γ`. If the model is extremely sure and correct (e.g., `p_t = 0.95`), the modulating factor approaches zero `(1 - 0.95)^2 = 0.0025`, an effective way to remove this simple example from the loss calculation. On the contrary, whenever the model has a hard minority-class example (e.g., `p_t = 0.2`), the modulating factor is large `(1 - 0.2)^2 = 0.64`, forcing the optimizer to pay a lot of attention to learning it.

---

## 4.7 Web Application Architecture

### 4.7.1 Backend Architecture — Django REST Framework

It is developed in Python with the Django 4.x web framework and Django REST Framework (DRF) for the backend. The project is structured as follows:

```
backend/
├── core/              # Django project settings, URL routing
├── federated/         # Main application: models, views, serializers, tasks
│   ├── models.py      # TrainingRound, LocalWeightSubmission, AuditLog
│   ├── views.py       # 15+ API view classes
│   ├── serializers.py # DRF serializers
│   ├── tasks.py       # FedAvg aggregation logic
│   └── urls.py        # Endpoint routing
├── users/             # Custom user model with roles
│   ├── models.py      # User: role, hospital_name, linked_hospital
│   └── views.py       # Login/token views
├── manage.py
└── db.sqlite3
```

**Key design decisions:**
- **Custom User Model** with `role` field (SUPER_ADMIN / HOSPITAL_CLIENT / DOCTOR) and a foreign key to associate a doctor with a hospital (`linked_hospital`).
- **Class-based API Views** for fine-grained permission control per endpoint.
- **Model caching** in `LiveDiagnosticView`: the Swin Transformer V2 model is loaded and stored as a class-level attribute (`_model`) to prevent repeated initialization latency.

### 4.7.2 Database Schema Design

**TrainingRound Model:**
| Field | Type | Description |
|-------|------|-------------|
| round_number | IntegerField | Federated round identifier |
| status | CharField | PENDING / IN_PROGRESS / AGGREGATING / COMPLETED |
| global_mean_auroc | FloatField | Mean AUROC after FedAvg |
| global_model_file | FileField | Path to aggregated .pth file |

**LocalWeightSubmission Model:**
| Field | Type | Description |
|-------|------|-------------|
| hospital | ForeignKey(User) | Submitting hospital node |
| weight_file | FileField | Uploaded .pth weights |
| local_mean_auroc | FloatField | Hospital's local validation AUROC |

**AuditLog Model:**
| Field | Type | Description |
|-------|------|-------------|
| actor | ForeignKey(User) | User who performed the action |
| action | TextField | Human-readable description |
| ip_address | GenericIPAddressField | Source IP of the request |

### 4.7.3 API Endpoints Overview

| Endpoint | Method | Role | Description |
|----------|--------|------|-------------|
| /api/token/ | POST | All | Obtain JWT access + refresh tokens |
| /api/federated/round/start/ | POST | SUPER_ADMIN | Start a new training round |
| /api/federated/weights/upload/ | POST | HOSPITAL_CLIENT | Upload local weights + AUROC |
| /api/federated/aggregate/ | POST | SUPER_ADMIN | Trigger FedAvg aggregation |
| /api/federated/diagnostics/predict/ | POST | All auth | Live X-ray diagnostic inference |
| /api/federated/hospitals/ | GET | SUPER_ADMIN | List registered hospitals |
| /api/federated/logs/ | GET | SUPER_ADMIN | View full audit log |

### 4.7.4 Frontend Architecture — Streamlit

The frontend is a multi-page Streamlit application enforcing role-based access checks.

```
frontend/
├── app.py                        # Login page and session initialization
└── pages/
    ├── 1_🏥_Local_Dashboard.py   # Hospital client view (training, doctors)
    ├── 2_🌐_Global_Analytics.py  # Super admin view (nodes, rounds, logs)
    └── 3_🩺_Live_Diagnostics.py  # Doctor / all roles: X-ray inference
```

---

## 4.8 Real-Time Inference Pipeline Design

```
HTTP POST /api/federated/diagnostics/predict/
    │
    ▼
JWT Authentication Check
    │
    ▼
Image received from request.FILES['image']
    │
    ▼
Model Loading (cached after first request):
  Swin Transformer V2 loaded from fyp_model/ directory
  Weights restored from .pth file
  Model set to eval() mode
    │
    ▼
Image Preprocessing:
  Open image (PIL) → convert to RGB
  SquarePad → Resize(256×256) → ToTensor → ImageNet Normalize
  Add batch dimension → unsqueeze(0)
    │
    ▼
Inference (torch.no_grad()):
  Forward pass → 14 raw logits
  torch.sigmoid() → 14 probabilities
    │
    ▼
Medical Logic:
  If any non-'No Finding' pathology > 69% → suppress 'No Finding'
  Filter: keep only labels where probability > 69%
  Sort descending by confidence
    │
    ▼
HTTP 200 Response:
  { "predictions": { "Pathology": confidence, ... },
    "threshold": 0.69,
    "total_findings": N }
```

---

## 4.9 Evaluation Module Design

### 4.9.1 AUROC Computation

This process is repeated in each round of federation and the global model is tested on the validation set (`model.eval()` and `torch.no_grad()`). The transformed output probabilities (sigmoid) are displayed and the ground-truth labels are gathered.

For each class `c`, `roc_auc_score`, as provided by scikit-learn, is used:
```python
AUROC(c) = roc_auc_score(all_targets[:, c], all_preds[:, c])
```
The overall AUROC is the average of all valid classes.

### 4.9.2 Per-Node Local AUROC

Every hospital user rates the model that was trained at their own hospital on the common global validation set prior to a weight submission. This is serialized to a local AUROC dictionary JSON file containing admin analytics information, in addition to the `.pth` file.

---

## 4.10 Security Architecture

### 4.10.1 JWT Authentication

It uses `djangorestframework-simplejwt` as the stateless authentication system. Clients are granted access and refresh tokens and pass the access token in the `Authorization: Bearer` header.

### 4.10.2 Role-Based Access Control

Permission is checked at the view level. In each API view, it performs a check on the role of the user. If the role does not match the required permission (either SUPER_ADMIN, HOSPITAL_CLIENT, or DOCTOR), it returns HTTP 403 Forbidden.

### 4.10.3 Audit Logging System

For every important action, an `AuditLog` is created which captures who did what, the action string, timestamp, and source IP address – providing a tamper-evident record for regulatory compliance and review.

---

## 4.11 Technology Stack Justification

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Deep Learning | PyTorch 2.x | Native support for CUDA, DataParallel, and model serialization |
| Model Architecture | `timm` Library | Provides robust, pre-trained Swin Transformer V2 architectures |
| Image Processing | Albumentations | Highly optimized, complex data augmentation for medical images |
| Training Platform | Kaggle Notebooks | Free dual T4 GPU access; CheXpert dataset pre-mounted |
| Backend Framework | Django REST Framework | Robust auth, serialization, and file upload handling |
| Frontend Framework | Streamlit | Rapid Python-native web UI; excellent for data science dashboards |
| Evaluation | scikit-learn | Reliable roc_auc_score implementation |

---

## 4.12 Data Flow Diagrams

### 4.12.1 Level 0 — Context Diagram

```
                    ┌──────────────────────────┐
CheXpert Data  ───► │                          │ ───► Global Model (.pth)
Hyperparameters ──► │  Federated Hospital      │ ───► Per-Round AUROC
X-Ray Images   ───► │  Collaborative Platform  │ ───► Diagnostic Reports
User Credentials──► │                          │ ───► Audit Logs
                    └──────────────────────────┘
```

### 4.12.2 Level 1 — System Processes

```
[CheXpert CSV] ──► [1. Data Pipeline] ──► [Client Datasets A/B/C]
                                                    │
                                    ┌───────────────▼───────────────┐
                                    │  2. Federated Training Loop   │
                                    │  Round 1–15:                  │
                                    │  Local Train → FedAvg → Eval  │
                                    └───────────────┬───────────────┘
                                                    │
                                          [Global Model .pth]
                                                    │
                                    ┌───────────────▼───────────────┐
                                    │  3. Web Application           │
                                    │  Backend: serve model,        │
                                    │  manage training workflow     │
                                    │  Frontend: dashboards, UI     │
                                    └───────────────┬───────────────┘
                                                    │
                                    [Clinical Diagnostic Reports]
                                    [Audit Logs] [Analytics Charts]
```

---

*End of Chapter 4*
