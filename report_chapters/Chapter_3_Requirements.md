# CHAPTER 3
# REQUIREMENT SPECIFICATIONS

## 3.1 Introduction to Requirements Engineering

Requirements engineering is the systematic process of defining, documenting and validating the requirements and resources needed for a system to operate. The use of AI in systems requires not only specifying the software, but also the expectations for data quality, model performance targets, and computational resource constraints. This chapter formally specifies the requirements of the Federated Hospital with respect to its functionality and non-functionality. This will be supported by the Collaborative Platform — both the federated training pipeline and the web-based platform deployment system (along with stakeholder analysis, use cases and system constraints).

---

## 3.2 Functional Requirements

The system must be able to perform the following functions. Every requirement will have an individual identifier and a priority (High / Medium / Low).

### 3.2.1 FR-01: Dataset Loading and Management

| Attribute | Detail |
|-----------|--------|
| ID | FR-01 |
| Priority | High |
| Description | The system shall load the CheXpert dataset from a CSV manifest file containing image paths, patient metadata, and 14 pathological labels. |
| Input | CSV file path (train.csv) |
| Output | Pandas DataFrame with 223,414 records |
| Validation | Total record count printed upon load; path integrity verified |

### 3.2.2 FR-02: Data Preprocessing Pipeline

| Attribute | Detail |
|-----------|--------|
| ID | FR-02 |
| Priority | High |
| Description | The system shall preprocess the dataset by: (a) filtering for frontal-view images only; (b) replacing NaN values with 0 (absent); (c) applying a **Hybrid Uncertainty Strategy** replacing uncertainty labels (−1) with 0 for five specific diseases (U-Zeros) and 1 for the remaining diseases (U-Ones); (d) engineering correct image file paths for the local storage environment. |
| Input | Raw CheXpert DataFrame |
| Output | Cleaned DataFrame with binary-only labels (0 or 1) |
| Validation | No NaN or −1 values in label columns post-processing |

### 3.2.3 FR-03: Patient-Level Data Splitting

| Attribute | Detail |
|-----------|--------|
| ID | FR-03 |
| Priority | High |
| Description | The system shall split the dataset into training (90%) and global validation (10%) sets at the patient level using GroupShuffleSplit, ensuring no patient's images appear in both sets. |
| Input | Cleaned DataFrame, patient ID column |
| Output | train_df (~171,927 images), global_val_df (~19,100 images) |
| Validation | Zero patient ID overlap between training and validation sets confirmed |

### 3.2.4 FR-04: Federated Data Partitioning

| Attribute | Detail |
|-----------|--------|
| ID | FR-04 |
| Priority | High |
| Description | The system shall partition the training dataset among three federated clients by distributing unique patient IDs uniformly across clients, creating three disjoint patient cohorts simulating Hospital A, B, and C. |
| Input | train_df, number of clients (3) |
| Output | Three client-specific DataFrames (~57,000 images each) |
| Validation | Total client image count equals len(train_df); no patient overlap across clients |

### 3.2.5 FR-05: Image Preprocessing and Augmentation

| Attribute | Detail |
|-----------|--------|
| ID | FR-05 |
| Priority | High |
| Description | The system shall apply advanced image transforms via **Albumentations**: (a) aspect-ratio-preserving square padding (SquarePad); (b) resizing to **256×256 pixels**; (c) complex augmentations for training images (ShiftScaleRotate, CLAHE, RandomBrightnessContrast, GaussNoise, CoarseDropout); (d) tensor conversion; (e) ImageNet normalization. |
| Input | Raw PIL image |
| Output | Normalized 3×256×256 float tensor |
| Validation | Tensor shape [B, 3, 256, 256] confirmed; pixel values span expected normalized range |

### 3.2.6 FR-06: Model Initialization and Configuration

| Attribute | Detail |
|-----------|--------|
| ID | FR-06 |
| Priority | High |
| Description | The system shall instantiate a **Swin Transformer V2** model pre-trained on ImageNet1K, replace the classification head with a 512-dimensional linear layer mapping to 14 output nodes (with BatchNorm and Dropout), and configure a **Focal Loss** function (alpha=0.25, gamma=2.0) to handle severe class imbalances. |
| Input | Device configuration, timm library |
| Output | Initialized global model, Focal Loss criterion |
| Validation | Model parameter count verified (~28M for tiny); classifier output dimension = 14 |

### 3.2.7 FR-07: Local Client Training

| Attribute | Detail |
|-----------|--------|
| ID | FR-07 |
| Priority | High |
| Description | In each federated round, each client model shall be initialized with current global weights, trained locally for one epoch, and the local weights returned. Training spans **15 rounds** using the **AdamW** optimizer with a decaying learning rate schedule (from 1e-4 down to 1e-5). |
| Input | Global weights, client DataLoader, optimizer config |
| Output | Updated local model state dictionary, per-class local AUROC |
| Validation | Training loss decreases within epoch; local AUROC is computed on the global validation set |

### 3.2.8 FR-08: FedAvg Global Aggregation

| Attribute | Detail |
|-----------|--------|
| ID | FR-08 |
| Priority | High |
| Description | The system shall aggregate local model weights from all three clients using the FedAvg algorithm — computing a weighted average proportional to each client's validation performance/size — and update the global model with the aggregated weights. |
| Input | List of local state dictionaries, client dataset sizes/AUROC |
| Output | Updated global model state dictionary |
| Validation | Aggregated weights differ from any individual client's weights; global model loads successfully |

### 3.2.9 FR-09: Per-Class Evaluation and Model Checkpointing

| Attribute | Detail |
|-----------|--------|
| ID | FR-09 |
| Priority | High |
| Description | After each federated round, the global model shall be evaluated on the validation set. Per-class AUROC shall be computed for all 14 pathologies and averaged. The best-performing model shall be saved to disk, along with backup checkpoints for all **15 rounds**. |
| Input | Global model, validation DataLoader |
| Output | Per-class and mean AUROC per round; saved .pth checkpoint files |
| Validation | AUROC improves across rounds; checkpoint files exist on disk |

### 3.2.10 FR-10: User Authentication and Authorization

| Attribute | Detail |
|-----------|--------|
| ID | FR-10 |
| Priority | High |
| Description | The web application shall authenticate users via JWT tokens. Three user roles shall be enforced: SUPER_ADMIN (full system access), HOSPITAL_CLIENT (training workflow, doctor management, diagnostic access), and DOCTOR (diagnostic inference only). Every API endpoint shall verify role before processing. |
| Input | Username, password |
| Output | Access token, refresh token |
| Validation | Unauthorized requests receive HTTP 403; tokens expire after configured duration |

### 3.2.11 FR-11: Federated Training Orchestration (Web Application)

| Attribute | Detail |
|-----------|--------|
| ID | FR-11 |
| Priority | High |
| Description | The Super Admin shall be able to start a new federated training round through the web interface. A guardrail shall prevent starting a new round if any previous round is still active. Hospital clients shall upload local model weights (as .pth files) and per-class AUROC metrics via API. |
| Input | Super Admin command; Hospital weight file and metrics JSON |
| Output | New TrainingRound record; LocalWeightSubmission record |
| Validation | Duplicate round start blocked; submission recorded with correct round association |

### 3.2.12 FR-12: FedAvg Web Aggregation

| Attribute | Detail |
|-----------|--------|
| ID | FR-12 |
| Priority | High |
| Description | The Super Admin shall manually trigger FedAvg aggregation through the web interface. The system shall load all submitted local weight files, perform weighted averaging, save the resulting global model, and update the TrainingRound status to COMPLETED. |
| Input | Super Admin aggregation trigger |
| Output | Aggregated global model file; updated round record with global_mean_auroc |
| Validation | Global model file written to media storage; round status transitions correctly |

### 3.2.13 FR-13: Real-Time Diagnostic Inference

| Attribute | Detail |
|-----------|--------|
| ID | FR-13 |
| Priority | High |
| Description | Authenticated users shall upload a chest X-ray image through the web interface. The system shall preprocess the image identically to the training pipeline (SquarePad → Resize **256×256** → ImageNet normalize), perform inference using the cached **Swin Transformer V2** federated model, and return only pathologies with sigmoid probability exceeding 69%. |
| Input | Chest X-ray image file (JPEG/PNG) |
| Output | Dictionary of pathology labels and confidence scores (>69% only), sorted by confidence |
| Validation | Output is empty for clearly normal X-rays; correct pathologies detected in known positive cases |

### 3.2.14 FR-14: Hospital and Doctor Management

| Attribute | Detail |
|-----------|--------|
| ID | FR-14 |
| Priority | Medium |
| Description | The Super Admin shall register and remove hospital node accounts. Each hospital client shall register and remove doctor accounts linked to their node. All management actions shall be audit-logged. |
| Input | Hospital/doctor credentials and metadata |
| Output | Created or deleted user accounts; audit log entries |
| Validation | Duplicate usernames rejected; removed accounts cannot authenticate |

### 3.2.15 FR-15: Audit Logging

| Attribute | Detail |
|-----------|--------|
| ID | FR-15 |
| Priority | High |
| Description | Every significant system action shall be recorded in an AuditLog table: actor identity, action description, timestamp, and source IP address. Super Admins shall be able to view the full audit log through a dedicated API endpoint. |
| Input | Any system event (training, submission, aggregation, model download, management) |
| Output | AuditLog record |
| Validation | All tested actions generate corresponding log entries; log is viewable by Super Admin only |

---

## 3.3 Non-Functional Requirements

| ID | Requirement | Description | Target |
|----|------------|-------------|--------|
| NFR-01 | Accuracy | The federated global model shall achieve a mean AUROC ≥ 0.80 on the 14-class validation set by the final round. | AUROC ≥ 0.80 |
| NFR-02 | Computational Efficiency | Training a single client epoch shall complete within a reasonable time given hardware constraints. Multi-GPU DataParallel shall be used when available. | < 30 min per client epoch on dual T4 GPU |
| NFR-03 | Reproducibility | All random seeds (Python, NumPy, PyTorch) shall be fixed prior to training to ensure fully reproducible results across runs. | Identical AUROC on re-run |
| NFR-04 | Scalability | The federated partitioning and aggregation logic shall be parameterizable (NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS) to support extension without code changes. | Config-driven via constants |
| NFR-05 | Privacy Compliance | Raw patient images shall not be shared across simulated federated clients; only model weight tensors are exchanged. | Zero raw data exchange |
| NFR-06 | API Response Time | All non-inference API endpoints (authentication, round management, logs) shall respond within 2 seconds. | ≤ 2 seconds |
| NFR-07 | Inference Latency | Diagnostic inference using the Swin Transformer V2 model shall complete within 10 seconds. | ≤ 10 seconds |
| NFR-08 | Security | All API endpoints shall require valid JWT authentication. Role-based authorization shall be enforced per endpoint. | No unauthorized access |
| NFR-09 | Usability | The Streamlit interface shall be navigable without technical training for medical and administrative users. | Role-appropriate dashboard per user type |

---

## 3.4 Stakeholder Analysis

### 3.4.1 Primary Stakeholders

| Stakeholder | Role | Interest |
|-------------|------|---------|
| Super Administrator | Manages the federated platform, starts rounds, triggers aggregation | System reliability; complete visibility into training and logs |
| Hospital IT Administrators | Deploy local training, upload weights, manage doctors | Privacy compliance; easy workflow integration |
| Doctors / Radiologists | Use the live diagnostic tool | High diagnostic accuracy; low false negative rate |
| Patients | Subjects of the imaging and diagnosis | Accurate diagnosis; confidentiality of medical records |

### 3.4.2 Secondary Stakeholders

| Stakeholder | Role | Interest |
|-------------|------|---------|
| Medical AI Researchers | Extend and benchmark the system | Reproducibility; open evaluation methodology |
| Healthcare Regulators | Ensure compliance with data protection law | HIPAA / GDPR compliance; audit trail |
| University Evaluation Committee | Assess the academic merit of the project | Technical depth; sound methodology |

---

## 3.5 Use Cases

### 3.5.1 Use Case 1 — Federated Training Round Lifecycle

| Field | Detail |
|-------|--------|
| Actor | Super Admin, Hospital Clients (3 nodes) |
| Precondition | No active round exists; all hospital nodes are registered |
| Main Flow | 1. Super Admin initiates Round N via web dashboard. 2. Hospital clients train locally and upload .pth weights + per-class AUROC via API. 3. Super Admin triggers FedAvg aggregation. 4. System averages weights, saves global model, marks round COMPLETED. 5. Hospital clients download updated global model for next round. |
| Postcondition | Global model updated; round AUROC recorded; all actions audit-logged |
| Exception | New round blocked if previous round not yet aggregated |

### 3.5.2 Use Case 2 — Real-Time Chest X-Ray Diagnosis

| Field | Detail |
|-------|--------|
| Actor | Doctor |
| Precondition | Doctor is authenticated with a valid JWT token |
| Main Flow | 1. Doctor navigates to Live Diagnostics page. 2. Doctor uploads a chest X-ray image. 3. System preprocesses image (256x256) and performs Swin Transformer V2 inference. 4. System filters output to pathologies with confidence > 69%. 5. Clinical report displayed with color-coded severity and overall verdict. |
| Postcondition | Diagnostic results displayed; inference action logged |
| Exception | No findings above threshold → report shows clear/normal result |

### 3.5.3 Use Case 3 — Hospital Node Registration

| Field | Detail |
|-------|--------|
| Actor | Super Admin |
| Precondition | Super Admin is authenticated |
| Main Flow | 1. Admin enters hospital name, username, and password. 2. System creates HOSPITAL_CLIENT user account. 3. Action is audit-logged. 4. Hospital can now authenticate and participate in training rounds. |
| Postcondition | Hospital node registered and active |
| Exception | Duplicate username rejected with HTTP 400 |

### 3.5.4 Use Case 4 — Doctor Account Management

| Field | Detail |
|-------|--------|
| Actor | Hospital Client |
| Precondition | Hospital client is authenticated |
| Main Flow | 1. Hospital client enters doctor name, username, password. 2. System creates DOCTOR user linked to the hospital. 3. Doctor can log in and access the diagnostic tool. 4. Hospital client can remove doctors no longer affiliated. |
| Postcondition | Doctor account created/removed; action audit-logged |

---

## 3.6 System Constraints

| Constraint | Type | Description |
|-----------|------|-------------|
| Hardware | Physical | Federated training limited to Kaggle environment with 2× NVIDIA Tesla T4 GPUs (16 GB VRAM total), 16 GB RAM |
| Time | Computational | Each federated round involves 3 full client training epochs totaling ~3-4 hours due to transformer complexity |
| Dataset | Legal/Access | CheXpert test set ground-truth labels are not publicly available; evaluation performed on training-split validation set |
| Framework | Software | Training requires Python 3.12, PyTorch, `timm` (Swin Transformer), and `albumentations` in Kaggle environment |
| Federation | Architectural | Federation is simulated on a single node; real distributed networking is not implemented |
| Privacy | Regulatory | No differential privacy mechanism is implemented; weight-based attacks are not defended against |
| Database | Deployment | SQLite database used; suitable for development but not production-scale concurrency |

---

## 3.7 Input, Output, and Processing Flow Definition

**System Inputs:**

| Input | Format | Source |
|-------|--------|--------|
| CheXpert training CSV | .csv (223,414 rows × 19 columns) | Kaggle CheXpert dataset |
| Chest X-ray images (training) | .jpg (variable resolution) | CheXpert image directory |
| Hyperparameters | Python constants (NUM_CLIENTS=3, ROUNDS=15, BATCH=32) | Source code |
| ImageNet pre-trained weights | .pth (Swin Transformer V2) | `timm` library |
| Local model weights (web) | .pth file uploaded via API | Hospital client |
| Chest X-ray image (inference) | .jpg / .png uploaded by doctor | Web interface |

**System Outputs:**

| Output | Format | Description |
|--------|--------|-------------|
| Best global model | .pth checkpoint | Highest mean AUROC model across all rounds |
| Round backup models | .pth × 15 | One checkpoint per federated round |
| Per-round global AUROC | Console log + web dashboard | Mean and class-wise AUROC after each round |
| Per-node local AUROC | Stored in LocalWeightSubmission | Per-class AUROC from each hospital per round |
| Diagnostic report | JSON API + Streamlit UI | Pathologies above 69% confidence, sorted by probability |
| Audit log | Database records | Timestamped record of every significant system action |

---

*End of Chapter 3*
