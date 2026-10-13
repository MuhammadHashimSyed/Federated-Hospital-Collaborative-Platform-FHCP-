# CHAPTER 5
# SYSTEM IMPLEMENTATION

---

# PART A: FEDERATED TRAINING IMPLEMENTATION

## 5.1 Development Environment Setup

### 5.1.1 Kaggle Notebook Configuration

The federated training pipeline is implemented entirely within a Kaggle Notebook environment, which provides a cloud-based Python execution environment with pre-installed scientific computing libraries.

| Resource | Specification |
|----------|--------------|
| Python Version | 3.12.12 |
| Deep Learning Framework | PyTorch 2.x with CUDA 12 |
| Libraries | `timm`, `albumentations`, `scikit-learn` |
| GPU Accelerator | NVIDIA Tesla T4 (×2) |
| RAM | 16 GB system memory |
| Storage | CheXpert dataset mounted as Kaggle dataset input |

The CheXpert dataset is accessed at runtime via the Kaggle dataset input path `/kaggle/input/chexpert/`, and all model checkpoints are saved to `/kaggle/working/`.

### 5.1.2 GPU Availability and Multi-GPU Setup

At the start of training, the system detects available GPU hardware:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()
```

With two T4 GPUs available, `gpu_count = 2`. When multiple GPUs are detected, PyTorch's `nn.DataParallel` is applied to wrap client models for distributed inference across both GPUs:

```python
if gpu_count > 1:
    parallel_client_model = nn.DataParallel(client_model)
else:
    parallel_client_model = client_model
```

This allows each forward pass to split the input batch across both GPUs, drastically reducing the high compute cost of training transformer architectures. Weight extraction for FedAvg aggregation is performed on the underlying `client_model` (not the DataParallel wrapper), ensuring clean weight extraction.

### 5.1.3 Reproducibility: Seed Locking

Full reproducibility is ensured by fixing all sources of randomness before training:

```python
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(42)
```

Setting `cudnn.deterministic = True` forces CUDA to use deterministic algorithms, while `cudnn.benchmark = False` prevents auto-tuning that selects non-deterministic fast kernels.

---

## 5.2 Dataset Loading and Exploratory Data Analysis

### 5.2.1 Loading the CheXpert CSV

```python
csv_path = '/kaggle/input/chexpert/train.csv'
df = pd.read_csv(csv_path)
print(f"Dataset Loaded. Total X-rays: {len(df)}")
# Output: Dataset Loaded. Total X-rays: 223414
```

The DataFrame contains 19 columns: the image path, patient metadata (Sex, Age, Frontal/Lateral, AP/PA), and 14 binary/uncertain pathology label columns.

### 5.2.2 Disease Distribution Analysis

All labels are first cleaned to handle NaNs (filled as 0) and −1s (uncertainties).

| Rank | Pathology | Positive Count |
|------|-----------|---------------|
| 1 | Support Devices | ~117,080 |
| 2 | Lung Opacity | ~111,179 |
| 3 | Pleural Effusion | ~97,815 |
| ... | ... | ... |
| 14 | Pleural Other | ~6,176 |

### 5.2.3 Class Imbalance Analysis

| Pathology | Imbalance Ratio (neg:pos) |
|-----------|--------------------------|
| Support Devices | 0.9:1 |
| Lung Opacity | 1.0:1 |
| ... | ... |
| Pleural Other | 35.2:1 |

This massive disparity is what necessitates the integration of Focal Loss to force the optimizer to focus on underrepresented classes.

---

## 5.3 Data Preprocessing Implementation

### 5.3.1 Hybrid Uncertainty Strategy

To handle data irregularities specifically tailored for each pathology, the system implements a **Hybrid Uncertainty Strategy**. A predefined mapping applies U-Zeros (replacing −1 with 0) to specific diseases known to cause high false positives, while applying U-Ones to the rest:

```python
UNCERTAIN_STRATEGY = {
    'Consolidation': 0.0,
    'Enlarged Cardiomediastinum': 0.0,
    'Lung Lesion': 0.0,
    'Fracture': 0.0,
    'Pleural Other': 0.0
}

# Apply default U-Ones to others, overriding specified ones to U-Zeros
for disease in all_diseases:
    if disease in UNCERTAIN_STRATEGY:
        clean_df[disease] = clean_df[disease].replace(-1.0, 0.0)
    else:
        clean_df[disease] = clean_df[disease].abs() # Maps -1 to 1
```

Patient IDs are extracted from image paths to facilitate group-aware splitting using `GroupShuffleSplit`, resulting in 171,927 training images and 19,100 validation images completely distinct at the patient level.

---

## 5.4 Custom Dataset and DataLoader Implementation

Instead of standard Torchvision transforms, the system implements complex spatial and pixel-level augmentations using `Albumentations` to prevent the Swin Transformer from overfitting:

```python
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            ], p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
```

The `CheXpertDataset` class wraps this pipeline, converting grayscale images to the 3-channel RGB format expected by the Swin Transformer and casting labels to `float32`.

---

## 5.5 Federated Data Partitioning Implementation

The training data is partitioned into three distinct clients by identifying unique patient IDs, shuffling their order (fixed seed), and splitting the list into three approximately equal groups using `np.array_split`. A dedicated DataLoader is initialized for each client.

| Client | Hospital | Images |
|--------|----------|--------|
| 0 | Hospital A | 56,648 |
| 1 | Hospital B | 57,610 |
| 2 | Hospital C | 57,669 |

---

## 5.6 Model Implementation: Swin Transformer V2

The model initialization utilizes the `timm` library to instantiate the Swin Transformer V2 architecture, overriding the default classifier with a custom 512-dimensional bottleneck head:

```python
import timm

class CheXpertSwin(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        # Load Swin V2 Tiny, ImageNet pre-trained
        self.backbone = timm.create_model('swinv2_tiny_window8_256', pretrained=True, num_classes=0)
        self.feature_dim = self.backbone.num_features
        
        # Custom Classification Head
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
```

---

## 5.7 FedAvg Algorithm Implementation

The `federated_average` function operates over PyTorch state dictionaries. Because client sizes are relatively equal, their AUROC performance weights the averaging:

```python
def federated_average(local_weights, client_sizes):
    total = sum(client_sizes)
    global_weights = copy.deepcopy(local_weights[0])
    for key in global_weights.keys():
        global_weights[key] = sum(
            local_weights[i][key] * (client_sizes[i] / total)
            for i in range(len(local_weights))
        )
    return global_weights
```

---

## 5.8 Federated Training Loop Implementation

### 5.8.1 Focal Loss Implementation

Instead of BCEWithLogits, a custom `FocalLoss` handles the CheXpert class imbalance:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)  # pt is confidence in correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

### 5.8.2 Training Schedule

The federated training is 15 rounds of training, using the AdamW optimizer and Mixed Precision (`torch.cuda.amp.GradScaler`) to speed up transformer training.

**Optimizer Strategy:**
- **Rounds 1–5:** `AdamW` (lr=1e-4) — First fast convergence.
- **Rounds 6–10:** `AdamW` (lr=5e-5) — Smaller tweaks when validation AUROC is no longer changing.
- **Rounds 11–15:** `AdamW` (lr=2e-5) — Final fine-tuning, pushing the model to 81.24% AUROC.

---

# PART B: WEB APPLICATION IMPLEMENTATION

## 5.9 Backend Implementation — Django REST Framework

Now you'll create a backend with Django REST Framework. The sections of the project and its configuration are given below:

### 5.9.1 Project Structure and Configuration

The Django project consists of three apps: `core`, `federated`, and `users`. The backend is accessed by the Streamlit front end via the development server of Django.

**Key Django settings:**
- `AUTH_USER_MODEL = 'users.User'`
- `DEFAULT_AUTHENTICATION_CLASSES = ['rest_framework_simplejwt.authentication.JWTAuthentication']`

### 5.9.2 Custom User Model

The User model is an extension of AbstractUser that includes three new fields: `role` (SUPER_ADMIN / HOSPITAL_CLIENT / DOCTOR), `hospital_name`, a hospital name that is required for the project, and a `linked_hospital` foreign key.

### 5.9.3 Key API View Implementations

This document describes the following key API implementations:
**StartRoundView** — enforces the round lifecycle guardrail, that is, if the round has been already begun, but the guardrail is found to be false, the game will not allow the round to be started again, preventing a new round if the previous is IN_PROGRESS.
**UploadWeightsView** — uploads `.pth` file and metrics, checks HOSPITAL_CLIENT role, and saves a `LocalWeightSubmission`.
**ManualAggregateView** — This command will run FedAvg, aggregate weights and save them to the database, which is triggered by `tasks.py`. Mark it as global model and then mark the round COMPLETED.

### 5.9.4 Live Diagnostic View Implementation

This is the implementation of the Live Diagnostic View. To prevent having to compute the costly Swin Transformer V2 architecture, the `LiveDiagnosticView` stores the pre-computed architecture to avoid massive initialization latency:

**Model caching mechanism:**
```python
_model = None  # class-level cache

@classmethod
def _load_model(cls):
    if cls._model is not None:
        return cls._model, cls._device
    # load SwinTransformer architecture, state_dict, then cache
    cls._model = model
    return cls._model, cls._device
```

**Medical logic for 'No Finding' suppression:**
```python
is_sick = any(
    raw_scores[label] > self.CONFIDENCE_THRESHOLD
    for label in self.LABELS
    if label != 'No Finding'
)
if is_sick:
    raw_scores['No Finding'] = 0.0
```

---

## 5.10 Frontend Implementation — Streamlit

### 5.10.1 Application Structure and Authentication

The JWT and the user role are stored in the main `app.py` and authenticated via `/api/token/` into `st.session_state`.

### 5.10.2 Super Admin Dashboard (Global Analytics)

The Global Analytics page (`2_🌐_Global_Analytics.py`) offers:
- Management of hospitals and rounds.
- Manual FedAvg trigger with real-time AUROC feedback.
- Auditing log tables that are paged.
- Line charts for progression of global AUROC for all 15 rounds completed.

### 5.10.3 Hospital Client Dashboard (Local Dashboard)

The Hospital Client Dashboard (Local Dashboard) will look like the image below. The Local Dashboard page (`1_🏥_Local_Dashboard.py`) offers:
- Round status and `.pth` weight uploader.
- Global model downloader.
- Local training history and physician management tools.

### 5.10.4 Live Diagnostics Page

The Live Diagnostics page (`3_🩺_Live_Diagnostics.py`) offers:
- X-ray drag-and-drop widget.
- Trigger for inferences is firing the 256x256 image to the back end.
- Confidence bars for each clinical report and overall verdict banners for each of the pathologies, and a medical disclaimer.

---

## 5.11 Audit Logging Implementation

Every major system action generates an audit record that is created and associates the actor, action, timestamp, and source IP address.

---

*End of Chapter 5*
