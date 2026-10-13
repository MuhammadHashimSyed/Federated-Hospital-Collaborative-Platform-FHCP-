# 🌐 Federated Hospital Collaborative Platform (FHCP)
### *Secure, Privacy-Preserving Collaborative Medical Diagnosis Network*

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-REST%20Framework-green.svg)](https://www.djangoproject.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20MIT-lightgrey.svg)]()

---

## 📌 Overview

**FHCP** is a decentralized healthcare platform designed to train deep learning models collaboratively across multiple hospital nodes without exchanging sensitive patient radiographs or Electronic Health Records (EHR). 

By adopting **Federated Learning (FedAvg)**, each participating hospital trains models on its own private clinical datasets. Only model weight updates (`.pth` state dicts) and validation metrics are transmitted to the aggregation server, maintaining strict data privacy and HIPAA compliance.

---

## 🌟 Key Capabilities

* 🔒 **Privacy-Preserving Federated Learning**: Implements the **Federated Averaging (FedAvg)** algorithm across distributed hospital clients. Raw patient data never leaves hospital premises.
* 👥 **Three-Tier Role-Based Access Control (RBAC)**:
  * 🛡️ **Super Admin**: Monitor global platform health, initialize training rounds, trigger weight aggregation, review audit trails, and manage hospital registrations.
  * 🏥 **Hospital Client**: Train local models on local datasets, submit model weights with AUROC verification, review submission history, and onboard physicians.
  * 🩺 **Doctor / Physician**: Run AI-assisted clinical diagnosis on Chest X-Rays with real-time inference across 14 chest pathologies.
* 🩺 **Deep Learning Diagnostics**: Powered by **DenseNet-121** and **Swin Transformer** architectures trained on the CheXpert benchmark, identifying pathologies including *Cardiomegaly*, *Edema*, *Consolidation*, *Pneumothorax*, *Pleural Effusion*, and more.
* 📊 **Interactive Visual Analytics**: Modern Streamlit dashboards featuring interactive Plotly performance metrics, convergence curves, and audit logs.

---

## 🏗️ System Architecture

```
FHCP/
├── backend/                  # Django REST Framework Backend
│   ├── core/                 # Settings, Celery configuration, and URL routing
│   ├── federated/            # FL logic (FedAvg aggregation, model management, tasks)
│   ├── users/                # Custom User model (Super Admin, Hospital Client, Doctor)
│   ├── media/                # Stored global and local model weight checkpoints
│   └── manage.py             # Django management CLI
├── frontend/                 # Streamlit Multi-Page Web Interface
│   ├── app.py                # Platform gateway / landing page
│   └── pages/
│       ├── 1_🏥_Local_Dashboard.py    # Hospital client training & submission portal
│       ├── 2_🌍_Global_Analytics.py   # Admin management & aggregation dashboard
│       └── 3_🩺_Live_Diagnostics.py   # Physician inference & diagnostic report portal
├── dummy_dataset/            # Sample radiographs & CheXpert metadata for local testing
├── report_chapters/          # Complete system architecture and research documentation
├── requirements.txt          # Python project dependencies
├── run_backend.bat           # One-click launcher for Django backend
└── run_frontend.bat          # One-click launcher for Streamlit frontend
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+ (tested on Python 3.12 and 3.13)
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/MuhammadHashimSyed/Federated-Hospital-Collaborative-Platform-FHCP-.git
cd Federated-Hospital-Collaborative-Platform-FHCP-
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Running the Platform

You can launch both services using the provided batch scripts or terminal commands:

### Step 1: Start Backend API (Port 8000)

**Option A (Windows Double-Click):**
* Double-click `run_backend.bat`

**Option B (Terminal):**
```bash
cd backend
python manage.py runserver 127.0.0.1:8000
```
*API Base URL: `http://127.0.0.1:8000/api/`*

### Step 2: Start Frontend Application (Port 8501)

**Option A (Windows Double-Click):**
* Double-click `run_frontend.bat`

**Option B (Terminal):**
```bash
cd frontend
streamlit run app.py
```
*Web Portal URL: `http://localhost:8501`*

---

## 🔑 Pre-configured Accounts (Demo)

The database includes ready-to-test accounts for all three platform roles:

| Portal | Username | Password | Role | Features |
| :--- | :--- | :--- | :--- | :--- |
| **🛡️ Admin Portal** | `admin` | `admin123` | `SUPER_ADMIN` | Start rounds, trigger FedAvg, register hospitals, view system logs |
| **🏥 Hospital Portal** | `PIMS` | `123` | `HOSPITAL_CLIENT` | Train locally, upload weights, doctor management |
| **🏥 Hospital Portal** | `shifa` | `123` | `HOSPITAL_CLIENT` | Secondary hospital node participation |
| **🩺 Diagnostic Portal** | `doc` | `123` | `DOCTOR` | Upload Chest X-Rays and generate diagnostic reports |

---

## 🔬 Federated Learning Workflow

1. **Round Initialization**: Super Admin opens Round $T$ through Global Analytics.
2. **Local Training**: Hospital nodes (`PIMS`, `shifa`) train locally on local patient datasets.
3. **Weight Submission**: Hospitals upload `.pth` state dictionaries along with validation AUROC.
4. **Aggregation**: Super Admin triggers Federated Averaging (FedAvg):
   $$\mathbf{w}_{global}^{T} = \sum_{k=1}^{K} \frac{n_k}{n} \mathbf{w}_k^{T}$$
5. **Distribution**: The aggregated global model is saved and made available for hospital download for Round $T+1$.
6. **Clinical Inference**: Physicians run live inference using the federated global model directly in the diagnostic portal.

---

## 🛠️ Technology Stack

* **Deep Learning**: PyTorch, Torchvision, DenseNet-121, Swin Transformer V2
* **Backend**: Django 5 / 6, Django REST Framework, SimpleJWT (Token Authentication), Celery
* **Frontend**: Streamlit, Plotly Express, Pandas, Pillow
* **Database**: SQLite3 (Development) / PostgreSQL compatible

---

## 📄 License & Attribution

This project is developed as a Final Year Capstone Project (FYP) on **Federated Learning in Healthcare**.  
Developed by **Muhammad Hashim Syed**.