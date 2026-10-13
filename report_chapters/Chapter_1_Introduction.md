# CHAPTER 1
# INTRODUCTION

## 1.1 Background and Motivation

### 1.1.1 Artificial Intelligence in Modern Healthcare

May we begin with a review of the state of Artificial Intelligence (AI) in healthcare today. AI, or Artificial Intelligence, is one of the most disruptive technologies in modern medicine. The convergence of large-scale digitized medical data, high-performance computing, and advances in deep learning has made AI systems capable of complex diagnostic tasks. The accuracy of diagnostic tasks is comparable to — and sometimes better than — trained clinicians. In AI, within the subdomain of Deep Learning, Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) have shown outstanding visual pattern recognition capabilities that can be directly applied to medical imaging.

With a shortage of healthcare workers worldwide, the need for these developments is even greater. The WHO estimates that there is a shortfall of more than 18 million health workers by 2030, and radiologists will be one of the most desperately needed specialists. AI-powered diagnostic tools are not intended to supplant clinical judgment but rather to supplement it — providing the clinician with faster, more consistent, and more scalable support, especially for resource-limited healthcare settings.

### 1.1.2 The Clinical Significance of Chest X-ray Imaging

The Chest X-ray (CXR) is the most common medical imaging procedure performed. In the world, an estimated 2 billion CXRs are acquired yearly. Its clinical utility is extensively used in a range of thoracic diseases: pneumonia, pulmonary edema, pleural effusion, cardiomegaly, pneumothorax, atelectasis, lung lesions, and fractures. CXRs are regularly ordered in EDs, ICUs, and outpatient clinics for their accessibility, speed, and moderately low price.

Although it is common, the correct interpretation of a chest X-ray is a cognitively challenging task. Pathologies may be subtle, overlapping, and very variable in presentation from one patient to another. The consequences of misinterpretation are high — a missed pneumothorax can escalate to a life-threatening emergency, and an overlooked pulmonary lesion may represent early signs of a cancer that can be treated.

### 1.1.3 Limitations of Manual Radiological Diagnosis

Interpretation of chest X-rays is known to have shortcomings:

- **Inter-observer Variability:** Agreement rate between radiologists for some pathologies is not high. The quality of images varies from person to person, and the level of agreement is only as high as 60-70%.
- **Diagnostic Fatigue:** Decrease in performance as a function of workload and time, raising the likelihood of oversight errors.
- **Turnaround Delays:** Transferring the image to the RIS/PACS and the time it reaches the physician for interpretation can take time; it can take hours to days to get the acquisition and the radiologist's report.
- **Geographical Inequity:** Poorly publicized and underserved areas of the country are poorly served by specialist radiological expertise compared to central areas.
- **Data Privacy Constraints:** Patient information is protected by stringent laws and regulations, such as HIPAA and GDPR, which limit what can be done with data in health IT systems. This creates a fundamental barrier to the sharing and centralization of imaging data across institutions for co-developing large scale AI systems.

### 1.1.4 The Promise of Automated Disease Detection

Since Rajpurkar et al. (2017) showed that a DenseNet121 model (CheXNet) was able to detect pneumonia with an accuracy surpassing that of radiologists, the field has developed quickly. Modern architectures, especially Vision Transformers, can now get competitive AUROC scores for a wide variety of thoracic diseases. Their benefits are that they are consistent (no fatigue), have speed (millisecond inference), and offer scalability (a single model can serve thousands of endpoints). But for deployment in the real world, it is necessary to overcome not only the accuracy problem, but also the privacy-preserving collaborative training problem — an important reason why Federated Learning is being used in this project.

**Gap in Previous Work:** Initial experiments showed that federated training is feasible for chest X-ray classification. Early models, however, were just research artifacts — there was no usable interface for clinical application, no mechanism for hospital nodes to discover one another, no protocol for offering a clinical service, and limited exploration of state-of-the-art transformer architectures in a managed workflow.

**This Project's Extension:** We solve the Research Gap and the Deployment Gap in this project by:
1. Re-training the federated model for **15 communication rounds** using a powerful **Swin Transformer V2** architecture, achieving a high mean AUROC of **81.24%** and a global accuracy of **87.25%**.
2. Introducing full **per-class AUROC tracking** for the global model after every round.
3. Reporting **per-hospital node local AUROC** submitted alongside weight updates.
4. **Deploying a complete web-based platform** — a Django REST API backend and Streamlit frontend — that enables real-time chest X-ray diagnostic inference using the trained Swin Transformer V2 federated model.
5. Implementing a **multi-role user management system** (Super Admin, Hospital Client, Doctor) with JWT authentication.
6. Providing **comprehensive audit logging** for regulatory compliance.

---

## 1.2 Introduction to the CheXpert Dataset

### 1.2.1 Dataset Overview and Scale

CheXpert is the dataset of interest. CheXpert (Chest Expert) is a publicly available big data benchmark set created by the Stanford Machine Learning Group in 2019. It consists of 224,316 chest X-rays of 65,240 unique patients, with labels automatically extracted from radiology reports using an NLP-based labeler. This project uses only frontal view images, as per conventional practice, resulting in approximately 191,000 training images.

### 1.2.2 The 14 Pathological Labels

Each image is annotated for the presence or absence of 14 thoracic conditions:

| No. | Pathology | Description |
|-----|-----------|-------------|
| 1 | No Finding | No observable abnormality |
| 2 | Enlarged Cardiomediastinum | Widened mediastinal silhouette |
| 3 | Cardiomegaly | Enlarged cardiac silhouette |
| 4 | Lung Opacity | Non-specific increased opacity |
| 5 | Lung Lesion | Focal parenchymal abnormality |
| 6 | Edema | Fluid accumulation in lung tissue |
| 7 | Consolidation | Alveolar air replaced by fluid/cells |
| 8 | Pneumonia | Infective consolidation |
| 9 | Atelectasis | Lung tissue collapse |
| 10 | Pneumothorax | Air in pleural space |
| 11 | Pleural Effusion | Fluid in pleural cavity |
| 12 | Pleural Other | Other pleural abnormalities |
| 13 | Fracture | Visible bone fracture |
| 14 | Support Devices | Medical lines/tubes visible |

The task is multi-label binary classification — a single image may simultaneously exhibit multiple co-occurring pathologies. The dataset is severely imbalanced: positive case counts range from 117,080 (Support Devices) to 6,176 (Pleural Other), with negative-to-positive ratios ranging from 0.9:1 to 35.2:1.

### 1.2.3 The Uncertainty Labeling Problem

The CheXpert NLP labeler gives a value of 1 (present), 0 (absent), or −1 (uncertain). When the source report is ambiguous, for example, with the words "possible" or "cannot exclude," there are uncertain labels. This project takes a **Hybrid Strategy** to address the challenges of these ambiguous labels:

- **U-Zeros Strategy:** A U-Zeros strategy was developed for diseases where ambiguous terms are substituted with 0 for specificity. This includes consolidation, enlarged cardiomediastinum, lung lesion, fracture, and pleural other, which are conditions that are difficult to distinguish in the absence of precise clinical history.
- **U-Ones Strategy:** Applied to all other pathologies, converting −1 values to 1. The principle behind the motivation is that missing a positive result is usually worse than getting a false alarm.

No label (NaN value) is considered as 0 for all classes.

---

## 1.3 Federated Learning in Medical AI

### 1.3.1 What is Federated Learning?

Federated Learning (FL) is a distributed machine learning paradigm where multiple clients (such as hospitals) work together to train a shared global model, sharing only local models, not raw data. The trainings take place locally on the private dataset of each client, and only the weights of the models are sent to the central server. These are consolidated in a global model by the server. This cycle continues around and around across multiple communication rounds.

This project employs the core aggregation algorithm **Federated Averaging (FedAvg)**, proposed by McMahan et al. (2017):

```
W_global = Σ_k (n_k / N) × W_k
```

Where W_k are local weights of client k, n_k is the number of samples in the client's data set, and N is the total number of samples across all clients.

### 1.3.2 Why Federated Learning for Healthcare?

Healthcare institutions have legal requirements which are quite strict — HIPAA (USA), GDPR (EU), and the same kinds of frameworks around the world — that don't allow patients' information to be shared elsewhere. Federated Learning will solve this problem: raw patient images are stored in hospitals' servers; only model weights are transmitted. This allows multi-institutional development of AI without privacy violations. It enables the model to be exposed to a wide range of patient clusters, which aids generalization, and is capable of being structurally aligned with healthcare data governance frameworks.

---

## 1.4 Problem Statement

This project is designed to solve two related problems:

**Challenge 1 — Diagnostic Accuracy:** Manual CXR interpretation is slow, variable, and inaccessible at scale. An automatic system with the ability to detect 14 thoracic co-occurring diseases was developed. To achieve high accurate classification of pathologies, advanced architectures with high capability are required, making significant strides toward reducing class size inequity.

**Challenge 2 — Privacy-Preserving Training with Practical Deployment:** Developing high-performing AI requires massive and diverse multi-institutional datasets, but patient data privacy is a concern. Centralized data pooling is not allowed by regulations. Furthermore, a model even when locked in a research notebook cannot be used by the clinical user — a complete web-based platform is required.

This project covers training, privacy, and deployment in all three dimensions. Using a state-of-the-art **Swin Transformer V2**, we implement a Federated Learning framework on CheXpert with three distributed hospitals, and deploy the trained model in a web application that is available to Administrators, hospital nodes, and clinical practitioners.

---

## 1.5 Aims and Objectives

**Primary Aim:** To design, implement, and deploy a comprehensive federated privacy-preserving machine learning system for the automated multi-label classification of chest pathologies, delivering high diagnostic accuracy and a production-ready web application for real-time clinical inference.

**Specific Objectives:**

**Training Objectives:**
1. Do exploratory data analysis on CheXpert to describe the distribution of the labels and class imbalance.
2. Develop a preprocessing pipeline which involves filtering by the front end and uncertainty handling by hybrid (U-Zeros/U-Ones), keeping aspect ratios through resizing to 256x256, and advanced data augmentation using the Albumentations library.
3. Prevent data leakage with patient-level GroupShuffleSplit.
4. Divide training patients into three disjoint patient groups representing a federated hospital environment.
5. Fine-tune Swin Transformer V2 (ImageNet pre-trained) with a custom 512-dimensional 14-class head and Focal Loss to address the severe class imbalance.
6. Add FedAvg aggregation of three federated clients.
7. Conduct **15 rounds** of training and testing of the global model, reporting per class AUROC for each.
8. Report per-hospital node AUROC at each submission, alongside model weights.

**Web Application Objectives:**
9. Create and execute a Django REST API, with role-based endpoints.
10. Create role-specific dashboards (Super Admin, Hospital Client, Doctor) on Streamlit.
11. Implement JWT based authorization and multi-role authentication.
12. Allow hospital node registration, management, and doctor account management.
13. Build real-time diagnostic inference based on the trained federated Swin Transformer V2 model.
14. Implement a federated training orchestration workflow (start round, upload weights, call FedAvg, download global model).
15. Design an extensive audit logging mechanism for regulatory requirements.

---

## 1.6 Scope and Limitations

**Scope:**

| Dimension | Detail |
|-----------|--------|
| Dataset | CheXpert frontal-view radiographs (~191,000 training images), 14 pathologies |
| Model | Swin Transformer V2 (`swinv2_tiny_window8_256`) with ImageNet pretraining |
| Federated Setting | 3 simulated hospital clients, 15 communication rounds, FedAvg |
| Platform | Kaggle Notebook (training), dual NVIDIA Tesla T4 GPUs |
| Web Application | Django REST Framework backend, Streamlit frontend, SQLite database |
| Authentication | JWT tokens, three user roles (Super Admin, Hospital Client, Doctor) |
| Inference | Real-time chest X-ray diagnosis via Swin Transformer V2 |
| Audit | Comprehensive action logging for every system event |

**Limitations:**

| Limitation | Description |
|------------|-------------|
| Simulated federation | Runs on a single physical Kaggle node; no real distributed networking |
| IID distribution | Patients split uniformly; real-world data is non-IID |
| No differential privacy | Weight-based privacy attacks are not defended against |
| No public test set | Official CheXpert test labels are unavailable; validation uses training-split held-out set |
| Development environment | SQLite database; not load-balanced for production-scale concurrency |

---

## 1.7 Significance of the Project

- **Clinical:** Offers automated support for the 14-class diagnosis of CXR in a fast and consistent manner, with a web browser to access real-time inference for credentialed doctors using a good performing transformer model.
- **Technical:** Provides a complete end-to-end federated deep learning pipeline — from the training of advanced Vision Transformers with Focal Loss on Kaggle GPUs through REST API deployment.
- **Privacy:** Inspires by showcasing the achievement of competitive diagnostic AI without centralizing patient data, and that the trained model can be served via a platform structurally HIPAA and GDPR compliant.
- **Academic:** The academic contribution combines federated learning, vision transformers, focal loss, and robust web application development, integrating real-time inference into a seamless, operational-grade system.

---

## 1.8 Methodology Overview

The project is broken into six phases:

1. **Data Analysis:** Load CheXpert CSV; analyze distribution and imbalance ratios of 14 labels.
2. **Preprocessing & Partitioning:** Filter frontal images; apply hybrid uncertainty handling; split patient-level; partition into three federated client datasets; apply Albumentations for augmentation.
3. **Federated Training (15 Rounds):** Run FedAvg across 3 hospital clients — local training, global aggregation, per-class AUROC evaluation, per-node AUROC reporting — for 15 communication rounds utilizing Swin Transformer V2 and Focal Loss.
4. **Backend Development:** Design and implement Django REST API with 15+ endpoints, JWT authentication, three-tier user roles, database models, and real-time inference integration for the Swin Transformer V2 model.
5. **Frontend Development:** Build Streamlit multi-page application with role-specific dashboards, file upload capabilities, analytics charts, and the live diagnostic report interface.
6. **Testing & Evaluation:** Validate the training pipeline (AUROC progression, convergence), the web application (API testing, authentication, inference), and the end-to-end federated workflow, ultimately achieving 81.24% AUROC.

---

## 1.9 Report Organization

| Chapter | Title | Content |
|---------|-------|---------|
| 2 | Literature Review | Prior works, architectures, FL state of the art, web-based medical AI |
| 3 | Requirement Specifications | Functional/non-functional requirements, stakeholders, use cases |
| 4 | System Design & Architecture | Pipeline design, Swin Transformer design, web app architecture, DFD |
| 5 | System Implementation | 5A: Federated training implementation; 5B: Web application implementation |
| 6 | Testing & Evaluation | Per-class AUROC results, per-node results, 15-round analysis, benchmarks |
| 7 | Conclusion | Achievements, contributions, future work |

---

*End of Chapter 1*
