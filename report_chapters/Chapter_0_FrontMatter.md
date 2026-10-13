# FINAL YEAR PROJECT REPORT

# FEDERATED HOSPITAL COLLABORATIVE PLATFORM: A WEB-BASED IMPLEMENTATION FOR PRIVACY-PRESERVING CHEST X-RAY DIAGNOSIS

---

**Submitted By:**
- Arbab Hussain (01-134222-173)
- Muhammad Hashim Syed (01-134222-095)

**Supervisor:**
- Dr. Sumaira Kausar

**Department of Computer Science**
**Bahria University Islamabad**

**April 2026**

---

## CERTIFICATE

We accept the work contained in the report titled "Federated Hospital Collaborative Platform: A Web-Based Implementation for Privacy-Preserving Chest X-Ray Diagnosis", written by Mr. Arbab Hussain and Mr. Muhammad Hashim Syed as a confirmation to the required standard for the partial fulfillment of the degree of Bachelor of Science in Computer Science.

**Approved by:**

Supervisor: Dr. Sumaira Kausar _______________

Internal Examiner: _______________

External Examiner: _______________

Project Coordinator: _______________

Head of Department: _______________

---

## ABSTRACT

Interpretation of the chest X-ray is one of the most common and important diagnostic tests that are conducted. In clinical radiology, the most challenging tasks. Lack of specialist radiologists, and combined with the ever-rising number of chest radiographic examinations, emphasizes the need of accurate, scalable, and private automated diagnostic systems that are urgently needed. Most deep learning algorithms that have been used to detect diseases from chest X-rays are based on the premise of: centralized training — an approach that is at odds with the strict patient data privacy Healthcare regulations across the globe, such as HIPAA and GDPR. 

Taking on both diagnostic and privacy issues at once in this project, Designing, implementing and deploying a full web-based Federated Learning (FL) system. The paper proposes a novel framework for automated multi-label classification on the CTSC challenge including 14 thoracic pathologies. a large-scale CheXpert dataset of 224,316 chest radiographs from 65,240 unique patients. This system is a simulation of three hospitals having federated environment with patient data. Only model weight updates are sent to a central point; it stays local per institution. aggregation server. 

The feature is extracted using the Swin Transformer V2 architecture pre-trained using ImageNet. The model is modified to contain an adapted 512 dimensional multi-label classification head and is extracted using a predefined extractor. The Federated Averaging (FedAvg) algorithm orchestrates collaborative model training across three patient-disjoint client cohorts of approximately 57,000 images each. Severe class imbalance — with negative-to-positive ratios ranging from 0.9:1 to 35.2:1 — is addressed through **Focal Loss** to explicitly focus the optimizer on hard, minority-class examples. A complex `Albumentations` preprocessing pipeline and a Hybrid Uncertainty strategy ensure robust data augmentation, while patient-level GroupShuffleSplit eliminates data leakage between training and validation sets.

The global federated model is trained over **15 communication rounds** on Kaggle's GPU infrastructure (2× NVIDIA Tesla T4), achieving a mean AUROC of 0.7582 after Round 1 and improving consistently to a **best mean AUROC of 0.8124 and an overall global accuracy of 87.25% after Round 15** across all 14 pathological classes. A 3-phase decaying learning rate schedule using the `AdamW` optimizer effectively forces the transformer to securely converge over the federated rounds. Full per-class AUROC is reported for every round, and per-hospital node local AUROC is tracked at each submission.

What's most important, this project provides a fully functional web application developed using Django REST Framework (backend) and Streamlit (frontend). The system enables:

- The federated model trained with 69% accuracy score to make real-time diagnostic inference. confidence threshold
- Multi role authentication (Super Admin, Hospital Client, Doctor) using JWT
- Dynamically register and remove node from hospital
- Under hospital supervision, doctor account management.
- A federated training orchestrator, which initiates training round, uploads weights, initiates FedAvg. download global model
- Full audit reporting for compliance reasons
- Per-round AUROC metrics for the local area of every node across the entire network

The deployed system shows that federated learning is feasible in real-world scenarios. World healthcare environments, from research prototypes to production. ready healthcare AI systems.

**Keywords:** Federated Learning, Medical Image Classification, Chest X-Ray, Deep Learning, Swin Transformer V2, Privacy-Preserving AI, Healthcare AI, Web Application, Django, Real-time Inference, CheXpert

---

## ACKNOWLEDGEMENTS

All praise is due to Allah, the Most Gracious and the Most Merciful, for granting the knowledge, patience, and strength to complete this work. Only Allah, the Most Generous and Most Merciful, has given the all these "kinds" to us. With wisdom, tolerance and courage to finish the task. 

We would like to thank our project supervisor Dr. Sumaira Kausar for their time and time again guidance, constructive feedback and encouragement, During the course of this project. Their knowledge in the area of machine learning and medical AI This work was greatly influenced by the direction and quality of the work of this company. 

We would like to express our thanks to the faculty members of the Department of Computer Science at Bahria University Islamabad for offering the academic environment and resources to To do such research. 

This work was made possible by the Stanford Machine Learning Group, thanks to their support in this matter. The CheXpert dataset is available publicly, and the computational is provided by Kaggle. Large scale federated training was made possible by the infrastructure (GPU resources). 

Last but not least, we are so blessed to have the support of our family and friends at all times, patience, and encouragement throughout our studies.

**Arbab Hussain**
**Muhammad Hashim Syed**
E-8, Islamabad, Pakistan
April 2026

---

## TABLE OF CONTENTS

**ABSTRACT** .................................................... i

**ACKNOWLEDGEMENTS** ............................................ ii

**LIST OF ABBREVIATIONS** ....................................... iii

**LIST OF TABLES** .............................................. iv

**LIST OF FIGURES** ............................................. v

---

### CHAPTER 1: INTRODUCTION ..................................... 1

1.1 Background and Motivation

&nbsp;&nbsp;&nbsp;&nbsp;1.1.1 Artificial Intelligence in Modern Healthcare

&nbsp;&nbsp;&nbsp;&nbsp;1.1.2 The Clinical Significance of Chest X-ray Imaging

&nbsp;&nbsp;&nbsp;&nbsp;1.1.3 Limitations of Manual Radiological Diagnosis

&nbsp;&nbsp;&nbsp;&nbsp;1.1.4 The Promise of Automated Disease Detection

1.2 Introduction to the CheXpert Dataset

&nbsp;&nbsp;&nbsp;&nbsp;1.2.1 Dataset Overview and Scale

&nbsp;&nbsp;&nbsp;&nbsp;1.2.2 The 14 Pathological Labels

&nbsp;&nbsp;&nbsp;&nbsp;1.2.3 The Uncertainty Labeling Problem

1.3 Federated Learning in Medical AI

&nbsp;&nbsp;&nbsp;&nbsp;1.3.1 What is Federated Learning?

&nbsp;&nbsp;&nbsp;&nbsp;1.3.2 Why Federated Learning for Healthcare?

1.4 Problem Statement

1.5 Aims and Objectives

1.6 Scope and Limitations

1.7 Significance of the Project

1.8 Methodology Overview

1.9 Report Organization

---

### CHAPTER 2: LITERATURE REVIEW

2.1 Overview of Deep Learning in Medical Imaging

2.2 Existing Work on Chest X-ray Disease Detection

2.3 Vision Transformers (ViT) in Medical Image Classification

2.4 Federated Learning: State of the Art

2.5 Handling Class Imbalance with Focal Loss

2.6 Web-Based Medical AI Systems

2.7 Research Gaps Identified

2.8 How This Project Addresses the Gaps

---

### CHAPTER 3: REQUIREMENT SPECIFICATIONS

3.1 Introduction to Requirements Engineering

3.2 Functional Requirements (FR-01 through FR-15)

3.3 Non-Functional Requirements

3.4 Stakeholder Analysis

3.5 Use Cases

3.6 System Constraints

3.7 Input, Output, and Processing Flow Definition

---

### CHAPTER 4: SYSTEM DESIGN AND ARCHITECTURE

4.1 Overview of System Architecture

4.2 High-Level Architecture Diagram

4.3 Data Pipeline Design

4.4 Model Architecture Design

4.5 Federated Learning Architecture

4.6 Loss Function Design

4.7 Web Application Architecture

4.8 Real-Time Inference Pipeline Design

4.9 Evaluation Module Design

4.10 Security Architecture

4.11 Technology Stack Justification

4.12 Data Flow Diagrams

---

### CHAPTER 5: SYSTEM IMPLEMENTATION

**Part A: Federated Training Implementation**

5.1 Development Environment Setup

5.2 Dataset Loading and Exploratory Data Analysis

5.3 Data Preprocessing Implementation

5.4 Custom Dataset and DataLoader Implementation

5.5 Federated Data Partitioning Implementation

5.6 Model Implementation

5.7 FedAvg Algorithm Implementation

5.8 Federated Training Loop Implementation

**Part B: Web Application Implementation**

5.9 Backend Implementation — Django REST Framework

5.10 Frontend Implementation — Streamlit

5.11 Audit Logging Implementation

---

### CHAPTER 6: SYSTEM TESTING AND EVALUATION

6.1 Testing Strategy

6.2 Dataset Split Validation

6.3 Evaluation Metrics

6.4 Batch Audit and Data Integrity Check

6.5 Federated Training Results

6.6 Per-Class AUROC Analysis

6.7 Global vs. Local Model Comparison

6.8 Comparison with Existing Architectures

6.9 Web Application Testing

6.10 Discussion of Results

6.11 Limitations of the Current System

6.12 Suggested Improvements

---

### CHAPTER 7: CONCLUSION

7.1 Summary of Project Achievements

7.2 Fulfillment of Objectives

7.3 Key Contributions

7.4 Future Work Directions

7.5 Concluding Statement

---

### REFERENCES

### APPENDIX A — API Endpoint Reference

### APPENDIX B — Glossary of Terms

---

## LIST OF ABBREVIATIONS

| Abbreviation | Full Form |
|-------------|-----------|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| AUROC | Area Under the Receiver Operating Characteristic Curve |
| BSCS | Bachelor of Science in Computer Science |
| CNN | Convolutional Neural Network |
| CT | Computed Tomography |
| CXR | Chest X-ray |
| DL | Deep Learning |
| DP | Differential Privacy |
| EDA | Exploratory Data Analysis |
| FedAvg | Federated Averaging |
| FL | Federated Learning |
| F1 | F1-Score (Harmonic Mean of Precision and Recall) |
| FN | False Negative |
| FP | False Positive |
| GDPR | General Data Protection Regulation |
| GPU | Graphics Processing Unit |
| HIPAA | Health Insurance Portability and Accountability Act |
| IID | Independently and Identically Distributed |
| ILSVRC | ImageNet Large Scale Visual Recognition Challenge |
| JWT | JSON Web Token |
| LR | Learning Rate |
| ML | Machine Learning |
| NLP | Natural Language Processing |
| NFR | Non-Functional Requirement |
| Non-IID | Non-Independently and Identically Distributed |
| NaN | Not a Number (missing value) |
| PIL | Python Imaging Library |
| REST | Representational State Transfer |
| ROC | Receiver Operating Characteristic |
| ReLU | Rectified Linear Unit |
| SGD | Stochastic Gradient Descent |
| SVM | Support Vector Machine |
| TN | True Negative |
| TP | True Positive |
| UC | Use Case |
| ViT | Vision Transformer |
| WHO | World Health Organization |

---

## LIST OF TABLES

| Table No. | Title | Chapter |
|-----------|-------|---------|
| 1.1 | The 14 CheXpert Pathological Labels | Chapter 1 |
| 1.2 | Uncertainty Label Handling Strategies | Chapter 1 |
| 1.3 | Scope and Limitations | Chapter 1 |
| 2.1 | ViT Landmarks in Medical Imaging | Chapter 2 |
| 2.2 | CheXpert Benchmark Model Comparison | Chapter 2 |
| 2.3 | Transformer Architecture Comparison | Chapter 2 |
| 2.4 | Federated Learning in Medical Imaging | Chapter 2 |
| 2.5 | Research Gaps and Project Responses | Chapter 2 |
| 3.1 | Functional Requirement FR-01 | Chapter 3 |
| 3.2 | Functional Requirement FR-02 | Chapter 3 |
| 3.3 | Functional Requirement FR-03 | Chapter 3 |
| 3.4 | Functional Requirement FR-04 | Chapter 3 |
| 3.5 | Functional Requirement FR-05 | Chapter 3 |
| 3.6 | Functional Requirement FR-06 | Chapter 3 |
| 3.7 | Functional Requirement FR-07 | Chapter 3 |
| 3.8 | Functional Requirement FR-08 | Chapter 3 |
| 3.9 | Functional Requirement FR-09 | Chapter 3 |
| 3.10 | Functional Requirement FR-10 | Chapter 3 |
| 3.11 | Functional Requirements FR-11 to FR-15 | Chapter 3 |
| 3.12 | Non-Functional Requirements | Chapter 3 |
| 3.13 | Primary Stakeholders | Chapter 3 |
| 3.14 | Secondary Stakeholders | Chapter 3 |
| 3.15 | System Constraints | Chapter 3 |
| 3.16 | System Inputs | Chapter 3 |
| 3.17 | System Outputs | Chapter 3 |
| 4.1 | Federated Client Dataset Sizes | Chapter 4 |
| 4.2 | API Endpoints Overview | Chapter 4 |
| 4.3 | Database Schema: TrainingRound | Chapter 4 |
| 4.4 | Database Schema: LocalWeightSubmission | Chapter 4 |
| 4.5 | Database Schema: AuditLog | Chapter 4 |
| 4.6 | Technology Stack Justification | Chapter 4 |
| 5.1 | Development Environment Specifications | Chapter 5 |
| 5.2 | Disease Positive Case Counts | Chapter 5 |
| 5.3 | Class Imbalance Ratios | Chapter 5 |
| 5.4 | Client Dataset Sizes and Batch Counts | Chapter 5 |
| 6.1 | Unit Testing Checklist | Chapter 6 |
| 6.2 | Dataset Split Verification | Chapter 6 |
| 6.3 | AUROC Interpretation Scale | Chapter 6 |
| 6.4 | Per-Round Federated Global AUROC Results | Chapter 6 |
| 6.5 | Per-Class AUROC Progression (Round 1 vs. Round 15) | Chapter 6 |
| 6.6 | Global vs Local Model Comparison (Round 1 vs. Round 15) | Chapter 6 |
| 6.7 | Benchmark Comparison: Swin Transformer vs Baseline | Chapter 6 |
| 6.8 | API Endpoint Testing Results | Chapter 6 |
| 6.9 | Suggested Improvements | Chapter 6 |
| 7.1 | Project Achievements Summary | Chapter 7 |
| 7.2 | Objectives Fulfillment | Chapter 7 |

---

## LIST OF FIGURES

| Figure No. | Title | Chapter |
|------------|-------|---------|
| 4.1 | High-Level System Architecture Diagram | Chapter 4 |
| 4.2 | Swin Transformer V2 Structure | Chapter 4 |
| 4.3 | Federated Learning Star Topology Diagram | Chapter 4 |
| 4.4 | Real-Time Inference Pipeline Flow | Chapter 4 |
| 4.5 | Level 0 DFD — Context Diagram | Chapter 4 |
| 4.6 | Level 1 DFD — System Processes | Chapter 4 |
| 6.1 | Per-Round Global AUROC Convergence Curve | Chapter 6 |
| 6.2 | Per-Class AUROC Comparison: Round 1 vs Round 15 | Chapter 6 |
| 6.3 | Hospital Node Local AUROC vs Global AUROC | Chapter 6 |
