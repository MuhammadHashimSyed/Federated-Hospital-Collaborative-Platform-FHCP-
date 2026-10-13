# CHAPTER 7
# CONCLUSION

## 7.1 Summary of Project Achievements

The Project Achievements come from the summary below. This project has successfully designed, implemented and deployed an entire privacy framework. We propose a Federated Learning system for the automatic multi-label classification of chest image. On the CheXpert dataset, the model is tested on pathologies. The system is a state-of-the-art Swin integrated system. Simulating a federated learning setting with FedAvg aggregation algorithm and Transformer V2 deep learning model. three-hospital federated environment in which patient data never leaves the local institutional boundary. Most importantly, the project is not limited to the training pipeline, but also provides a clinical diagnostic inference service that is production-ready for a web application, allowing real-time inference via a structured role-based interface.

| Achievement | Detail |
|------------|--------|
| Large-scale EDA | Analyzed 223,414 CXR records; quantified imbalance ratios for all 14 pathologies |
| Robust preprocessing | Implemented Hybrid Uncertainty (U-Zeros/U-Ones) and complex Albumentations augmentations |
| Patient-level data integrity | Zero patient overlap between training and validation sets via GroupShuffleSplit |
| Federated partitioning | 171,927 training images divided across 3 patient-disjoint hospital clients |
| Vision Transformers | Swin Transformer V2 successfully adapted for 14-class multi-label CXR classification |
| Class imbalance handling | Focal Loss (alpha=0.25, gamma=2.0) applied to focus on hard, rare examples |
| FedAvg implementation | Weighted parameter averaging across three clients per communication round |
| 15-round federated training | Consistent per-round AUROC improvement; 0.7582 → 0.8124 |
| Optimizer scheduling | AdamW with 3-phase decaying learning rate schedule (1e-4 → 5e-5 → 2e-5) |
| System Performance | Achieved **81.24% Mean AUROC** and **87.25% Global Accuracy** |
| Per-node reporting | Each hospital reports local per-class AUROC alongside weight submission |
| Reproducibility | Fixed all random seeds; results fully reproducible across runs |
| Web application | Complete Django REST API + Streamlit frontend deployed and operational |
| Real-time inference | Live diagnostic endpoint serving Swin V2 predictions with model caching |
| Multi-role authentication | JWT-based access control for SUPER_ADMIN, HOSPITAL_CLIENT, and DOCTOR roles |
| Audit logging | Comprehensive action log for all significant system events |

---

## 7.2 Fulfillment of Objectives

All specific objectives defined in Chapter 1 have been fulfilled:

| # | Objective | Status | Evidence |
|---|-----------|--------|---------|
| 1 | Exploratory data analysis of CheXpert labels | ✓ Complete | 14-label count table, imbalance ratio analysis (Section 5.2) |
| 2 | Preprocessing pipeline with Albumentations & Hybrid labeling | ✓ Complete | Albumentations pipeline implemented (Section 5.3–5.4) |
| 3 | Patient-level GroupShuffleSplit | ✓ Complete | Zero patient overlap confirmed (Section 6.2) |
| 4 | Federated data partitioning into 3 client cohorts | ✓ Complete | 56,648 / 57,610 / 57,669 image splits (Section 5.5) |
| 5 | Swin Transformer V2 fine-tuning with Focal Loss | ✓ Complete | Custom 512-dim head, FocalLoss implementation (Section 5.6) |
| 6 | FedAvg aggregation implementation | ✓ Complete | Weighted averaging across all model layers (Section 5.7) |
| 7 | 15-round federated training with per-class AUROC | ✓ Complete | Rounds 1–15 completed; 0.7582→0.8124 (Section 6.5–6.6) |
| 8 | Per-hospital node local AUROC reporting | ✓ Complete | Node-level AUROC at Round 15 (Section 6.7) |
| 9 | Django REST API backend with role-based endpoints | ✓ Complete | 15+ endpoints implemented (Section 5.9) |
| 10 | Streamlit frontend with role-specific dashboards | ✓ Complete | Three-page application (Section 5.10) |
| 11 | JWT authentication and multi-role authorization | ✓ Complete | SimpleJWT; role checks per endpoint (Section 5.9.2) |
| 12 | Hospital node and doctor management | ✓ Complete | Register/remove hospitals and doctors (Section 5.9.3) |
| 13 | Real-time diagnostic inference with thresholding | ✓ Complete | LiveDiagnosticView; inference tested (Section 5.9.4, 6.9.2) |
| 14 | Federated training orchestration workflow | ✓ Complete | Start round → upload → aggregate → download cycle |
| 15 | Comprehensive audit logging | ✓ Complete | AuditLog model; all events recorded (Section 5.11) |

---

## 7.3 Key Contributions

### Clinical Contribution

The system shows automated multi-label chest pathology detection — covering reticulate, opaqueness, and cardiac enlargement — is possible. Federated training of 14 clinically relevant conditions together is possible. On a large scale without sharing of raw patient data across institutional boundaries. Mean AUROC: 0.8124, Overall accuracy: 87.25%, the system delivers a clinically sound diagnostic assistance signal. The Web interface in its live mode allows qualified doctors/medical professionals to upload chest X-rays and get a response. Structured clinical report in seconds and more—useful for clinical practice research prototype.

### Technical Contribution

The project provides an integrated end-to-end implementation of a federated deep learning pipeline with modern vision transformers, including:

- A complex Albumentations augmentation framework specifically designed for 256x256 medical imagery.
- A Hybrid Uncertainty strategy mapping different pathologies to U-Zeros or U-Ones Based on clinical assumptions.
- Every patient's data is secured with patient-level leakage prevention using GroupShuffleSplit.
- A federated FedAvg context with a heavy Swin Transformer V2.
- The massive class imbalance of (35:1) was addressed by implementing Focal Loss to explicitly solve the problem in the CheXpert dataset.
- A 3-phase AdamW learning rate decay schedule that guarantees the transformer to be securely running and converge over 15 rounds.
- A production-ready Django REST API with 15+ endpoints, JWT authentication, and memory-efficient model caching.
- A Streamlit multi-page front-end GUI that displays real-time analytics.

### Privacy Contribution

In the federated architecture, the raw patient images are never sent outside of the ARTIFICIAL environment in a local hospital setting. Only model weight tensors are traded, so that it's an alternative to data pooling that is not centralized that preserves privacy. The web application strengthens the activities of this architecture: hospital nodes communicates via authenticated API endpoints, weight files are persisted on the server and all accesses are logged. This is a structurally designed design for HIPAA and GDPR compliant data protection solutions.

### Academic Contribution

The work combines and implements a complex sequence of cutting edge ideas — Vision Transformers, multi-label classification, Focal Loss for imbalance, federated optimization, web application development, JWT security, and real-time inference — all in one coherent system. The methodology is rigorous, reproducible and documented at a high level Standard, exhibiting excellence at BSCS Final Year Project level.

---

## 7.4 Future Work Directions

### 7.4.1 Enhanced Federated Algorithms
FedProx is a proximal variant of FedAvg that includes a proximal term to the local objective that constrains client updates are sent from nodes that have deviated too much from the global model. Especially, this is crucial for non-IID settings. Develop and compare FedProx with FedAvg (baseline) would quantify its benefit for chest X-ray federated learning.

### 7.4.2 Differential Privacy Integration
The local training step can be done with the addition of Differential Privacy (DP-SGD), which would provide formal mathematical privacy ensures that gradient inversion attacks are not possible. Libraries such as Opacus (Meta AI) integrate DP-SGD directly with PyTorch and could be integrated into the existing training loop.

### 7.4.3 Non-IID Federated Simulation
The current implementation is based on some approximation of IID data partitioning. A more realistic in simulation it would be a partitioning of patients based on the prevalence of the disease in a hospital, which is the real-world situation. specializations — e.g., Hospital A has high prevalence of cardiomegaly, Hospital B has high prevalence of pneumonia prevalence. This will put the FedAvg convergence to the test.

### 7.4.4 Real Multi-Institution Deployment
The most important direction for the future is deployment in a real multi-hospital setting, where the federated learning system is distributed among physically-distributed servers in the real world Population as the patient base and real communication over the network. This would give ground truth for the simulated setting.

---

## 7.5 Concluding Statement

This project will prove that privacy preserving, multi-institutional chest X-ray diagnosis is possible. Using Federated Learning is not only technically possible but also can be encapsulated as a complete, Accessible web application to help the gap between research and clinical practice. From a research prototype, this work provided a system that enables a doctor to upload a patient's clinical data to it. Check a chest X-ray on the web and get a detailed diagnosis back. It was powered by a Swin Transformer V2 model that has been trained jointly in three simulated All hospital nodes — without any raw patient data ever leaving any of the institutions that are the nodes — generated it.

The federated training pipeline leverages Focal Loss and Albumentations to achieve an impressive **81.24% mean AUROC and 87.25% Accuracy** across 14 pathological classes after 15 communication rounds. The web application implements a complete federated learning lifecycle — from round initiation to weight submission to FedAvg aggregation to global model distribution — within a secure, audit-logged, role-based platform.

This work makes a meaningful contribution to the convergence of privacy-preserving AI, medical imaging, and practical software engineering — demonstrating that the complex demands of healthcare AI are not in conflict with the fundamental requirement to protect patient privacy.

---

*End of Chapter 7*

---

# REFERENCES

1. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.
2. Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciosi-Brown, S., Chute, C., ... & Lungren, M. P. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. *AAAI Conference on Artificial Intelligence*.
3. Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang, Z., Dong, L., Wei, F., & Guo, B. (2022). Swin Transformer V2: Scaling Up Capacity and Resolution. *CVPR*.
4. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV*.
5. Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and Flexible Image Augmentations. *Information*.
6. Sheller, M. J., Reina, G. A., Edwards, B., Martin, J., & Bakas, S. (2019). Multi-institutional Deep Learning Modeling Without Sharing Patient Data. *Brainlesion*.
7. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Smola, A., & Smith, V. (2020). Federated Optimization in Heterogeneous Networks. *MLSys*.
8. Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S., ... & Seth, K. (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning. *ACM SIGSAC (CCS)*.

---

# APPENDIX A — API Endpoint Reference

| Endpoint | Method | Required Role | Description |
|----------|--------|--------------|-------------|
| /api/token/ | POST | None | Obtain JWT access and refresh tokens |
| /api/token/refresh/ | POST | None | Refresh expired access token |
| /api/federated/round/current/ | GET | Any auth | Get current active round info |
| /api/federated/round/start/ | POST | SUPER_ADMIN | Start new federated training round |
| /api/federated/weights/upload/ | POST | HOSPITAL_CLIENT | Upload local model weights + AUROC |
| /api/federated/aggregate/ | POST | SUPER_ADMIN | Trigger FedAvg aggregation |
| /api/federated/model/download/ | GET | HOSPITAL_CLIENT | Download latest global model |
| /api/federated/diagnostics/predict/ | POST | Any auth | Live X-ray inference |
| /api/federated/analytics/ | GET | Any auth | Per-round analytics |
| /api/federated/hospitals/ | GET | SUPER_ADMIN | List hospital nodes |
| /api/federated/hospitals/register/ | POST | SUPER_ADMIN | Register hospital node |
| /api/federated/hospitals/{id}/delete/ | DELETE | SUPER_ADMIN | Remove hospital node |
| /api/federated/doctors/ | GET/POST | HOSPITAL_CLIENT | List/register doctors |
| /api/federated/doctors/{id}/delete/ | DELETE | HOSPITAL_CLIENT | Remove doctor |
| /api/federated/logs/ | GET | SUPER_ADMIN | View full audit log |
| /api/federated/history/ | GET | HOSPITAL_CLIENT | View local submission history |
| /api/federated/reset/ | POST | SUPER_ADMIN | Reset all training rounds |

---

# APPENDIX B — Glossary of Terms

| Term | Definition |
|------|-----------|
| AUROC | Area Under the ROC Curve; primary evaluation metric; 0.5 = random, 1.0 = perfect |
| Albumentations | Python library used for highly optimized spatial and pixel-level image augmentations |
| CheXpert | Stanford chest X-ray dataset; 224,316 images; 14 pathology labels |
| Focal Loss | Dynamically scaled cross-entropy loss designed to focus on hard, underrepresented examples |
| FedAvg | Federated Averaging algorithm; weighted average of client model weights |
| Federated Learning | Distributed ML paradigm; clients train locally, share model weights, not data |
| GroupShuffleSplit | scikit-learn splitter ensuring no patient appears in both train and validation sets |
| JWT | JSON Web Token; stateless authentication mechanism used for API security |
| Swin Transformer V2 | Hierarchical Vision Transformer using shifted window attention |
| Hybrid Uncertainty | Preprocessing strategy applying U-Zeros or U-Ones depending on clinical pathology priors |
| Super Admin | System administrator role; manages hospitals, starts rounds, triggers aggregation |
| Hospital Client | Hospital node role; uploads weights, manages doctors, accesses diagnostics |
| Doctor | Medical practitioner role; accesses live diagnostic inference only |
