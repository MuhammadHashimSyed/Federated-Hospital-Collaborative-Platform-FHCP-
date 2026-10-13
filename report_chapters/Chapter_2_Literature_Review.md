# CHAPTER 2
# LITERATURE REVIEW

## 2.1 Overview of Deep Learning in Medical Imaging

### 2.1.1 Evolution from Traditional Methods to Deep Learning

There are three paradigms of automated medical image analysis. The earliest systems adopted classical image processing and hand-engineered algorithms based on edges. For detecting anatomical structures, this involves using edge detection, thresholding, and texture descriptors. While these methods were interpretable, they were brittle and demanded manual redesign for each new imaging task or modality that arose.

The second paradigm was using classic machine learning algorithms: Support Vector Machines (SVM), Random Forests, and k-Nearest Neighbours were used with manually designed feature vectors. The performance was heavily reliant on the quality of feature engineering, which required a great deal of background knowledge and did not often generalize across broad clinical settings.

The third, current paradigm — Deep Learning — gets rid of feature engineering. The bottleneck is addressed by directly learning from raw pixel data using end-to-end learning. Hierarchical representations are learned automatically in Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). This paradigm shift was initially catalyzed by Krizhevsky et al.'s AlexNet (2012) and was recently revolutionized by Dosovitskiy et al.'s Vision Transformer (2020), showing the power of deep neural networks in visual recognition.

A critical enabler of deep learning in medicine is **transfer learning**: initializing models with weights pre-trained on large natural image datasets (ImageNet) and fine-tuning on medical data. Transfer learning substantially reduces labeled data requirements — a critical advantage given the high cost of expert medical annotation.

### 2.1.2 Deep Learning Architectures in Medical Imaging

Deep learning has achieved landmark results across multiple medical imaging domains:

| Domain | Work | Key Finding |
|--------|------|-------------|
| Dermatology | Esteva et al. (2017) | CNN matched board-certified dermatologists on skin cancer classification |
| Ophthalmology | Gulshan et al. (2016) | Deep learning detected diabetic retinopathy with specialist-level sensitivity |
| Radiology | Rajpurkar et al. (2017) | Deep learning exceeded radiologist F1 on pneumonia detection |
| Pathology | Litjens et al. (2016) | Neural networks outperformed pathologists on prostate cancer histology |

These results established deep learning as the method of choice for medical image classification and motivated its adoption for chest X-ray analysis.

---

## 2.2 Existing Work on Chest X-ray Disease Detection

### 2.2.1 CheXNet — DenseNet121 Baseline (Stanford, 2017)

A highly relevant prior work is CheXNet by Rajpurkar, Irvin, Ball et al. (Stanford Machine Learning Group, 2017). CheXNet applied a DenseNet121 architecture to the NIH ChestX-ray14 dataset (112,120 radiographs, 14 disease labels) and achieved:

- F1 score of 0.435 on pneumonia detection vs. mean radiologist F1 of 0.387.
- AUROC exceeding individual radiologists on 11 of 14 pathologies.
- Demonstrated that ImageNet pre-training followed by task-specific fine-tuning is effective even with relatively small medical datasets.

CheXNet established the standard baseline for chest X-ray AI performance, but more recent transformer-based architectures have consistently surpassed these early CNN baselines.

### 2.2.2 CheXpert Benchmark Models

Following the CheXpert dataset release (Irvin et al., 2019), several models have been evaluated on its five competition pathologies:

| Model / Study | Architecture | Mean AUROC (5 classes) |
|---------------|-------------|------------------------|
| CheXpert Baseline (Irvin et al., 2019) | DenseNet121 (centralized) | 0.889 |
| Guan et al. (2020) | ResNet-50 + Attention | 0.871 |
| Pham et al. (2021) — CheXpert Champion | EfficientNet-B7 Ensemble | 0.930 |
| Ghesu et al. (2022) | Vision Transformer (ViT) | 0.921 |

These benchmarks establish the upper performance bounds for centralized training on CheXpert. The federated system in this project achieves **81.24% mean AUROC** across all **14 pathologies** (not just the 5 competition classes) after 15 communication rounds — a highly competitive result given the strict constraints of privacy-preserving federated training and the broader, more challenging 14-class evaluation scope.

### 2.2.3 Other Notable Works

**Wang et al. (2017) — NIH ChestX-ray14:** Introduced the 112,120-image ChestX-ray14 dataset and a multi-label CNN baseline, establishing the template for large-scale chest X-ray AI research.

**Johnson et al. (2019) — MIMIC-CXR:** Released 377,110 chest radiographs linked to clinical records. Models trained on MIMIC-CXR demonstrate strong cross-dataset generalization, highlighting the value of large, diverse training corpora.

**Yao et al. (2017):** Modeled label co-occurrence dependencies using LSTM networks, capturing the clinical reality that certain pathologies frequently co-occur (e.g., Edema and Pleural Effusion).

---

## 2.3 Deep Learning Architectures for Medical Classification

### 2.3.1 Convolutional Neural Networks (CNNs)

For many years, CNNs such as ResNet and DenseNet have been the mainstays of medical imaging. ResNet (He et al., 2015) proposed a skip connection method to alleviate the vanishing gradient problem, which allowed the training of very deep networks. This was extended by DenseNet (Huang et al., 2016) by linking each layer with each subsequent layer, thus encouraging aggressive feature reuse and parameter efficiency. Although CNNs are very powerful, they are based on local convolution kernels, which restricts their capacity to see long-range global connections in an image early in the network.

### 2.3.2 Vision Transformers (ViTs)

The Vision Transformer (Dosovitskiy et al., 2020) revolutionized computer vision by treating images as sequences of patches (similar to words in NLP) and applying the Self-Attention mechanism. Self-attention allows the model to weigh the importance of every patch relative to every other patch, capturing global context perfectly. However, standard ViTs require massive datasets to train from scratch and suffer from quadratic computational complexity concerning image resolution, making high-resolution medical imaging computationally expensive.

### 2.3.3 Swin Transformer V2

The Swin (Shifted Window) Transformer (Liu et al., 2021) solved the scaling issues of the standard ViT by computing self-attention locally within non-overlapping windows and "shifting" those windows in subsequent layers to allow cross-window connections. This hierarchical structure brings linear computational complexity while maintaining the power of global self-attention.

**Swin Transformer V2** (Liu et al., 2022) further improved this architecture by:
1. **Post-Normalization:** Moving the LayerNorm to improve training stability at scale.
2. **Scaled Cosine Attention:** Replacing dot-product attention with a cosine similarity function, preventing feature maps from dominating the attention weights and improving generalization.
3. **Log-Spaced Continuous Position Bias:** Enabling smoother transfer from low-resolution pre-training (ImageNet) to higher-resolution medical fine-tuning.

These properties make Swin Transformer V2 exceptionally well-suited for chest X-ray analysis, where both fine-grained localized abnormalities (e.g., nodules) and global contextual patterns (e.g., cardiomegaly) must be simultaneously recognized.

### 2.3.4 Comparative Analysis and Justification

| Architecture | Paradigm | Key Advantage | Key Limitation | Suitability |
|-------------|----------|--------------|----------------|-------------|
| ResNet-50 | CNN | Residual connections | Limited global context | Moderate |
| DenseNet121 | CNN | Parameter efficiency | Localized receptive field | High |
| Standard ViT | Transformer | Global attention | Quadratic scaling cost | Low |
| **Swin Transformer V2** | **Hierarchical Transformer** | **Global context + Linear scaling** | **Complex implementation** | **Optimal** |

Swin Transformer V2 is selected for this project because it provides the state-of-the-art representational power of transformers while remaining computationally tractable for a federated multi-model environment.

---

## 2.4 Federated Learning: State of the Art

### 2.4.1 FedAvg Algorithm — McMahan et al. (2017)

McMahan, Moore, Ramage et al. (Google, 2017) formally established Federated Learning and proposed the FedAvg algorithm in "Communication-Efficient Learning of Deep Networks from Decentralized Data." The algorithm operates as follows:

```
Algorithm: FedAvg
─────────────────────────────────────
Server initializes global model W₀
For each round t = 1, 2, ..., T:
    Broadcast W_t to all clients
    For each client k (in parallel):
        Train locally for E epochs on D_k
        Compute local weights W_k
        Send W_k to server
    Aggregate: W_{t+1} = Σ_k (n_k / N) × W_k
Return final global model W_T
─────────────────────────────────────
```

FedAvg's key innovation over federated SGD is allowing multiple local gradient steps per round, dramatically reducing communication overhead. It remains the most widely adopted federated optimization algorithm.

### 2.4.2 Federated Learning in Medical Imaging

Several landmark works have demonstrated FL feasibility in healthcare:

| Study | Application | Key Finding |
|-------|-------------|------------|
| Sheller et al. (2018, 2020) | Brain tumor segmentation (BraTS) | Federated ≈ centralized performance |
| Roth et al. (2020) | COVID-19 CT detection | First real-world clinical FL deployment |
| Dou et al. (2021) | Prostate cancer histology | FL accommodates staining heterogeneity |
| Liu et al. (2021) | CXR disease detection | Federated models highly competitive |

These studies collectively validate the federated approach for medical image analysis and directly support the methodology adopted in this project.

---

## 2.5 Handling Class Imbalance and Data Scarcity

### 2.5.1 Data Augmentation (Albumentations)

It is important to use aggressive data augmentation in medical datasets, where positive samples are scarce, to prevent overfitting. Regular enhancement typically fails to deliver the desired results. The **Albumentations** library (Buslaev et al., 2020) gives very optimized and complex transformations. By applying ShiftScaleRotate, Contrast Limited Adaptive Histogram Equalization (CLAHE), RandomBrightnessContrast, GaussNoise, and CoarseDropout, the model is forced to learn strong representations instead of memorizing precise pixel layouts, emulating the natural differences between various x-ray machines.

### 2.5.2 Focal Loss

Class imbalance is a serious problem in CheXpert (e.g., Pleural Other is heavily underrepresented). Standard cross-entropy loss is able to allow the vast number of "easy" negative examples to overwhelm the loss gradient. To address this, this project implements **Focal Loss** (Lin et al., 2017):

```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```

Focal Loss adds a modulating factor `(1 - p_t)^γ` to the standard cross-entropy criterion. In the case of an example being misclassified and `p_t` being small, the modulating factor is close to 1 and the loss is unaffected. When `p_t` gets closer to 1 (easy enough to be well classified), the factor tends to 0, down-weighting the loss. This makes sure that the Swin Transformer V2 optimizer puts a strong emphasis on hard examples of the minority class (such as Lung Lesions).

### 2.5.3 AUROC as an Imbalance-Robust Metric

The idea of standard accuracy is misleading when class imbalance occurs — a very trivial baseline classifier that always predicts the majority class can get high accuracy with no diagnostic value. The Area Under the ROC Curve (AUROC) is independent of the threshold and class proportions and is the benchmark metric used to evaluate medical AI. A value ≥ 0.8 is generally regarded as clinically useful for diagnostic support tasks.

---

## 2.6 Web-Based Medical AI Systems

### 2.6.1 REST API Architectures for Medical AI

AI systems for production are now being deployed more and more based on REST APIs that isolate and abstract the model from user-level interfaces. This means that the model can be updated independently of the frontend, provides role-based access control, and allows easy integration with existing hospital information systems.

**Django REST Framework (DRF)** is a popular backend framework for medical AI APIs because of its comprehensive authentication capabilities, integrated serialization, and support for scientific Python libraries (PyTorch, NumPy, Pillow). Key features relevant to deployment in health include:
- Secure, stateless, scalable access control via JWT token authentication
- Fine-grained permission classes based on role-based authorization
- Handling the uploading of files for medical image ingestion

### 2.6.2 Real-Time Inference Systems

Real-time inference poses certain engineering challenges: loading latency of the models (especially on transformer architectures that have large initialization times), the speed of forward and backward propagation, memory management on CPU/GPU, and thread-safe request handling. Common solutions include:
- **Model caching:** Load the Swin Transformer V2 model at application start-up and serve many accesses from the cached instance.
- **CUDA/CPU fallback:** Run inferences on CPU if GPU is not available in development environments.
- **Confidence thresholding:** Filter model output to report only clinically relevant data that exceed a specified probability level.

In this project, all three strategies are implemented in the `LiveDiagnosticView` Django view.

### 2.6.3 Audit Logging and Regulatory Compliance

Healthcare AI systems need to have thorough audit trails in order to comply with HIPAA and GDPR requirements. Best practice involves logging each access event with: actor identity, description of action, time, and source IP address. This project adopts an own-developed `AuditLog` database model that automatically logs weight submissions, round initiations, model downloads, hospital registrations, and doctor management operations.

---

## 2.7 Research Gaps Identified

The review of existing literature reveals the following gaps that this project addresses:

| Gap | Description |
|-----|-------------|
| G1 | Limited federated learning research specifically applying Vision Transformers on the CheXpert dataset |
| G2 | Most federated CXR studies evaluate only 5 competition classes; 14-class federated evaluation is rarely reported |
| G3 | Advanced class imbalance handling (Focal Loss) within federated settings is underexplored |
| G4 | Many published works use image-level splits, risking patient-level data leakage |
| G5 | Complex data augmentation (Albumentations) is often overlooked in federated pipelines |
| G6 | Per-hospital node AUROC transparency within federated systems is rarely reported |
| G7 | Most federated learning prototypes lack a practical web-based deployment interface serving a ViT |

---

## 2.8 How This Project Addresses the Gaps

| Gap | This Project's Approach |
|-----|------------------------|
| G1 | Applies FL to the full CheXpert training set using Swin Transformer V2 over 15 rounds |
| G2 | Evaluates mean AUROC across all 14 pathological labels per federated round |
| G3 | Implements Focal Loss to dynamically focus training on hard, minority classes |
| G4 | Uses patient-level GroupShuffleSplit to eliminate data leakage |
| G5 | Integrates the Albumentations library for aggressive, realistic data augmentation |
| G6 | Each hospital node reports per-class local AUROC alongside weight submission |
| G7 | Deploys a complete Django + Streamlit web application for real-time diagnostic use |

---

*End of Chapter 2*
