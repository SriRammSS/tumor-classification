<div align="center">

# Cross-Organ Bridge Transfer Learning for Lung Cancer Detection

### IEEE Published Research — BTech Capstone Project

[![IEEE](https://img.shields.io/badge/Published-IEEE_R10--HTC_2023-00629B?style=for-the-badge&logo=ieee&logoColor=white)](https://doi.org/10.1109/R10-HTC57504.2023.10461796)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-VGG19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Academic_Non--Commercial-dc2626?style=for-the-badge)](LICENSE)

<br/>

> **Published at:** 2023 IEEE 11th Region 10 Humanitarian Technology Conference (R10-HTC)
> **DOI:** [10.1109/R10-HTC57504.2023.10461796](https://doi.org/10.1109/R10-HTC57504.2023.10461796)
> **Institution:** Amrita School of Engineering, Coimbatore, India

<br/>

> Novel **Cross-Organ Bridge Transfer Learning** approach — leveraging kidney CT knowledge as a domain bridge to improve lung cancer classification, achieving **93% accuracy** vs 90% baseline.

</div>

---

## Table of Contents

- [Research Overview](#research-overview)
- [The Novel Contribution — Bridge Transfer Learning](#the-novel-contribution--bridge-transfer-learning)
- [Published Results](#published-results)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Web Applications](#web-applications)
- [Project Structure](#project-structure)
- [Setup & Running](#setup--running)
- [Authors](#authors)

---

## Research Overview

This project proposes a **Modality-Bridge Transfer Learning** framework for medical CT scan image classification. The core challenge addressed is the **domain mismatch** between natural image pre-training (ImageNet) and medical image target domains.

### The Problem
Traditional transfer learning:
```
ImageNet (natural photos) ──────────────────→ Lung CT scans
 Source Target
 [Large domain gap — different textures, structures, imaging artifacts]
```

### The Solution: Cross-Organ Bridge Transfer Learning
```
ImageNet → Kidney CT scans → Lung CT scans
 Source Bridge Target
 [Same CT modality] [Same CT modality]
 [Smaller gap] [Organ-level feature transfer]
```

By inserting a **bridge domain** — kidney CT scans, which share the same acquisition modality (CT) as the target lung CT — the model progressively adapts from natural image features to medical CT features before reaching the target task. This reduces distribution mismatch and improves generalization even with limited labelled target data.

---

## The Novel Contribution — Bridge Transfer Learning

### Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Modality-Bridge Transfer Learning Pipeline │
│ │
│ Step 1: Pre-train on ImageNet (natural images — 14M images, 1000 classes) │
│ │ │
│ ▼ │
│ Step 2: Fine-tune on Kidney CT Dataset (bridge domain — same CT modality) │
│ → VGG19 learns CT-specific features: tissue density, organ │
│ boundaries, contrast gradients, bone structure │
│ → Achieves 95.6% accuracy on kidney tumor classification │
│ │ │
│ ▼ │
│ Step 3: Transfer to Lung CT Dataset (target domain) │
│ → Model already understands CT imaging characteristics │
│ → Fine-tune final layers for lung cancer histology │
│ → Achieves 93% accuracy (vs 90% without bridge) │
│ │
│ KEY INSIGHT: Same modality (CT) → Shared low-level features │
│ (edges, textures, density gradients) transfer cleanly across organs │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Bridge Transfer Learning (MBTL)
The paper also introduces **Multi-Bridge Transfer Learning**, which allows leveraging multiple source domains simultaneously, further improving robustness and handling data imbalance.

---

## Published Results

### Table I — Cross-Organ Bridge (CT Modality)

| Model | Domain | Accuracy |
|-------|--------|----------|
| Baseline VGG19 | Lung CT (standalone) | 90% |
| Bridge VGG19 | Kidney CT (bridge) | **95.6%** |
| **Bridge Transfer** | **Lung CT (via Kidney CT bridge)** | **93% ** |

> **Key finding:** Bridge transfer learning improves Lung CT classification by **+3%** over the standalone baseline — demonstrating effective cross-organ knowledge transfer.

### Table II — Cross-Organ Bridge (USG Modality)

| Model | Domain | Accuracy |
|-------|--------|----------|
| Baseline VGG19 | Lung CT | 90% |
| VGG19 | Kidney USG | 82% |
| Bridge Transfer | Lung CT (via Kidney USG bridge) | 75% |

> **Conclusion from Table II:** CT-to-CT transfer (93%) significantly outperforms USG-to-CT transfer (75%), confirming that **same-modality bridging is critical** to the method's effectiveness.

### Training Curve Metrics (per training run)

#### Kidney CT Model
| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 95.60% | **98.40%** |
| Precision | 95.78% | 98.40% |
| Recall | 95.40% | 98.40% |
| AUC | 99.45% | **99.88%** |
| Loss | 0.0934 | 0.0462 |

#### Lung CT Model — Balanced Dataset
| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 90.60% | **84.40%** |
| Precision | 90.58% | 84.40% |
| Recall | 90.40% | 84.40% |
| AUC | 97.41% | **92.05%** |

> **Note on discrepancy:** The paper reports 90% for the lung CT baseline — this corresponds to the training accuracy of the balanced dataset model. The test accuracy of 84.4% reflects true generalisation on unseen data. The bridge learning result of 93% (paper) represents the improvement achieved by the cross-organ knowledge transfer.

---

## Model Architecture

### VGG19 with Transfer Learning

```
Input Image (224×224×3)
 │
 ▼
┌─────────────────────┐
│ Block 1 │ 2× Conv(64, 3×3) + ReLU → MaxPool(2×2)
├─────────────────────┤
│ Block 2 │ 2× Conv(128, 3×3) + ReLU → MaxPool(2×2)
├─────────────────────┤
│ Block 3 │ 4× Conv(256, 3×3) + ReLU → MaxPool(2×2)
├─────────────────────┤
│ Block 4 │ 4× Conv(512, 3×3) + ReLU → MaxPool(2×2)
├─────────────────────┤
│ Block 5 │ 4× Conv(512, 3×3) + ReLU → MaxPool(2×2)
├─────────────────────┤
│ Flatten │ 7×7×512 = 25,088 features
├─────────────────────┤
│ FC-4096 + Dropout │ Fine-tuned for medical domain
├─────────────────────┤
│ FC-4096 + Dropout │ Fine-tuned for medical domain
├─────────────────────┤
│ Output (Softmax) │ Kidney: 2 classes | Lung: 4 classes
└─────────────────────┘
```

### Why VGG19?
- Proven feature extractor on ImageNet (top-5 accuracy: 92.7% on 1000 classes)
- Deep enough to learn hierarchical medical image features (edges → textures → organ structures → pathology)
- Simple uniform architecture makes fine-tuning predictable and controllable
- Well-studied transfer learning properties

---

## Datasets

### Kidney CT Dataset (Bridge Domain)
- **Task:** Binary classification — Normal vs Tumor
- **Classes:** `Normal`, `Tumor`
- **Modality:** CT scan (KUB — Kidney, Ureter, Bladder)
- **Role:** Bridge domain — same CT modality as target

### Lung Cancer CT Dataset (Target Domain)
- **Task:** Multi-class classification — 4 histological subtypes
- **Classes:**
 - `Adenocarcinoma` — solid nodule with spiculated margins, ground-glass opacity
 - `Large Cell Carcinoma` — large mass with irregular borders and necrotic center
 - `Squamous Cell Carcinoma` — cavitary lesion with thick walls and calcifications
 - `Normal` — healthy lung tissue
- **Modality:** Chest CT scan

### Data Preprocessing
- Resize all images to **224×224×3** (VGG19 input requirement)
- Normalize pixel values to [0, 1]
- Data augmentation: shift, scale, rotate, flip, brightness adjustment, noise introduction
- Balanced dataset created to address class imbalance (original dataset overrepresented normal class)

---

## Web Applications

Two Flask web apps are included for live inference:

| App | URL | Task | Classes |
|-----|-----|------|---------|
| **Kidney Tumor Classifier** | `http://localhost:5000` | Binary | Normal, Tumor |
| **Lung Cancer Classifier** | `http://localhost:5001` | Multi-class | Adenocarcinoma, Large Cell Carcinoma, Normal, Squamous Cell Carcinoma |

### Features
- Upload CT scan image (`.jpg`, `.jpeg`, `.png`) or provide a URL
- VGG19 inference with top-2/top-3 class probabilities
- Clinical recommendation text per prediction
- Responsive web interface

---

## Project Structure

```
Tumor Classification/
│
├── Notebooks (Training)
│ ├── KUB-ct-scan-VGG19(for-kidney-tumors).ipynb # Kidney tumor VGG19 training
│ ├── chest-ct-scan-VGG19-with-transfer-learning.ipynb # Lung cancer VGG19 training
│ ├── lung-ct-scan-VGG19(for-lung-tumors)-D2.ipynb # Lung cancer balanced training
│ └── Comparsion.ipynb # Model comparison analysis
│
├── Trained Models
│ ├── kidney_tumor_model.hdf5 # Kidney VGG19 model weights
│ ├── lung_cancer_model.hdf5 # Lung VGG19 model weights (unbalanced)
│ ├── lung_cancer_B_model.hdf5 # Lung VGG19 model weights (balanced)
│ └── chest_CT_SCAN.h5 # Chest CT scan model
│
├── Results/
│ ├── VGG_KUB_cancer_result.png # Kidney model training curves
│ ├── VGG_lung_cancer_result.png # Lung model training curves (unbalanced)
│ └── VGG_lung_cancer_balanced_results.png # Lung model training curves (balanced)
│
├── Webapp_Kidney_tumor/
│ ├── app.py # Flask application
│ ├── model.hdf5 # Deployed kidney model
│ ├── templates/index.html # Upload page
│ ├── templates/success.html # Results page
│ └── static/ # CSS and uploaded images
│
├── Webapp_lung_Cancer/
│ ├── app.py # Flask application
│ ├── model.hdf5 # Deployed lung cancer model
│ ├── templates/index.html # Upload page
│ ├── templates/success.html # Results page
│ └── static/ # CSS and uploaded images
│
├── kidney_tumor_dataset/ # Kidney CT scan dataset
├── lung_cancer_dataset/ # Lung CT scans (original)
├── lung_cancer_dataset_balanced/ # Lung CT scans (balanced)
│
├── requirements.txt
├── Interview_Guide.pdf # Detailed interview preparation guide
└── README.md
```

---

## Setup & Running

### Prerequisites
- Python 3.12+ (or 3.10 for original TF 2.10)
- 4GB+ RAM recommended for model loading

### Installation

```bash
# Clone repository
git clone https://github.com/SriRammSS/tumor-classification.git
cd "tumor-classification"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow tf-keras Flask==2.2.2 Pillow Werkzeug==2.2.3 Flask-Cors
```

### Run the Kidney Tumor Webapp

```bash
cd Webapp_Kidney_tumor
python app.py
# → http://localhost:5000
```

### Run the Lung Cancer Webapp

```bash
cd Webapp_lung_Cancer
python app.py
# → http://localhost:5001
```

### Run Jupyter Notebooks (Model Training)

```bash
pip install jupyter notebook
jupyter notebook
# Open any .ipynb file to view/re-run training
```

---

## Access Policy & Intellectual Property

> **This is a peer-reviewed, IEEE-published research project. The core IP is protected.**

### What Is Public (This Repo)

| Asset | Status | Reason |
|-------|--------|--------|
| Flask web application code | Public | Deployment layer only |
| Web UI templates & CSS | Public | Frontend only |
| Training result plots | Public | Already in IEEE paper |
| Interview guide | Public | Documentation only |
| `requirements.txt` | Public | Standard setup |

### What Is Protected (Available on Request)

| Asset | Status | Reason |
|-------|--------|--------|
| Trained model weights (`.hdf5`) | On Request | Proprietary — months of compute, novel training pipeline |
| Training notebooks | On Request | Core bridge learning implementation — the published IP |
| CT scan datasets | On Request | Medical data — licensed separately, not for redistribution |

### Requesting Access

For **academic collaboration**, research reproduction, or dataset access, contact:

 **srirammsekar07@gmail.com**

Include in your request:
- Your institution and role
- Intended use (research / teaching / benchmarking)
- Confirmation of non-commercial purpose

> Commercial use requires a separate written licensing agreement.

### Citation

If you use this work, you **must** cite the IEEE paper:

```bibtex
@inproceedings{sriramm2023crossorgan,
 author = {Sriramm, S. S. and Kamali, R. and Kishorkumar, S. M.
 and Venkatesh, K. V. Prasanna and Suguna, G.},
 title = {Cross Organ Bridge Transfer Learning for Lung Cancer Detection},
 booktitle = {2023 IEEE 11th Region 10 Humanitarian Technology Conference (R10-HTC)},
 year = {2023},
 pages = {876--883},
 doi = {10.1109/R10-HTC57504.2023.10461796}
}
```

---

## Authors

**Sri Ramm Sekar Sasirekha** — Department of Electronics & Communication Engineering, Amrita School of Engineering, Coimbatore

**Kamali R** · **S M Kishorkumar** · **K V Prasanna Venkatesh**

**Guide:** Prof. Suguna G. — Dept. of ECE, Amrita School of Engineering

[![IEEE](https://img.shields.io/badge/Full_Paper-IEEE_Xplore-00629B?style=flat&logo=ieee)](https://doi.org/10.1109/R10-HTC57504.2023.10461796)
[![GitHub](https://img.shields.io/badge/GitHub-SriRammSS-181717?style=flat&logo=github)](https://github.com/SriRammSS)

---

<div align="center">
<sub>Published at IEEE R10-HTC 2023 · DOI: 10.1109/R10-HTC57504.2023.10461796 · © 2023 IEEE</sub>
</div>
