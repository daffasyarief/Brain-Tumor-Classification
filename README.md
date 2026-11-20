# A Hybrid ConvNeXt–ResNet Architecture with Feature-Level Fusion for Brain MRI Classification

<p align="center">
  <img src="Hybrid CNN Brain Tumor Classification Architecture.png" width="850" alt="Hybrid ConvNeXt ResNet Architecture"/>
</p>

## Overview
This project explores a hybrid deep learning approach for brain tumor classification using Magnetic Resonance Imaging (MRI) scans. The proposed architecture integrates two convolutional neural network backbones, **ConvNeXt** and **ResNet**, to extract complementary feature representations from medical images. A feature-level fusion strategy is employed to combine the extracted representations before passing them through a classification head to predict one of four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor. The primary objective is to investigate how combining different CNN architectures can improve feature representation and classification performance in medical image analysis.

## Key Features
- Hybrid CNN architecture combining ConvNeXt and ResNet.
- Feature-level fusion for complementary feature integration.
- Transfer learning using pretrained architectures.
- Four-class brain tumor classification.
- Image preprocessing and data augmentation.
- Model evaluation using multiple classification metrics.

## Model Architecture
The proposed hybrid model consists of three main components.
### 1. ConvNeXt Branch
ConvNeXt is utilized as one of the feature extraction backbones to learn hierarchical visual representations from MRI images.
### 2. ResNet Branch
ResNet provides complementary feature representations through residual connections, enabling effective deep feature extraction.
### 3. Feature-Level Fusion
The extracted feature vectors from both backbones are concatenated and passed through a fully connected classification head.

**Architecture Configuration:**
| Component | Configuration |
|---|---|
| Backbone 1 | ConvNeXt Tiny |
| Backbone 2 | ResNet50 |
| ConvNeXt Feature Dimension | 768 |
| ResNet Feature Dimension | 2048 |
| Fusion Strategy | Concatenation |
| Fused Feature Dimension | 2816 |
| Classification Head | Fully Connected Layers |
| Activation | ReLU |
| Regularization | Dropout |
| Output Classes | 4 |

## Dataset
The project uses a Brain MRI dataset containing four classification categories.

| Class | Description |
|---|---|
| Glioma | Glioma tumor |
| Meningioma | Meningioma tumor |
| No Tumor | No detectable tumor |
| Pituitary | Pituitary tumor |

### Dataset Structure
```text
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Dataset images are organized into training and testing directories according to their respective class labels.

## Methodology
The overall experimental pipeline consists of the following stages:
1. Data loading and exploratory data analysis.
2. Image preprocessing and normalization.
3. Data augmentation.
4. Training and validation dataset preparation.
5. Transfer learning using ConvNeXt and ResNet.
6. Feature extraction from both backbones.
7. Feature-level fusion through concatenation.
8. Classification using fully connected layers.
9. Model evaluation and performance analysis.

## Technology Stack
| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Computer Vision | Torchvision |
| Model Architecture | ConvNeXt, ResNet |
| Model Library | timm |
| Data Processing | NumPy, Pandas |
| Image Processing | PIL |
| Visualization | Matplotlib |
| Evaluation | Scikit-learn |

## Project Structure
```text
Brain-Tumor-Classification/
│
├── assets/
│   └── architecture.png
│
├── data/
│   ├── Training/
│   └── Testing/
│
├── notebook/
│   ├── eda.ipynb
│   └── model_evaluation.ipynb
│
├── model/
│   └── (Model weights excluded from Git)
│
├── src/
│   ├── data.py
│   ├── model.py
│   └── train.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/daffasyarief/Brain-Tumor-Classification.git

cd Brain-Tumor-Classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

To start the training pipeline:

```bash
python run.py
```

Ensure that the dataset is available in the expected directory structure before running the training process.

## Model Weights

Trained model checkpoints are excluded from this repository to keep the source code repository lightweight.

The following model checkpoints are maintained separately:

- `HybridModel.pth`
- `ConvNeXt.pth`
- `ResNet.pth`

Model weights can be made available separately through external model storage.

## Evaluation Metrics

The model performance is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Experimental Results

Performance results will be updated after the final evaluation.

| Metric | Result |
|---|---|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |

## Future Work

Potential directions for future development include:

- Ablation studies to evaluate individual backbone contributions.
- Comparison of different feature fusion strategies.
- Explainable AI using Grad-CAM.
- Hyperparameter optimization.
- Model generalization and robustness evaluation.
- Deployment for medical image analysis applications.

## Disclaimer

This project is intended for educational and research purposes only. It is not designed to replace professional medical diagnosis, clinical assessment, or the judgment of qualified healthcare professionals.

