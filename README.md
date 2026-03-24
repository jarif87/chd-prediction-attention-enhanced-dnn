# A Comparative Study of Machine Learning Models for Coronary Heart Disease Prediction with an Attention-Based Deep Learning Approach

## Description

This repository contains the full source code, notebooks, and trained model weights for a comparative study evaluating six machine learning models for coronary heart disease (CHD) prediction using the Framingham Heart Study dataset. The study extends prior work by Bakhtawar et al. (2024) by employing a larger and more representative dataset, addressing class imbalance, expanding model diversity, and introducing a novel **Attention-Enhanced Deep Neural Network (DNN)** with a feature-level TabAttention mechanism for clinical interpretability.

All six models surpassed the 83% accuracy benchmark from prior work. The proposed Attention-Enhanced DNN achieved **90.00% accuracy**, **94.72% recall**, and **94.94% ROC-AUC**, while Random Forest achieved the highest overall accuracy of **96.25%** with a ROC-AUC of **99.13%**.

---

## Dataset Information

- **Name:** Framingham Heart Study Dataset
- **Source:** Originally from the National Heart, Lung, and Blood Institute (NHLBI) BioLINCC Teaching Repository: https://biolincc.nhlbi.nih.gov/teaching/
- **GitHub Mirror Used:** https://github.com/GauravPadawe/Framingham-Heart-Study
- **Records:** 4,240 patient records
- **Features:** 16 clinical features (demographic + clinical variables):
  - Gender, Age, Education, Current Smoker, Cigarettes Per Day, BP Medication, Prevalent Stroke, Prevalent Hypertension, Diabetes, Total Cholesterol, Systolic Blood Pressure, Diastolic Blood Pressure, BMI, Heart Rate, Glucose
  - One engineered feature: **Pulse Pressure** (Systolic BP − Diastolic BP), resulting in 17 total features
- **Target Variable:** `TenYearCHD` — binary label (1 = CHD risk within 10 years, 0 = no risk)
- **Class Imbalance:** ~85% negative cases, ~15% positive CHD cases (addressed via random upsampling)
- **Balanced Dataset Size:** 7,188 samples (3,594 per class after upsampling)

---

## Code Information

- **Language:** Python 3
- **Notebooks:** Jupyter Notebooks (.ipynb) for each model and preprocessing pipeline
- **Trained Model Weights:** Best checkpoint of the Attention-Enhanced DNN saved as `.pt` file (PyTorch)
- **Repository:** https://github.com/jarif87/chd-prediction-attention-enhanced-dnn

---

## Usage Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/jarif87/chd-prediction-attention-enhanced-dnn.git
cd chd-prediction-attention-enhanced-dnn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the Framingham Heart Study dataset from:
- https://github.com/GauravPadawe/Framingham-Heart-Study

Place the CSV file in the `data/` directory.

### 4. Run the Notebooks

- **Preprocessing & Baseline Models (KNN, Decision Tree, Random Forest):** Open and run `baseline_models.ipynb`
- **Advanced Models (XGBoost, LightGBM):** Open and run `advanced_models.ipynb`
- **Attention-Enhanced DNN:** Open and run `attention_dnn.ipynb`

### 5. Load Trained DNN Model (Inference Only)

```python
import torch
from model import AttentionDNN  # import model class from model.py

model = AttentionDNN(input_dim=17)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))
model.eval()
```

---

## Requirements

The following Python libraries are required:

```
Python >= 3.8
numpy
pandas
scikit-learn
xgboost
lightgbm
torch >= 1.13
matplotlib
seaborn
imbalanced-learn
jupyter
```

Install all dependencies at once:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm torch matplotlib seaborn imbalanced-learn jupyter
```

**Hardware used during experiments:**
- Baseline and advanced models: NVIDIA P100 GPU (Kaggle)
- Attention-Enhanced DNN: NVIDIA T4 GPU (Google Colaboratory)

---

## Methodology

### 1. Data Preprocessing
- **Missing value imputation:** Continuous variables (Cigarettes Per Day, Total Cholesterol, BMI, Heart Rate, Glucose) imputed with median; categorical variables (Education, BP Medication) imputed with mode.
- **Feature engineering:** Pulse Pressure = Systolic BP − Diastolic BP (added as 17th feature).
- **Class imbalance handling:** Random upsampling with replacement applied to the minority class (random seed = 42), producing 7,188 balanced samples.
- **Data splitting:** 80/10/10 stratified split → 5,750 training / 719 validation / 719 test samples.
- **Normalization:** StandardScaler fitted on training set only; applied to validation and test sets to prevent data leakage.

### 2. Models Evaluated
| Model | Type |
|---|---|
| K-Nearest Neighbors (KNN) | Baseline (k=5, Euclidean distance) |
| Decision Tree (DT) | Baseline (CART, Gini impurity) |
| Random Forest (RF) | Baseline (200 estimators) |
| XGBoost | Advanced (500 estimators, lr=0.05, max depth=6) |
| LightGBM | Advanced (500 estimators, lr=0.05, 31 leaves) |
| Attention-Enhanced DNN | Proposed novel model |

### 3. Attention-Enhanced DNN Architecture
- **TabAttention module:** Linear transformation + Softmax → element-wise feature weighting
- **Block 1:** Linear(17→128) + BatchNorm1d + GELU + Dropout(0.25) + skip connection
- **Block 2:** Linear(128→128) + BatchNorm1d + GELU + Dropout(0.25)
- **Block 3:** Linear(128→64) + BatchNorm1d + GELU + Dropout(0.20) + skip connection
- **Output:** Linear(64→32) + GELU → Linear(32→1) + Sigmoid
- **Trainable parameters:** ~50,000
- **Optimizer:** Adam (lr=3×10⁻⁴, weight decay=1×10⁻⁴)
- **Loss:** Binary Cross-Entropy
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Training:** Max 400 epochs, early stopping (patience=40), best checkpoint saved on minimum validation loss

### 4. Evaluation
All models evaluated using: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**  
Baseline and advanced models: Stratified 10-fold cross-validation + held-out test set  
Attention-Enhanced DNN: Fixed held-out test set with best model checkpoint  
Global random seed: **42** (set across Python, NumPy, PyTorch for reproducibility)

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| KNN | 79.64% | 73.93% | 91.52% | 81.79% | 87.29% |
| Decision Tree | 89.16% | 83.47% | 97.64% | 90.00% | 89.17% |
| Random Forest | **96.25%** | 94.99% | 97.64% | 96.30% | **99.13%** |
| XGBoost | 85.48% | — | — | — | 90.75% |
| LightGBM | 89.85% | — | — | — | 95.45% |
| Attention-Enhanced DNN | 90.00% | 86.55% | 94.72% | 90.45% | 94.94% |

---

## Citations

If you use this code or dataset in your research, please cite:

**This paper:**
> Sadik Al Jarif. A Comparative Study of Machine Learning Models for Coronary Heart Disease Prediction with an Attention-Based Deep Learning Approach. *PeerJ Computer Science* (under review).

**Dataset:**
> Dawber TR, Meadors GF, and Moore FE. 1951. Epidemiological approaches to heart disease: the Framingham Heart Study. *American Journal of Public Health* 41(3):279–281. DOI: 10.2105/AJPH.41.3.279

> National Heart, Lung, and Blood Institute (NHLBI). Framingham Heart Study Teaching Dataset. BioLINCC. https://biolincc.nhlbi.nih.gov/teaching/

**Baseline reference:**
> Bakhtawar HK, et al. 2024. Heart disease prediction using machine learning. In: *Proceedings of the 2nd DMIHER International Conference on Artificial Intelligence in Healthcare, Education and Industry (IDICAIEI)*. IEEE. DOI: 10.1109/IDICAIEI61867.2024.10842908

---

## License

This project is released for academic and research purposes. Please refer to the repository for the full license terms.

## Contribution Guidelines

This repository is maintained by the corresponding author. For questions, issues, or suggestions, please open an issue on GitHub or contact: sadikaljarif05@gmail.com
