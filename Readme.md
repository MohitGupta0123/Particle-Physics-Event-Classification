# 🧪 Particle Physics Event Classification

Distinguishing rare **Higgs-boson signal events** from overwhelming **background noise** in proton–proton collisions at CERN.
This project builds a robust machine-learning pipeline that leverages **ensemble learning (stacking)** to classify events using rich kinematic features from detector data.

---

## 📂 Project Structure

```
├── .gitattributes
├── .gitignore
├── Artifacts
│   ├── Best_artifacts.pkl
│   └── inference_pipeline.pkl
├── Baseline1.ipynb
├── Baseline2.ipynb
├── Data
│   ├── Dataset.csv
│   ├── Test.csv
│   ├── Train.csv
│   └── Transformed_Dataset.csv
├── EDA.ipynb
├── Mohit_Gupta.pdf
├── Mohit_Gupta.pptx
├── Readme.md
└── catboost_info
    ├── catboost_training.json
    ├── learn
    │   └── events.out.tfevents
    ├── learn_error.tsv
    └── time_left.tsv
```

---

## ⚙️ Methodology

### 🔹 Preprocessing

* Handle missing values (`ApplyTransformDataset`):

  * Replace infinite values with NaN
  * Impute with mean/median depending on outliers
  * Cap outliers using **z-score clipping**
* Drop irrelevant columns (`ColumnDropper`)

### 🔹 Model Architecture

* **Base Learners**:

  * XGBoost, LightGBM, CatBoost
  * Parameters: `n_estimators=700`, `max_depth=7`, `learning_rate=0.1`
* **Stacking Ensemble**:

  * Combines base learners using **Logistic Regression** as a meta-learner
  * Uses `predict_proba` outputs as features
  * 5-fold cross-validation

### 🔹 Training Setup

* **Dataset**:

  * Total: **200,000 events**
  * Signal (s): **68,534**
  * Background (b): **131,466**
  * Features used: **18** (from original 30)
* **Threshold Tuning**: 0.49 (balance recall vs. precision)

---

## 📊 Results

### ✅ Training Set (200k samples)

* **Accuracy**: `0.863`
* **ROC-AUC**: `0.934`
* Strong separation between signal & background.

### 📦 Cross-Validation (CV folds)

* **Accuracy**: `0.838 ± 0.002`
* **ROC-AUC**: `0.908 ± 0.002`

### 🧪 Test Set (50k samples)

* **Accuracy**: `0.838`
* **ROC-AUC**: `0.907`

🔍 **Observation**:

* Good generalization; mild overfitting (training slightly higher).
* More **false negatives** than false positives → threshold tuning could improve recall.

---

## 📌 How to Run

### 1️⃣ Train & Save Pipeline

```bash
Run Baseline2.ipynb
```

This will:

* Preprocess the dataset
* Train base learners + stacking ensemble
* Save pipeline as `artifacts/inference_pipeline.pkl`

### 2️⃣ Inference on New Data

```python
import pickle
import pandas as pd

with open("artifacts/inference_pipeline.pkl", "rb") as f:
    saved = pickle.load(f)

pipe = saved["pipeline"]
threshold = saved["threshold"]

df_new = pd.read_csv("Data/Test.csv")
y_prob = pipe.predict_proba(df_new)[:, 1]
y_pred = (y_prob >= threshold).astype(int)
```

---

## 🚀 Future Work

* Apply **calibration** & **regularization** to reduce overfitting.
* Explore **deep learning models** for richer feature interactions.
* Optimize threshold for better **recall of rare signal events**.

---