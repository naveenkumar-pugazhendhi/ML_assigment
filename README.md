# 🔍 Financial Fraud Detection Using Machine Learning

A machine learning project that investigates fraud detection on financial transactions using five research questions — from baseline models to hyperparameter tuning.

---

## 📁 Project Structure

```
financial-fraud-detection/
│
├── RQ1_Baseline_Models.ipynb            # Baseline model comparison
├── RQ2_Feature_Importance.ipynb         # Feature importance analysis
├── RQ3_Class_Imbalance.ipynb            # Class imbalance handling
├── RQ4_Ensemble_Models.ipynb            # Advanced ensemble models
├── RQ5_Hyperparameter_Tuning.ipynb      # Hyperparameter tuning
│
├── results/
│   ├── table_rq1_baseline_comparison.csv
│   ├── table_rq2_feature_importance.csv
│   ├── table_rq3_imbalance_handling.csv
│   ├── table_rq4_ensemble_models.csv
│   ├── table_rq5_tuning_comparison.csv
│   ├── table_rq5_best_params.csv
│   ├── figure_rq1_baseline_comparison.pdf
│   ├── figure_rq2_feature_importance.pdf
│   ├── figure_rq3_imbalance_handling.pdf
│   ├── figure_rq4_ensemble_models.pdf
│   └── figure_rq5_hyperparameter_tuning.pdf
│
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

**Name:** Financial Transactions Dataset for Fraud Detection  
**Source:** [Kaggle - aryan208/financial-transactions-dataset-for-fraud-detection](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection)  
**Rows:** 50,000 (used) | **Columns:** 18  
**Target Variable:** `is_fraud` (0 = Legitimate, 1 = Fraud)  
**Task:** Binary Classification

### Key Features Used
| Feature | Description |
|---------|-------------|
| `amount` | Transaction amount |
| `velocity_score` | Transaction frequency score |
| `spending_deviation_score` | Deviation from normal spending |
| `geo_anomaly_score` | Geographic anomaly indicator |
| `time_since_last_transaction` | Time gap between transactions |
| `transaction_type` | Type of transaction |
| `merchant_category` | Merchant category code |
| `payment_channel` | Payment method used |

### Dropped Columns (leakage/non-informative)
`transaction_id`, `timestamp`, `sender_account`, `receiver_account`, `fraud_type`, `ip_address`, `device_hash`

---

## ❓ Research Questions

| # | Research Question | Notebook |
|---|-------------------|----------|
| RQ1 | Which baseline ML models best detect financial fraud? | `RQ1_Baseline_Models.ipynb` |
| RQ2 | Which features are most important for predicting fraud? | `RQ2_Feature_Importance.ipynb` |
| RQ3 | How does handling class imbalance improve fraud detection? | `RQ3_Class_Imbalance.ipynb` |
| RQ4 | Do advanced ensemble models outperform baseline models? | `RQ4_Ensemble_Models.ipynb` |
| RQ5 | How does hyperparameter tuning improve the best model? | `RQ5_Hyperparameter_Tuning.ipynb` |

---

## 🧪 Methodology

Each notebook follows this pipeline:

```
Load Dataset (50,000 rows)
        ↓
Drop leakage & non-informative columns
        ↓
Label encode categorical features
        ↓
Drop missing values
        ↓
80/20 Train/Test Split (stratified)
        ↓
Train ML Models
        ↓
Evaluate: Accuracy, Precision, Recall, F1, AUC-ROC
        ↓
Save CSV table + PDF figure
```

---

## 🤖 Models Used

| RQ | Models |
|----|--------|
| RQ1 | Logistic Regression, Random Forest, Decision Tree, KNN, Naive Bayes |
| RQ2 | Random Forest, Gradient Boosting (feature importance) |
| RQ3 | Random Forest + SMOTE / Undersampling / Class Weighting |
| RQ4 | Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, Stacking |
| RQ5 | XGBoost Default vs XGBoost Tuned (RandomizedSearchCV) |

---

## 📈 Evaluation Metrics

| Metric | Why Used |
|--------|----------|
| **Accuracy** | Overall correctness |
| **Precision** | Avoids false fraud alerts |
| **Recall** | Most critical — catches actual fraud |
| **F1-Score** | Balance between Precision & Recall |
| **AUC-ROC** | Discrimination across all thresholds |
| **CV AUC Mean/Std** | Model stability (RQ5 only) |

> ⚠️ **Recall and F1 are the most important metrics** — missing real fraud is far more costly than a false alarm.

---

## 🚀 How to Run

### Option 1: Run on Kaggle (Recommended)

1. Go to [Kaggle](https://www.kaggle.com) and sign in
2. Open each `.ipynb` file and upload to a new Kaggle notebook:
   - `File → Import Notebook → Select .ipynb`
3. Add the dataset:
   - Right panel → `+ Add Data` → `Your Datasets` → Search **"Financial Transactions Dataset for Fraud Detection"** → Add
4. Run cells one by one using `Shift + Enter`
5. Outputs (CSV + PDF) will be saved in the Kaggle working directory

### Option 2: Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/financial-fraud-detection.git
cd financial-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
# Place the CSV file at: data/financial_fraud_detection_dataset.csv

# 4. Update DATA_PATH in each notebook:
DATA_PATH = 'data/financial_fraud_detection_dataset.csv'

# 5. Launch Jupyter
jupyter notebook
```

---

## 📦 Requirements

See `requirements.txt` for full list. Key libraries:

```
pandas, numpy, matplotlib, scikit-learn,
xgboost, lightgbm, imbalanced-learn, jupyter
```

---

## 📂 How to Download Dataset

1. Visit: https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection
2. Click **Download**
3. Extract and place `financial_fraud_detection_dataset.csv` in the project folder
4. Update `DATA_PATH` in each notebook to point to the file

---

## 📋 Results Summary

| RQ | Best Model | Best F1 | Best AUC-ROC |
|----|-----------|---------|-------------|
| RQ1 | Random Forest | ~0.70 | ~0.88 |
| RQ2 | Top feature: velocity_score | — | — |
| RQ3 | SMOTE + RF | ~0.67 | ~0.87 |
| RQ4 | XGBoost / LightGBM | ~0.75 | ~0.93 |
| RQ5 | Tuned XGBoost | ~0.80 | ~0.96 |

---

## 👤 Author

**Naveenkumar**  
Machine Learning Assignment — Fraud Detection  
Dataset: Financial Transactions Dataset for Fraud Detection (Kaggle)
