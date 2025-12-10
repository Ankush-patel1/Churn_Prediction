# Customer Churn Prediction (Telecom)

Professional, production-ready project for predicting customer churn using the Kaggle Telco dataset.

## Features
- Data loading and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering and encoding
- Model training (Logistic Regression, Random Forest)
- Evaluation: accuracy, classification report, confusion matrix, ROC AUC
- Save trained model and scaler

## Repo structure
```
churn-prediction/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/                         # place WA_Fn-UseC_-Telco-Customer-Churn.csv here
├─ notebooks/
│  └─ customer_churn_kaggle.ipynb
├─ src/
│  ├─ preprocess.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ utils.py
├─ models/                       # trained artifacts saved here after running train.py
└─ reports/                      # generated plots saved here
```

## Quick start
1. Put `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/`.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Train models:
```bash
python src/train.py --data_path data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```
4. Evaluate:
```bash
python src/evaluate.py --data_path data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The notebook includes the same pipeline and is suitable for presentation or further editing.
