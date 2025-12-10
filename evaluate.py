import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from preprocess import load_data, preprocess
from utils import save_confusion_matrix

def main(data_path):
    df = load_data(data_path)
    df = preprocess(df)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    # load artifacts
    rf = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X[num_cols] = scaler.transform(X[num_cols])
    y_pred = rf.predict(X)
    y_proba = rf.predict_proba(X)[:,1]
    print('Accuracy:', accuracy_score(y, y_pred))
    print('\nClassification Report:\n', classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    save_confusion_matrix(cm, 'reports/confusion_matrix_full.png')
    try:
        auc = roc_auc_score(y, y_proba)
        print('\nROC AUC:', auc)
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    args = parser.parse_args()
    main(args.data_path)
