import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from preprocess import load_data, preprocess
from utils import save_confusion_matrix, save_feature_importance

def main(data_path):
    df = load_data(data_path)
    df = preprocess(df)
    if 'Churn' not in df.columns:
        raise ValueError('Churn column missing')
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    # Train models
    lr = LogisticRegression(max_iter=300)
    lr.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    # Evaluate RF on test for printing and saving artifacts
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:,1]
    print('Random Forest Accuracy:', accuracy_score(y_test, y_pred))
    print('\nClassification Report:\n', classification_report(y_test, y_pred))
    # Save models and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/rf_model.pkl')
    joblib.dump(lr, 'models/lr_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    # Save reports
    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, 'reports/confusion_matrix.png')
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    save_feature_importance(importances, 'reports/feature_importance.png')
    # Save feature importance CSV
    importances.to_csv('reports/feature_importance.csv', header=['importance'])
    print('Models and reports saved to models/ and reports/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    args = parser.parse_args()
    main(args.data_path)
