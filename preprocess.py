import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def clean_total_charges(df):
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    return df

def preprocess(df):
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    df = clean_total_charges(df)
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)
    # Binary encode Yes/No
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
    # One-hot remaining categoricals
    cat = df.select_dtypes(include='object').columns.tolist()
    if cat:
        df = pd.get_dummies(df, columns=cat, drop_first=True)
    # Ensure target is 0/1
    if 'Churn' in df.columns and df['Churn'].dtype != 'int64':
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df
