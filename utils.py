import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_confusion_matrix(cm, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No','Yes'], yticklabels=['No','Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_feature_importance(importances, out_path, top_n=15):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imp = importances.head(top_n)
    plt.figure(figsize=(10,5))
    imp.plot(kind='bar')
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
