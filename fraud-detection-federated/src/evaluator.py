"""
Evaluates model performance on test data using multiple metrics and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_type='keras', plot=True):
    """
    Evaluates a model and prints metrics.

    Args:
        model: Trained model (Keras or scikit-learn).
        X_test (array-like): Test features.
        y_test (array-like): True labels.
        model_type (str): 'keras' or 'sklearn'.
        plot (bool): Show plots (ROC, confusion matrix, PR).

    Returns:
        dict: Metric scores.
    """
    # Get predictions
    if model_type == 'keras':
        y_pred_probs = model.predict(X_test).ravel()
        y_pred = (y_pred_probs >= 0.5).astype(int)
    else:  # sklearn
        y_pred = model.predict(X_test)
        y_pred_probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    if plot:
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    # Precision-R
