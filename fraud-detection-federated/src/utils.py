"""
Utility functions for logging metrics, visualizing results, and exporting reports.
"""

import json
import csv
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def log_metrics(metrics_dict, prefix=""):
    """
    Logs training or evaluation metrics in formatted output.

    Args:
        metrics_dict (dict): Dictionary of metric name to value.
        prefix (str): Optional text to prefix logs.
    """
    print(f"{prefix} Metrics:")
    for key, val in metrics_dict.items():
        print(f"  {key}: {val:.4f}")

def save_metrics(metrics_dict, filepath, fmt='json'):
    """
    Saves metrics to a file in JSON or CSV format.

    Args:
        metrics_dict (dict): Dictionary of metrics.
        filepath (str): Output file path.
        fmt (str): Format ('json' or 'csv').
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if fmt == 'json':
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
    elif fmt == 'csv':
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in metrics_dict.items():
                writer.writerow([k, v])
    else:
        raise ValueError("Unsupported format. Use 'json' or 'csv'.")

    print(f"Metrics saved to {filepath}")

def display_classification_report(y_true, y_pred):
    """
    Prints classification report.
    """
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    Plots a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Example usage for standalone testing
if __name__ == "__main__":
    # Mock metrics
    mock_metrics = {
        "loss": 0.356,
        "accuracy": 0.894,
        "precision": 0.774,
        "recall": 0.620,
        "f1_score": 0.687
    }

    log_metrics(mock_metrics, prefix="Test")
    save_metrics(mock_metrics, "outputs/metrics.json", fmt='json')
    save_metrics(mock_metrics, "outputs/metrics.csv", fmt='csv')

    # Mock classification data
    y_true = [0, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0, 0, 1, 0, 0, 1, 1, 1]

    display_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, labels=["Legit", "Fraud"])
