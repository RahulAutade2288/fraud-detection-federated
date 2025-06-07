"""
Simulates multiple clients by splitting dataset.
Enhanced for edge case handling, analysis, and reproducibility.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)

def ensure_numpy(X, y):
    """
    Converts pandas DataFrames or Series to numpy arrays.
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.values
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values
    return X, y

def stratified_split(X, y, num_clients):
    """
    Attempts a stratified split to preserve label proportions across clients.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
    client_data = []
    for _, idx in skf.split(X, y):
        client_data.append((X[idx], y[idx]))
    return client_data

def analyze_distribution(y, title="Label Distribution"):
    """
    Plots the distribution of labels in a pie chart.
    """
    counter = Counter(y)
    labels = list(counter.keys())
    sizes = list(counter.values())
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.show()

def split_clients(X, y, num_clients=3, shuffle=True, seed=None, stratify=False, analyze=False):
    """
    Splits dataset into federated clients.

    Args:
        X (array-like): Features.
        y (array-like): Labels.
        num_clients (int): Number of clients.
        shuffle (bool): Shuffle before split.
        seed (int): Random seed.
        stratify (bool): Enable stratified splitting.
        analyze (bool): Plot label distribution for each client.

    Returns:
        List of (X_i, y_i) tuples for each simulated client.
    """
    X, y = ensure_numpy(X, y)

    if shuffle and not stratify:
        if seed is not None:
            np.random.seed(seed)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    if stratify:
        logging.info("Performing stratified split...")
        client_data = stratified_split(X, y, num_clients)
    else:
        X_split = np.array_split(X, num_clients)
        y_split = np.array_split(y, num_clients)
        client_data = list(zip(X_split, y_split))

    for i, (Xi, yi) in enumerate(client_data):
        logging.info(f"Client {i+1}: {Xi.shape[0]} samples, fraud rate: {np.mean(yi):.2f}")
        if analyze:
            analyze_distribution(yi, title=f"Client {i+1} Label Distribution")

    return client_data

# Example usage
if __name__ == "__main__":
    from sklearn.datasets imp
