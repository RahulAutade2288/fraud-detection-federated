"""
Loads and preprocesses transaction data for fraud detection and federated learning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)

def load_raw_data(file_path):
    """
    Loads raw transaction data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def validate_columns(df, required_cols):
    """
    Validates that the required columns exist in the dataset.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def preprocess_data(df, features, target='is_fraud'):
    """
    Selects features, encodes categoricals, scales numerics.
    """
    validate_columns(df, features + [target])
    df = df.copy()

    # Encode categorical features
    for col in df[features].select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        logging.info(f"Encoded column: {col}")

    X = df[features].values
    y = df[target].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Features scaled.")

    return X_scaled, y

def load_and_preprocess(file_path, features, target='is_fraud'):
    df = load_raw_data(file_path)
    return preprocess_data(df, features, target)

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def split_clients(X, y, num_clients=3):
    """
    Splits data for simulated federated learning clients.
    """
    client_data = []
    X_split = np.array_split(X, num_clients)
    y_split = np.array_split(y, num_clients)
    for i in range(num_clients):
        client_data.append((X_split[i], y_split[i]))
        logging.info(f"Client {i+1} data size: {len(X_split[i])}")
    return client_data

# Example usage for testing
if __name__ == "__main__":
    file = "../data/synthetic_data.csv"
    features = ['amount', 'is_international', 'merchant_id']
    
    try:
        X, y = load_and_preprocess(file, features)
        X_train, X_test, y_train, y_test = split_data(X, y)
        clients = split_clients(X_train, y_train, num_clients=3)

        print("Data preprocessing complete.")
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    except Exception as ex:
        logging.error(f"Execution failed: {ex}")
