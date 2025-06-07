"""
Applies normalization and preprocessing to input features.
Supports MinMaxScaler and StandardScaler.
"""

import numpy as np
import pandas as pd
import argparse
import os
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data(X, method="minmax", scaler=None):
    """
    Normalizes data using MinMax or Standard scaling.

    Args:
        X (np.ndarray): Input data.
        method (str): 'minmax' or 'standard'.
        scaler: Optional pretrained scaler.

    Returns:
        np.ndarray: Scaled data.
        scaler: Trained scaler object.
    """
    if scaler is None:
        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError("Unsupported scaling method")

        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler

def inverse_transform(X_scaled, scaler):
    """
    Converts scaled data back to original scale.
    """
    return scaler.inverse_transform(X_scaled)

def save_scaler(scaler, path):
    """
    Saves the scaler to a file.
    """
    joblib.dump(scaler, path)

def load_scaler(path):
    """
    Loads a scaler from file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found: {path}")
    return joblib.load(path)

def normalize_csv(input_path, output_path, method="minmax"):
    """
    Normalizes numeric columns from CSV and writes output.
    """
    df = pd.read_csv(input_path)
    num_cols = df.select_dtypes(include=[np.number]).columns
    X = df[num_cols].values

    X_scaled, scaler = normalize_data(X, method=method)
    df_scaled = pd.DataFrame(X_scaled, columns=num_cols)
    df_scaled.to_csv(output_path, index=False)

    print(f"Normalized data saved to {output_path}")
    return scaler

# CLI usage
def main():
    parser = argparse.ArgumentParser(description="Normalize CSV data using MinMax or Standard scaler")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, default="normalized_output.csv", help="Output CSV file")
    parser.add_argument("--method", choices=["minmax", "standard"], default="minmax", help="Scaling method")
    parser.add_argument("--save_scaler", type=str, help="Path to save scaler object")
    args = parser.parse_args()

    scaler = normalize_csv(args.input, args.output, method=args.method)

    if args.save_scaler:
        save_scaler(scaler, args.save_scaler)
        print(f"Scaler saved to {args.save_scaler}")

# Run standalone
if __name__ == "__main__":
    # Sample test
    X_dummy = np.array([[100], [2000], [500], [1500]])
    X_scaled, scaler = normalize_data(X_dummy, method="standard")

    print("Original:\n", X_dummy)
    print("Scaled:\n", X_scaled)
    print("Inversed:\n", inverse_transform(X_scaled, scaler))

    # Uncomment below to run as CLI
    # main()
