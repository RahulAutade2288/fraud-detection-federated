"""
Model Inference Script for fraud-detection-federated
Loads a trained model and performs inference on sample transactions.
"""

import numpy as np
import pandas as pd
import argparse
import os
from tensorflow.keras.models import load_model

def load_sample_data(from_csv=False, csv_path=None):
    """
    Loads input data for prediction.

    Args:
        from_csv (bool): If True, loads from a CSV file.
        csv_path (str): Path to CSV file (must have numerical features).

    Returns:
        np.ndarray: Input features.
    """
    if from_csv and csv_path:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples from CSV.")
        return df.values
    else:
        print("Using sample transaction amounts.")
        return np.array([
            [120.0, 0, 531],
            [5000.0, 1, 544],
            [75.5, 0, 510]
        ])

def load_model_for_inference(model_path):
    """
    Loads a trained Keras model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = load_model(model_path)
    model.summary()
    return model

def run_inference(model, data, threshold=0.5, save_output=False, output_path="predictions.csv"):
    """
    Runs model inference on input data.

    Args:
        model: Trained Keras model.
        data (np.ndarray): Input features.
        threshold (float): Classification threshold.
        save_output (bool): Whether to save results to CSV.
        output_path (str): Output file path if saving.
    """
    predictions = model.predict(data)
    print("\n=== Inference Results ===")
    results = []
    for i, row in enumerate(data):
        prob = predictions[i][0]
        is_fraud = int(prob >= threshold)
        print(f"Input: {row} -> Fraud Probability: {prob:.4f} -> Classified as: {'FRAUD' if is_fraud else 'LEGIT'}")
        results.append(list(row) + [prob, is_fraud])

    if save_output:
        col_names = [f"feature_{i+1}" for i in range(data.shape[1])] + ["fraud_prob", "is_fraud"]
        df = pd.DataFrame(results, columns=col_names)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Inference Script")
    parser.add_argument("--model", type=str, default="../models/simple_model.h5", help="Path to trained model")
    parser.add_argument("--csv", type=str, help="Path to input CSV file (optional)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--save", action='store_true', help="Save predictions to CSV")
    args = parser.parse_args()

    input_data = load_sample_data(from_csv=bool(args.csv), csv_path=args.csv)
    model = load_model_for_inference(args.model)
    run_inference(model, input_data, threshold=args.threshold, save_output=args.save)

if __name__ == "__main__":
    main()
