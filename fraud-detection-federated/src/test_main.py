
import unittest
import pandas as pd
import os
import subprocess

DATA_PATH = "../data/synthetic_data.csv"

class TestDataIntegrity(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(DATA_PATH)

    def test_required_columns_exist(self):
        required_columns = [
            "transaction_id", "user_id", "amount", "transaction_type",
            "location", "timestamp", "device_type", "merchant_id",
            "is_international", "is_fraud"
        ]
        for col in required_columns:
            self.assertIn(col, self.df.columns)

    def test_fraud_column_binary(self):
        self.assertTrue(self.df["is_fraud"].isin([0, 1]).all())

    def test_no_missing_values(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_row_count_and_column_count(self):
        self.assertGreaterEqual(len(self.df), 50)
        self.assertGreaterEqual(len(self.df.columns), 7)

    def test_valid_transaction_types(self):
        valid_types = {"online", "instore", "transfer"}
        self.assertTrue(set(self.df["transaction_type"]).issubset(valid_types))

    def test_valid_locations(self):
        self.assertGreaterEqual(self.df["location"].nunique(), 3)

    def test_timestamp_format(self):
        try:
            pd.to_datetime(self.df["timestamp"])
            success = True
        except Exception:
            success = False
        self.assertTrue(success)

    def test_fraud_distribution(self):
        fraud_count = self.df["is_fraud"].sum()
        total = len(self.df)
        fraud_rate = fraud_count / total
        self.assertGreaterEqual(fraud_rate, 0.01)
        self.assertLessEqual(fraud_rate, 0.5)

class TestMainCLI(unittest.TestCase):
    def run_script_with_arg(self, arg):
        result = subprocess.run(
            ["python3", "../src/main.py", arg],
            capture_output=True, text=True
        )
        return result.stdout, result.stderr

    def test_summary_flag(self):
        out, err = self.run_script_with_arg("--summary")
        self.assertIn("Summary Statistics", out)

    def test_fraud_flag(self):
        out, err = self.run_script_with_arg("--fraud")
        self.assertIn("Fraudulent Transactions", out)

if __name__ == "__main__":
    unittest.main()
