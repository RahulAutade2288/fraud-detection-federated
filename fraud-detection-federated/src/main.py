
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime

# Load dataset
df = pd.read_csv("../data/synthetic_data.csv")

# === Summary Statistics ===
def summarize_data(df):
    print("=== Summary Statistics ===")
    print(df.describe(include='all'))
    print("\nMissing Values:")
    print(df.isnull().sum())

# === Fraud Stats ===
def fraud_stats(df):
    fraud_count = df["is_fraud"].sum()
    total = len(df)
    fraud_rate = fraud_count / total * 100
    print(f"\nTotal Transactions: {total}")
    print(f"Fraudulent Transactions: {fraud_count}")
    print(f"Fraud Rate: {fraud_rate:.2f}%")

# === Visualize Fraud vs Legit ===
def plot_fraud_distribution(df):
    plt.figure(figsize=(6,4))
    df['is_fraud'].value_counts().plot(kind='bar', color=['blue', 'orange'])
    plt.title("Fraud vs Legit Transactions")
    plt.xticks([0, 1], ['Legit', 'Fraud'])
    plt.xlabel("Transaction Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("../docs/fraud_distribution.png")
    plt.close()

# === Transaction Amounts ===
def plot_amount_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['amount'], bins=30, kde=True)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("../docs/amount_distribution.png")
    plt.close()

# === Fraud by Transaction Type ===
def fraud_by_transaction_type(df):
    type_fraud = df.groupby('transaction_type')['is_fraud'].mean()
    print("\nFraud Rate by Transaction Type:")
    print(type_fraud)

# === High Risk Locations ===
def high_risk_locations(df):
    location_fraud = df.groupby('location')['is_fraud'].mean().sort_values(ascending=False)
    print("\nHigh Risk Locations (by fraud rate):")
    print(location_fraud)

# === Top Users with Most Fraud ===
def top_users_by_fraud(df, top_n=5):
    user_fraud = df[df['is_fraud'] == 1].groupby('user_id').size().sort_values(ascending=False).head(top_n)
    print(f"\nTop {top_n} Users with Most Fraudulent Transactions:")
    print(user_fraud)

# === Time Series of Transactions ===
def plot_time_series(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    ts = df.set_index('timestamp').resample('D').size()
    plt.figure(figsize=(10, 4))
    ts.plot()
    plt.title("Daily Transaction Volume")
    plt.xlabel("Date")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig("../docs/daily_transactions.png")
    plt.close()

# === CLI Handler ===
def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Data Analysis")
    parser.add_argument('--summary', action='store_true', help='Show data summary')
    parser.add_argument('--fraud', action='store_true', help='Show fraud statistics')
    parser.add_argument('--charts', action='store_true', help='Generate charts')
    parser.add_argument('--risks', action='store_true', help='Show risk analysis')
    args = parser.parse_args()

    if args.summary:
        summarize_data(df)
    if args.fraud:
        fraud_stats(df)
    if args.charts:
        plot_fraud_distribution(df)
        plot_amount_distribution(df)
        plot_time_series(df)
    if args.risks:
        fraud_by_transaction_type(df)
        high_risk_locations(df)
        top_users_by_fraud(df)

if __name__ == "__main__":
    main()
