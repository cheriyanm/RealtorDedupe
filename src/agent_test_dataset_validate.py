import pandas as pd

df = pd.read_csv("data/agent_test_dataset.csv")

# Overall summary
print(f"Total Records:      {len(df):,}")
print(f"Unique True Agents: {df['true_agent_id'].nunique():,}")
print(f"Avg Records/Agent:  {len(df)/df['true_agent_id'].nunique():.1f}")

# By test case
print("\nRecords by Test Case:")
print(df.groupby("test_case")["record_id"].count().to_string())

# Missing fields summary
print("\nMissing Values:")
print(df.isnull().sum())
