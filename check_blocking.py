import pandas as pd

df = pd.read_csv('data/processed/candidate_pairs.csv')

print('Total Candidate Pairs:', len(df))
print('True Matches Found:   ', df['is_true_match'].sum())
print('Non Matches:          ', (df['is_true_match']==0).sum())
print('True Match Rate:      ', f"{df['is_true_match'].mean()*100:.1f}%")