import pandas as pd

df = pd.read_csv('data/processed/scored_pairs.csv')

print('Total Scored Pairs:', len(df))
print('Avg Similarity Score:', f"{df['similarity_score'].mean():.4f}")
print('Max Score:', f"{df['similarity_score'].max():.4f}")
print('Min Score:', f"{df['similarity_score'].min():.4f}")
print()
print('Decision Buckets:')
print('Auto Merge  (>=0.95):', len(df[df['similarity_score'] >= 0.95]))
print('LLM Review  (0.75-0.95):', len(df[(df['similarity_score'] >= 0.75) & (df['similarity_score'] < 0.95)]))
print('Keep Sep    (<0.75):', len(df[df['similarity_score'] < 0.75]))
print()
print('True Matches by Bucket:')
print('Auto Merge  true matches:', df[df['similarity_score'] >= 0.95]['is_true_match'].sum())
print('LLM Review  true matches:', df[(df['similarity_score'] >= 0.75) & (df['similarity_score'] < 0.95)]['is_true_match'].sum())
print('Keep Sep    true matches:', df[df['similarity_score'] < 0.75]['is_true_match'].sum())


# Add this to check_embeddings.py to find them!
import pandas as pd

df = pd.read_csv('data/processed/scored_pairs.csv')

# Find true matches we're going to miss
missed = df[(df['similarity_score'] < 0.75) & 
            (df['is_true_match'] == 1)]

print('Missed True Matches:')
print(missed[['record_id_1', 'record_id_2', 
              'similarity_score', 
              'true_agent_id_1']].to_string())

