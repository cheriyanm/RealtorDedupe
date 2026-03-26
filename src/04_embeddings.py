"""
RealtorDedupe - Step 4: Embeddings
=====================================
Converts each agent record into a vector embedding
then calculates cosine similarity for each candidate pair.

Process:
1. Load normalized agent records
2. Build a text representation of each agent
3. Convert each agent record to a vector embedding
4. For each candidate pair, calculate cosine similarity
5. Save pairs with similarity scores

Output:
- scored_pairs.csv → candidate pairs with similarity scores
"""

import pandas as pd
import numpy as np
import os
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    PROCESSED_DATA_PATH,
    CANDIDATE_PAIRS_PATH,
    SCORED_PAIRS_PATH,
    EMBEDDING_MODEL,
)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD AGENT TEXT REPRESENTATION
# Combine all normalized fields into one text string
# This is what gets converted to an embedding
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_text(row):
    """
    Combine all normalized fields into a single text string.
    This captures the full context of the agent record.

    Example output:
    "name: thomas johnson phone: 7045551234
     email: tjohnson@gmail.com license: 12345 state: nc"
    """
    parts = []

    # Name is the most important field
    if pd.notna(row.get('norm_name')):
        parts.append(f"name: {row['norm_name']}")

    # Phone
    if pd.notna(row.get('norm_phone')):
        parts.append(f"phone: {row['norm_phone']}")

    # Email
    if pd.notna(row.get('norm_email')):
        parts.append(f"email: {row['norm_email']}")

    # License
    if pd.notna(row.get('norm_license')):
        parts.append(f"license: {row['norm_license']}")

    # State
    if pd.notna(row.get('state')):
        parts.append(f"state: {str(row['state']).lower()}")

    # Office
    if pd.notna(row.get('norm_office')):
        parts.append(f"office: {row['norm_office']}")

    return ' '.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE EMBEDDINGS
# Convert each agent text to a vector embedding
# ─────────────────────────────────────────────────────────────────────────────

def generate_embeddings(df, model):
    """
    Generate embeddings for all agent records.
    Returns a dict of record_id → embedding vector
    """
    print(f"  Building text representations...")
    df['agent_text'] = df.apply(build_agent_text, axis=1)

    # Show a sample text representation
    print(f"\n  Sample agent text representation:")
    print(f"  {df['agent_text'].iloc[0]}")
    print(f"  {df['agent_text'].iloc[1]}")

    print(f"\n  Generating embeddings for {len(df):,} records...")
    print(f"  This may take a moment...")

    # Generate all embeddings at once (batch processing = faster!)
    embeddings = model.encode(
        df['agent_text'].tolist(),
        batch_size=32,
        show_progress_bar=True,
    )

    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Each agent = vector of {embeddings.shape[1]} numbers")

    # Return as dict: record_id → embedding
    return dict(zip(df['record_id'], embeddings))


# ─────────────────────────────────────────────────────────────────────────────
# SCORE CANDIDATE PAIRS
# Calculate cosine similarity for each candidate pair
# ─────────────────────────────────────────────────────────────────────────────

def score_pairs(pairs_df, embedding_dict):
    """
    For each candidate pair, calculate cosine similarity
    between their embedding vectors.
    Score ranges from 0 (completely different) to 1 (identical)
    """
    scores = []

    print(f"\n  Scoring {len(pairs_df):,} candidate pairs...")

    for _, row in pairs_df.iterrows():
        id1 = row['record_id_1']
        id2 = row['record_id_2']

        # Get embeddings for both records
        emb1 = embedding_dict.get(id1)
        emb2 = embedding_dict.get(id2)

        if emb1 is not None and emb2 is not None:
            # Calculate cosine similarity
            # Reshape needed for sklearn's cosine_similarity function
            score = cosine_similarity(
                emb1.reshape(1, -1),
                emb2.reshape(1, -1)
            )[0][0]
            scores.append(round(float(score), 4))
        else:
            scores.append(None)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("  STEP 4 — EMBEDDINGS & SIMILARITY SCORING")
    print("="*55)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n📂 Loading normalized agents from: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"   Records loaded: {len(df):,}")

    print(f"\n📂 Loading candidate pairs from: {CANDIDATE_PAIRS_PATH}")
    pairs_df = pd.read_csv(CANDIDATE_PAIRS_PATH)
    print(f"   Pairs loaded: {len(pairs_df):,}")

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"\n🤖 Loading embedding model: {EMBEDDING_MODEL}")
    print(f"   (downloading if first time — may take a minute...)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"   Model loaded! ✅")

    # ── Generate embeddings ───────────────────────────────────────────────────
    print(f"\n🔢 Generating embeddings...")
    embedding_dict = generate_embeddings(df, model)

    # ── Score candidate pairs ─────────────────────────────────────────────────
    print(f"\n📊 Scoring candidate pairs...")
    pairs_df['similarity_score'] = score_pairs(pairs_df, embedding_dict)

    # ── Score Distribution ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  SIMILARITY SCORE DISTRIBUTION")
    print(f"{'='*55}")

    scored = pairs_df.dropna(subset=['similarity_score'])

    # Overall distribution
    print(f"\n  Score ranges across all {len(scored):,} pairs:")
    bins = [0, 0.5, 0.7, 0.75, 0.85, 0.90, 0.95, 1.01]
    labels = ['<0.50', '0.50-0.70', '0.70-0.75',
              '0.75-0.85', '0.85-0.90', '0.90-0.95', '>0.95']

    scored['score_range'] = pd.cut(
        scored['similarity_score'],
        bins=bins,
        labels=labels,
        right=False
    )

    dist = scored.groupby('score_range', observed=True).agg(
        count=('similarity_score', 'count'),
        true_matches=('is_true_match', 'sum')
    ).reset_index()

    print(f"\n  {'Score Range':<15} {'Total Pairs':>12} {'True Matches':>13}")
    print(f"  {'─'*42}")
    for _, row in dist.iterrows():
        print(f"  {str(row['score_range']):<15} {int(row['count']):>12,} {int(row['true_matches']):>13,}")

    # ── Decision Buckets ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  DECISION BUCKETS")
    print(f"{'='*55}")

    auto_merge  = scored[scored['similarity_score'] >= 0.95]
    llm_review  = scored[(scored['similarity_score'] >= 0.75) &
                         (scored['similarity_score'] <  0.95)]
    keep_sep    = scored[scored['similarity_score'] <  0.75]

    print(f"\n  ✅ Auto Merge   (>= 0.95):   {len(auto_merge):>6,} pairs  "
          f"| True matches: {auto_merge['is_true_match'].sum():,}")
    print(f"  🤔 LLM Review  (0.75-0.95): {len(llm_review):>6,} pairs  "
          f"| True matches: {llm_review['is_true_match'].sum():,}")
    print(f"  ❌ Keep Sep    (< 0.75):    {len(keep_sep):>6,} pairs  "
          f"| True matches: {keep_sep['is_true_match'].sum():,}")

    # ── Sample High Scoring Pairs ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  SAMPLE — TOP 5 HIGHEST SCORING PAIRS")
    print(f"{'='*55}")

    # Merge agent names for readability
    name_map = df.set_index('record_id')['agent_full_name'].to_dict()
    scored['name_1'] = scored['record_id_1'].map(name_map)
    scored['name_2'] = scored['record_id_2'].map(name_map)

    top5 = scored.nlargest(5, 'similarity_score')[
        ['name_1', 'name_2', 'similarity_score', 'is_true_match']
    ]

    for _, row in top5.iterrows():
        match = "✅ Match" if row['is_true_match'] else "❌ No Match"
        print(f"\n  {row['name_1']:<25} vs {row['name_2']:<25}")
        print(f"  Score: {row['similarity_score']:.4f}  {match}")

    # ── Save scored pairs ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(SCORED_PAIRS_PATH), exist_ok=True)
    pairs_df.to_csv(SCORED_PAIRS_PATH, index=False)
    print(f"\n💾 Saved scored pairs to: {SCORED_PAIRS_PATH}")
    print(f"\n✅ Embeddings & Scoring Complete!\n")


if __name__ == "__main__":
    main()