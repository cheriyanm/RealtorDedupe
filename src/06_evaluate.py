"""
RealtorDedupe - Step 6: Evaluate & Build Unique Agent List
===========================================================
Evaluates pipeline performance and builds the final
unique agent list by merging duplicate records.

Process:
1. Load final match decisions
2. Calculate precision, recall, F1 score
3. Analyze errors (missed matches, false positives)
4. Build unique agent list from AUTO_MERGE decisions
5. Save evaluation report and unique agent list

Output:
- unique_agents.csv      → final deduplicated agent list
- evaluation_report.csv  → detailed performance metrics
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    PROCESSED_DATA_PATH,
    FINAL_MATCHES_PATH,
    OUTPUT_PATH,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
)

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# Calculate precision, recall and F1 score
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(df, threshold_label, true_col, pred_col):
    """
    Calculate precision, recall and F1 for a given decision bucket.

    Precision = of all pairs we said MATCH, how many were correct?
    Recall    = of all true matches, how many did we find?
    F1        = harmonic mean of precision and recall
    """
    tp = len(df[(df[pred_col] == 1) & (df[true_col] == 1)])  # True Positive
    fp = len(df[(df[pred_col] == 1) & (df[true_col] == 0)])  # False Positive
    fn = len(df[(df[pred_col] == 0) & (df[true_col] == 1)])  # False Negative
    tn = len(df[(df[pred_col] == 0) & (df[true_col] == 0)])  # True Negative

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * (precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0

    return {
        'threshold':  threshold_label,
        'tp':         tp,
        'fp':         fp,
        'fn':         fn,
        'tn':         tn,
        'precision':  round(precision, 4),
        'recall':     round(recall, 4),
        'f1':         round(f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ERROR ANALYSIS
# Find and explain missed matches and false positives
# ─────────────────────────────────────────────────────────────────────────────

def analyze_errors(final_df, agents_df):
    """
    Identify and analyze:
    - False Positives: pairs we said MATCH but aren't
    - False Negatives: true matches we missed
    """
    name_map  = agents_df.set_index('record_id')['agent_full_name'].to_dict()
    phone_map = agents_df.set_index('record_id')['norm_phone'].to_dict()
    email_map = agents_df.set_index('record_id')['norm_email'].to_dict()

    # False Positives — wrongly merged different agents
    fp = final_df[
        (final_df['decision'] == 'AUTO_MERGE') &
        (final_df['is_true_match'] == 0)
    ].copy()

    # False Negatives — missed true matches
    fn = final_df[
        (final_df['decision'] == 'KEEP_SEPARATE') &
        (final_df['is_true_match'] == 1)
    ].copy()

    return fp, fn, name_map, phone_map, email_map


# ─────────────────────────────────────────────────────────────────────────────
# BUILD UNIQUE AGENT LIST
# Merge duplicate records into single canonical agent records
# ─────────────────────────────────────────────────────────────────────────────

def build_unique_agents(agents_df, final_df):
    """
    Build the final unique agent list by:
    1. Starting with all records as separate agents
    2. Merging records that AUTO_MERGE decided are the same
    3. Keeping the most complete record for each unique agent
    """

    # Start with a copy of all records
    df = agents_df.copy()

    # Get all AUTO_MERGE pairs
    merges = final_df[final_df['decision'] == 'AUTO_MERGE'][
        ['record_id_1', 'record_id_2']
    ].values.tolist()

    # Build connected components (groups of duplicate records)
    # Using a simple union-find approach
    parent = {rid: rid for rid in df['record_id']}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union all merged pairs
    for id1, id2 in merges:
        if id1 in parent and id2 in parent:
            union(id1, id2)

    # Assign canonical ID to each record
    df['canonical_id'] = df['record_id'].apply(find)

    # For each group, keep the most complete record
    # (the one with fewest null values)
    df['null_count'] = df[['norm_name', 'norm_phone',
                            'norm_email', 'norm_license']].isnull().sum(axis=1)

    # Sort by null count so most complete record comes first
    df = df.sort_values('null_count')

    # Keep first (most complete) record per canonical group
    unique_agents = df.groupby('canonical_id').first().reset_index()

    # Add a clean unique agent ID
    unique_agents.insert(0, 'unique_agent_id',
        [f"AGT{str(i).zfill(5)}" for i in range(len(unique_agents))]
    )

    # Add count of how many records were merged
    merge_counts = df.groupby('canonical_id').size().reset_index()
    merge_counts.columns = ['canonical_id', 'records_merged']
    unique_agents = unique_agents.merge(merge_counts, on='canonical_id', how='left')

    return unique_agents


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("  STEP 6 — EVALUATE & BUILD UNIQUE AGENT LIST")
    print("="*55)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n📂 Loading data...")
    agents_df = pd.read_csv(PROCESSED_DATA_PATH)
    final_df  = pd.read_csv(FINAL_MATCHES_PATH)
    print(f"   Agent records:  {len(agents_df):,}")
    print(f"   Match pairs:    {len(final_df):,}")

    # ── Calculate metrics ─────────────────────────────────────────────────────
    print(f"\n📊 Calculating evaluation metrics...")

    # Create binary prediction columns for each threshold
    final_df['pred_auto']  = (final_df['decision'] == 'AUTO_MERGE').astype(int)
    final_df['pred_merge'] = (final_df['decision'] != 'KEEP_SEPARATE').astype(int)

    # Metrics for AUTO_MERGE only
    metrics_auto = calculate_metrics(
        final_df, 'AUTO_MERGE only',
        'is_true_match', 'pred_auto'
    )

    # Metrics for AUTO_MERGE + LLM_REVIEW combined
    metrics_combined = calculate_metrics(
        final_df, 'AUTO + LLM Review',
        'is_true_match', 'pred_merge'
    )

    # ── Print Evaluation Report ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  EVALUATION REPORT")
    print(f"{'='*55}")

    for m in [metrics_auto, metrics_combined]:
        print(f"\n  Strategy: {m['threshold']}")
        print(f"  {'─'*45}")
        print(f"  True Positives  (correct matches):     {m['tp']:>6,}")
        print(f"  False Positives (wrong matches):       {m['fp']:>6,}")
        print(f"  False Negatives (missed matches):      {m['fn']:>6,}")
        print(f"  True Negatives  (correct rejections):  {m['tn']:>6,}")
        print(f"  {'─'*45}")
        print(f"  Precision:  {m['precision']:.4f}  "
              f"(of matches found, how many correct?)")
        print(f"  Recall:     {m['recall']:.4f}  "
              f"(of true matches, how many found?)")
        print(f"  F1 Score:   {m['f1']:.4f}  "
              f"(balance of precision & recall)")

    # ── Error Analysis ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ERROR ANALYSIS")
    print(f"{'='*55}")

    fp, fn, name_map, phone_map, email_map = analyze_errors(final_df, agents_df)

    # False Positives
    print(f"\n  ❌ False Positives (wrongly merged): {len(fp):,}")
    if len(fp) > 0:
        print(f"  These DIFFERENT agents got merged:")
        for _, row in fp.head(3).iterrows():
            n1 = name_map.get(row['record_id_1'], 'Unknown')
            n2 = name_map.get(row['record_id_2'], 'Unknown')
            print(f"    {n1:<25} vs {n2:<25} score: {row['final_score']:.4f}")

    # False Negatives
    print(f"\n  ⚠️  False Negatives (missed matches): {len(fn):,}")
    if len(fn) > 0:
        print(f"  These SAME agents were not found:")
        for _, row in fn.head(3).iterrows():
            n1 = name_map.get(row['record_id_1'], 'Unknown')
            n2 = name_map.get(row['record_id_2'], 'Unknown')
            print(f"    {n1:<25} vs {n2:<25} score: {row['final_score']:.4f}")

    # ── Build Unique Agent List ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  BUILDING UNIQUE AGENT LIST")
    print(f"{'='*55}")

    unique_agents = build_unique_agents(agents_df, final_df)

    print(f"\n  Input records:          {len(agents_df):,}")
    print(f"  Unique agents found:    {len(unique_agents):,}")
    print(f"  Duplicates removed:     {len(agents_df) - len(unique_agents):,}")
    print(f"  Dedup rate:             "
          f"{(len(agents_df)-len(unique_agents))/len(agents_df)*100:.1f}%")

    # Records merged distribution
    merge_dist = unique_agents['records_merged'].value_counts().sort_index()
    print(f"\n  Records merged per unique agent:")
    for count, freq in merge_dist.items():
        bar = '█' * min(freq, 40)
        print(f"    {count} record(s):  {freq:>4,} agents  {bar}")

    # ── Sample Unique Agents ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  SAMPLE — UNIQUE AGENT LIST (top 5)")
    print(f"{'='*55}")

    sample_cols = ['unique_agent_id', 'agent_full_name',
                   'norm_phone', 'norm_email',
                   'norm_license', 'state', 'records_merged']

    for _, row in unique_agents[sample_cols].head(5).iterrows():
        print(f"\n  Agent ID:  {row['unique_agent_id']}")
        print(f"  Name:      {row['agent_full_name']}")
        print(f"  Phone:     {row['norm_phone']}")
        print(f"  Email:     {row['norm_email']}")
        print(f"  License:   {row['norm_license']}")
        print(f"  State:     {row['state']}")
        print(f"  Merged:    {row['records_merged']} record(s)")
        print(f"  {'─'*45}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save unique agents
    unique_agents.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved unique agents to: {OUTPUT_PATH}")

    # Save evaluation report
    report_path = OUTPUT_PATH.replace('unique_agents.csv', 'evaluation_report.csv')
    pd.DataFrame([metrics_auto, metrics_combined]).to_csv(report_path, index=False)
    print(f"💾 Saved evaluation report to: {report_path}")

    print(f"\n{'='*55}")
    print(f"  PIPELINE COMPLETE! 🎉")
    print(f"{'='*55}")
    print(f"\n  Input:   {len(agents_df):,} raw agent records")
    print(f"  Output:  {len(unique_agents):,} unique agents")
    print(f"  Recall:  {metrics_combined['recall']:.1%}")
    print(f"  F1:      {metrics_combined['f1']:.1%}")
    print(f"\n  ✅ RealtorDedupe prototype complete!\n")


if __name__ == "__main__":
    main()