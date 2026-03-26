"""
RealtorDedupe - Step 5: Splink Matching
=========================================
Uses Splink to calculate match probabilities
for each candidate pair by weighing multiple
field similarities against coincidence probability.

Process:
1. Load normalized agent records
2. Configure Splink with field comparisons
3. Train Splink model on the data
4. Predict match probabilities for all pairs
5. Combine with embedding scores
6. Make final match decisions

Output:
- final_matches.csv → all pairs with final decisions
"""

import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    PROCESSED_DATA_PATH,
    SCORED_PAIRS_PATH,
    FINAL_MATCHES_PATH,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
)

# ── Splink imports ────────────────────────────────────────────────────────────
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE SPLINK
# Tell Splink which fields to compare and how
# ─────────────────────────────────────────────────────────────────────────────

def configure_splink(df):
    """
    Configure Splink settings:
    - Which fields to compare
    - How to compare each field
    - Blocking rules for Splink's internal use
    """

    settings = SettingsCreator(
        link_type="dedupe_only",  # we're deduping one dataset

        comparisons=[
            # Name comparison — most important field
            # Jaro-Winkler is great for name variations
            cl.JaroWinklerAtThresholds(
                "norm_name",
                [0.95, 0.88, 0.70],  # high, medium, low thresholds
            ),

            # Phone comparison — exact match after normalization
            cl.ExactMatch("norm_phone").configure(
                term_frequency_adjustments=True
            ),

            # Email comparison — exact match
            cl.ExactMatch("norm_email").configure(
                term_frequency_adjustments=True
            ),

            # License comparison — exact match after normalization
            cl.ExactMatch("norm_license").configure(
                term_frequency_adjustments=True
            ),

            # Email username — fuzzy match
            cl.JaroWinklerAtThresholds(
                "email_username",
                [0.95, 0.88],
            ),
        ],

        # Blocking rules for Splink's internal comparison
        blocking_rules_to_generate_predictions=[
            block_on("norm_phone"),
            block_on("norm_license"),
            block_on("email_username"),
            "l.first_name = r.first_name AND l.last_name = r.last_name",
        ],

        # How many EM training iterations
        max_iterations=10,
        em_convergence=0.0001,
    )

    return settings


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN AND PREDICT
# Train Splink model and generate match probabilities
# ─────────────────────────────────────────────────────────────────────────────

def run_splink(df):
    """
    Train Splink and generate match probabilities.
    Returns a dataframe with match probabilities.
    """

    # Splink needs a unique id column called 'unique_id'
    df = df.copy()
    df['unique_id'] = df['record_id']

    # Fill nulls with empty string for Splink
    for col in ['norm_name', 'norm_phone', 'norm_email',
                'norm_license', 'email_username',
                'first_name', 'last_name']:
        df[col] = df[col].fillna('')

    # Configure Splink
    settings = configure_splink(df)

    # Initialize Splink linker with DuckDB backend
    db_api = DuckDBAPI()
    linker  = Linker(df, settings, db_api=db_api)

    # ── Train the model ───────────────────────────────────────────────────────
    print("\n  Training Splink model...")

    # Step 1: Estimate probability that two random records match
    linker.training.estimate_probability_two_random_records_match(
        [
            block_on("norm_phone"),
            block_on("norm_license"),
        ],
        recall=0.7,
    )

    # Step 2: Train u probabilities (coincidence rates)
    # using records that definitely don't match
    linker.training.estimate_u_using_random_sampling(max_pairs=1e5)

    # Step 3: Train m probabilities (true match rates)
    # using EM algorithm on name blocking
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("norm_phone"),
        estimate_without_term_frequencies=True,
    )

    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("norm_license"),
        estimate_without_term_frequencies=True,
    )

    print("  Splink model trained! ✅")

    # ── Generate predictions ──────────────────────────────────────────────────
    print("\n  Generating match predictions...")
    predictions = linker.inference.predict(threshold_match_probability=0.5)
    results_df  = predictions.as_pandas_dataframe()

    print(f"  Predictions generated: {len(results_df):,} pairs")

    return results_df, linker


# ─────────────────────────────────────────────────────────────────────────────
# COMBINE SPLINK + EMBEDDING SCORES
# Use both scores together for final decision
# ─────────────────────────────────────────────────────────────────────────────

def combine_scores(splink_df, embedding_df):
    """
    Combine Splink match probability with embedding similarity score.
    Final score = weighted average of both.
    """

    # Standardize record id columns for merging
    splink_df = splink_df.rename(columns={
        'unique_id_l': 'record_id_1',
        'unique_id_r': 'record_id_2',
        'match_probability': 'splink_score',
    })

    # Keep only what we need from Splink
    splink_slim = splink_df[['record_id_1', 'record_id_2', 'splink_score']].copy()

    # Ensure consistent pair ordering for merge
    splink_slim['pair_key'] = splink_slim.apply(
        lambda r: f"{min(r['record_id_1'], r['record_id_2'])}_{max(r['record_id_1'], r['record_id_2'])}",
        axis=1
    )
    embedding_df['pair_key'] = embedding_df.apply(
        lambda r: f"{min(r['record_id_1'], r['record_id_2'])}_{max(r['record_id_1'], r['record_id_2'])}",
        axis=1
    )

    # Merge on pair key
    combined = embedding_df.merge(
        splink_slim[['pair_key', 'splink_score']],
        on='pair_key',
        how='left'
    )

    # Fill missing Splink scores with 0
    combined['splink_score'] = combined['splink_score'].fillna(0)

    # Combined score = weighted average
    # Embedding: 40%, Splink: 60% (Splink weighs evidence more carefully)
    combined['final_score'] = (
        combined['similarity_score'] * 0.4 +
        combined['splink_score']     * 0.6
    ).round(4)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# MAKE FINAL DECISIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_decisions(df):
    """
    Apply thresholds to make final match decisions.
    """
    def decide(score):
        if score >= HIGH_CONFIDENCE:
            return 'AUTO_MERGE'
        elif score >= LOW_CONFIDENCE:
            return 'LLM_REVIEW'
        else:
            return 'KEEP_SEPARATE'

    df['decision'] = df['final_score'].apply(decide)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("  STEP 5 — SPLINK MATCHING")
    print("="*55)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n📂 Loading normalized agents...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"   Records loaded: {len(df):,}")

    print(f"\n📂 Loading scored pairs...")
    scored_df = pd.read_csv(SCORED_PAIRS_PATH)
    print(f"   Scored pairs loaded: {len(scored_df):,}")

    # ── Run Splink ────────────────────────────────────────────────────────────
    print(f"\n🔗 Running Splink...")
    splink_results, linker = run_splink(df)

    # ── Combine scores ────────────────────────────────────────────────────────
    print(f"\n🔀 Combining Splink + Embedding scores...")
    combined_df = combine_scores(splink_results, scored_df)
    print(f"   Combined pairs: {len(combined_df):,}")

    # ── Make decisions ────────────────────────────────────────────────────────
    print(f"\n⚖️  Making final decisions...")
    final_df = make_decisions(combined_df)

    # ── Results Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL MATCH DECISIONS")
    print(f"{'='*55}")

    for decision in ['AUTO_MERGE', 'LLM_REVIEW', 'KEEP_SEPARATE']:
        bucket    = final_df[final_df['decision'] == decision]
        matches   = bucket['is_true_match'].sum()
        total     = len(bucket)
        pct       = (matches / total * 100) if total > 0 else 0
        icon      = '✅' if decision == 'AUTO_MERGE' else '🤔' if decision == 'LLM_REVIEW' else '❌'
        print(f"\n  {icon} {decision:<15} {total:>6,} pairs")
        print(f"     True matches:  {matches:>6,} ({pct:.1f}%)")

    # ── Overall Performance ───────────────────────────────────────────────────
    total_true    = final_df['is_true_match'].sum()
    auto_caught   = final_df[
        (final_df['decision'] == 'AUTO_MERGE') &
        (final_df['is_true_match'] == 1)
    ].shape[0]
    llm_caught    = final_df[
        (final_df['decision'] == 'LLM_REVIEW') &
        (final_df['is_true_match'] == 1)
    ].shape[0]
    missed        = final_df[
        (final_df['decision'] == 'KEEP_SEPARATE') &
        (final_df['is_true_match'] == 1)
    ].shape[0]

    print(f"\n{'='*55}")
    print(f"  OVERALL PERFORMANCE")
    print(f"{'='*55}")
    print(f"  Total true matches:        {total_true:,}")
    print(f"  Caught by Auto Merge:      {auto_caught:,} ({auto_caught/total_true*100:.1f}%)")
    print(f"  Caught by LLM Review:      {llm_caught:,} ({llm_caught/total_true*100:.1f}%)")
    print(f"  Missed (Keep Separate):    {missed:,}  ({missed/total_true*100:.1f}%)")
    print(f"  {'─'*45}")
    print(f"  Combined Recall:           {(auto_caught+llm_caught)/total_true*100:.1f}%")

    # ── Save final matches ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(FINAL_MATCHES_PATH), exist_ok=True)
    final_df.to_csv(FINAL_MATCHES_PATH, index=False)
    print(f"\n💾 Saved final matches to: {FINAL_MATCHES_PATH}")
    print(f"\n✅ Splink Matching Complete!\n")


if __name__ == "__main__":
    main()