"""
RealtorDedupe - Step 3: Blocking
==================================
Groups agent records into blocks using multiple
blocking rules, then generates candidate pairs
within each block for comparison.

Blocking Rules (OR logic):
1. Last name prefix (first 3 chars) + First name initial
2. Last 7 digits of phone number
3. Email username (part before @)
4. Normalized license number

Output:
- candidate_pairs.csv → all unique pairs to compare
"""

import pandas as pd
import itertools
import os
import sys

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PROCESSED_DATA_PATH, CANDIDATE_PAIRS_PATH


# ─────────────────────────────────────────────────────────────────────────────
# BLOCKING KEY GENERATORS
# Each function returns a blocking key for a record
# Records with the same key go into the same block
# ─────────────────────────────────────────────────────────────────────────────

def blocking_key_name(row):
    """
    Rule 1: First 3 chars of last name + First char of first name
    Example: thomas johnson → joh_t
    Catches: Thomas Johnson, Tom Johnson, T. Johnson
    """
    try:
        last  = str(row['last_name']).strip()[:3]   if pd.notna(row['last_name'])  else None
        first = str(row['first_name']).strip()[:1]  if pd.notna(row['first_name']) else None
        if last and first:
            return f"{last}_{first}"
    except:
        pass
    return None


def blocking_key_phone(row):
    """
    Rule 2: Last 7 digits of normalized phone
    Example: 7045551234 → 5551234
    Catches: same phone despite area code formatting differences
    """
    try:
        phone = str(row['norm_phone']).strip() if pd.notna(row['norm_phone']) else None
        if phone and len(phone) >= 7:
            return phone[-7:]
    except:
        pass
    return None


def blocking_key_email_username(row):
    """
    Rule 3: Email username (part before @)
    Example: tjohnson@gmail.com → tjohnson
    Catches: same person with different email domains
    """
    try:
        username = row['email_username']
        if pd.notna(username) and len(str(username)) > 2:
            return str(username).strip()
    except:
        pass
    return None


def blocking_key_license(row):
    """
    Rule 4: Normalized license number
    Example: NC-12345, NC12345, 12345 → all become 12345
    Catches: same license with different state prefixes
    """
    try:
        license_num = row['norm_license']
        if pd.notna(license_num) and len(str(license_num)) > 3:
            return str(license_num).strip()
    except:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK GENERATOR
# Applies all blocking rules and groups records into blocks
# ─────────────────────────────────────────────────────────────────────────────

def generate_blocks(df):
    """
    Apply all blocking rules and return a dict of blocks.
    Each block = list of record_ids that share a blocking key.
    """
    blocks = {}

    blocking_rules = [
        ("name_prefix",     blocking_key_name),
        ("phone_last7",     blocking_key_phone),
        ("email_username",  blocking_key_email_username),
        ("license_num",     blocking_key_license),
    ]

    for rule_name, rule_func in blocking_rules:
        print(f"  Applying rule: {rule_name}...")
        rule_blocks = 0

        for _, row in df.iterrows():
            key = rule_func(row)
            if key:
                # Prefix key with rule name to keep blocks separate
                block_key = f"{rule_name}:{key}"
                if block_key not in blocks:
                    blocks[block_key] = []
                blocks[block_key].append(row['record_id'])
                rule_blocks += 1

        print(f"    → {rule_blocks:,} records assigned to blocks")

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# PAIR GENERATOR
# For each block, generate all possible pairs of records
# Then deduplicate pairs across all blocks
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidate_pairs(blocks):
    """
    For each block, generate all possible pairs.
    Deduplicate pairs across blocks.
    Returns a set of (record_id_1, record_id_2) tuples.
    """
    candidate_pairs = set()
    skipped_blocks  = 0

    for block_key, record_ids in blocks.items():
        # Skip blocks with only 1 record (nothing to compare)
        if len(record_ids) < 2:
            continue

        # Skip very large blocks (likely bad blocking key)
        # Avoids exploding number of pairs
        if len(record_ids) > 500:
            skipped_blocks += 1
            continue

        # Generate all pairs within this block
        for id1, id2 in itertools.combinations(record_ids, 2):
            # Always store pair in consistent order (smaller id first)
            pair = (min(id1, id2), max(id1, id2))
            candidate_pairs.add(pair)

    if skipped_blocks > 0:
        print(f"  ⚠️  Skipped {skipped_blocks} oversized blocks")

    return candidate_pairs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("  STEP 3 — BLOCKING")
    print("="*55)

    # ── Load normalized data ──────────────────────────────────────────────────
    print(f"\n📂 Loading normalized data from: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"   Records loaded: {len(df):,}")

    # ── Generate blocks ───────────────────────────────────────────────────────
    print(f"\n🔲 Generating blocks...")
    blocks = generate_blocks(df)
    print(f"\n   Total blocks generated: {len(blocks):,}")

    # Block size distribution
    block_sizes = [len(v) for v in blocks.values() if len(v) >= 2]
    if block_sizes:
        print(f"   Avg block size:         {sum(block_sizes)/len(block_sizes):.1f}")
        print(f"   Max block size:         {max(block_sizes):,}")
        print(f"   Min block size:         {min(block_sizes):,}")

    # ── Generate candidate pairs ──────────────────────────────────────────────
    print(f"\n🔗 Generating candidate pairs...")
    candidate_pairs = generate_candidate_pairs(blocks)
    print(f"   Total candidate pairs:  {len(candidate_pairs):,}")

    # ── Build pairs dataframe ─────────────────────────────────────────────────
    pairs_df = pd.DataFrame(
        list(candidate_pairs),
        columns=['record_id_1', 'record_id_2']
    )

    # ── Add ground truth for evaluation ──────────────────────────────────────
    # Merge true_agent_id for both records so we can measure accuracy later
    id_map = df.set_index('record_id')['true_agent_id'].to_dict()
    pairs_df['true_agent_id_1'] = pairs_df['record_id_1'].map(id_map)
    pairs_df['true_agent_id_2'] = pairs_df['record_id_2'].map(id_map)
    pairs_df['is_true_match']   = (
        pairs_df['true_agent_id_1'] == pairs_df['true_agent_id_2']
    ).astype(int)

    # ── Blocking Quality Metrics ──────────────────────────────────────────────
    total_possible = len(df) * (len(df) - 1) / 2
    reduction      = (1 - len(candidate_pairs) / total_possible) * 100
    true_matches   = pairs_df['is_true_match'].sum()

    print(f"\n{'='*55}")
    print(f"  BLOCKING QUALITY REPORT")
    print(f"{'='*55}")
    print(f"  Total records:            {len(df):,}")
    print(f"  Total possible pairs:     {int(total_possible):,}")
    print(f"  Candidate pairs:          {len(candidate_pairs):,}")
    print(f"  Pairs reduction:          {reduction:.1f}%")
    print(f"  True matches in pairs:    {true_matches:,}")
    print(f"  {'─'*45}")
    print(f"  🎯 Recall check:")
    print(f"  True matches captured:    {true_matches:,}")
    print(f"  (Higher = better blocking)")

    # ── Save candidate pairs ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(CANDIDATE_PAIRS_PATH), exist_ok=True)
    pairs_df.to_csv(CANDIDATE_PAIRS_PATH, index=False)
    print(f"\n💾 Saved candidate pairs to: {CANDIDATE_PAIRS_PATH}")
    print(f"\n✅ Blocking Complete!\n")


if __name__ == "__main__":
    main()