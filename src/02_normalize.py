"""
RealtorDedupe - Step 2: Normalization
=======================================
Cleans and standardizes all agent fields
so downstream steps can compare apples to apples.

Normalizations Applied:
1. Phone numbers    → digits only, 10 digits
2. License numbers  → digits only, strip state prefix
3. Agent names      → lowercase, strip extra spaces
4. Email addresses  → lowercase, strip whitespace
5. Office names     → lowercase, strip extra spaces
"""

import pandas as pd
import re
import os
import sys

# ── Add project root to path so we can import config ─────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RAW_DATA_PATH, PROCESSED_DATA_PATH

# ─────────────────────────────────────────────────────────────────────────────
# 1. PHONE NORMALIZATION
# Goal: strip everything, keep digits only, expect 10 or 11 digits
#
# Examples:
#   (704) 555-1234  → 7045551234
#   704.555.1234    → 7045551234
#   +17045551234    → 7045551234
#   704-555-1234    → 7045551234
# ─────────────────────────────────────────────────────────────────────────────
def normalize_phone(phone):
    if pd.isna(phone) or phone is None:
        return None

    # Remove everything except digits
    digits = re.sub(r'\D', '', str(phone))

    # Remove leading country code "1" if 11 digits
    if len(digits) == 11 and digits.startswith('1'):
        digits = digits[1:]

    # Must be exactly 10 digits to be valid
    if len(digits) == 10:
        return digits

    # Return None if invalid
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. LICENSE NORMALIZATION
# Goal: strip state prefix and non-digits, keep core number only
#
# Examples:
#   NC-12345        → 12345
#   NC12345         → 12345
#   LIC-NC-12345    → 12345
#   12345           → 12345
# ─────────────────────────────────────────────────────────────────────────────
def normalize_license(license_str):
    if pd.isna(license_str) or license_str is None:
        return None

    # Remove all non-digit characters
    digits = re.sub(r'\D', '', str(license_str))

    if len(digits) > 0:
        return digits

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. NAME NORMALIZATION
# Goal: lowercase, strip extra spaces, standardize format
#
# Examples:
#   "  Thomas  Johnson  "  → "thomas johnson"
#   "THOMAS JOHNSON"       → "thomas johnson"
#   "Thomas  Johnson"      → "thomas johnson"
# ─────────────────────────────────────────────────────────────────────────────
def normalize_name(name):
    if pd.isna(name) or name is None:
        return None

    # Lowercase and strip leading/trailing spaces
    name = str(name).lower().strip()

    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)

    # Remove special characters except spaces and hyphens
    name = re.sub(r'[^a-z\s\-]', '', name)

    return name if len(name) > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# 4. EMAIL NORMALIZATION
# Goal: lowercase, strip whitespace
#
# Examples:
#   "Thomas.Johnson@Gmail.COM  " → "thomas.johnson@gmail.com"
#   "  TJOHNSON@REALTY.COM"      → "tjohnson@realty.com"
# ─────────────────────────────────────────────────────────────────────────────
def normalize_email(email):
    if pd.isna(email) or email is None:
        return None

    email = str(email).lower().strip()

    # Basic email validation - must contain @ and .
    if '@' in email and '.' in email:
        return email

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. OFFICE NAME NORMALIZATION
# Goal: lowercase, strip extra spaces, remove common suffixes
#
# Examples:
#   "Keller Williams Realty, Inc." → "keller williams realty"
#   "RE/MAX LLC"                   → "remax"
#   "Century 21"                   → "century 21"
# ─────────────────────────────────────────────────────────────────────────────
def normalize_office(office):
    if pd.isna(office) or office is None:
        return None

    office = str(office).lower().strip()

    # Remove common suffixes
    suffixes = [', inc.', ', llc', ', ltd', ' inc.', ' llc', ' ltd', ' realty', ' real estate']
    for suffix in suffixes:
        office = office.replace(suffix, '')

    # Remove special characters except spaces
    office = re.sub(r'[^a-z0-9\s]', '', office)

    # Remove multiple spaces
    office = re.sub(r'\s+', ' ', office).strip()

    return office if len(office) > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXTRACT EMAIL FEATURES
# Goal: extract username and domain for blocking later
#
# Example:
#   "tjohnson@gmail.com" → username: "tjohnson", domain: "gmail.com"
# ─────────────────────────────────────────────────────────────────────────────
def extract_email_username(email):
    if pd.isna(email) or email is None:
        return None
    try:
        return str(email).split('@')[0]
    except:
        return None

def extract_email_domain(email):
    if pd.isna(email) or email is None:
        return None
    try:
        return str(email).split('@')[1]
    except:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXTRACT NAME FEATURES
# Goal: extract first and last name for blocking later
# ─────────────────────────────────────────────────────────────────────────────
def extract_first_name(name):
    if pd.isna(name) or name is None:
        return None
    parts = str(name).strip().split()
    return parts[0] if len(parts) >= 1 else None

def extract_last_name(name):
    if pd.isna(name) or name is None:
        return None
    parts = str(name).strip().split()
    return parts[-1] if len(parts) >= 2 else None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run all normalizations
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("  STEP 2 — NORMALIZATION")
    print("="*55)

    # ── Load raw data ─────────────────────────────────────────────────────────
    print(f"\n📂 Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   Records loaded: {len(df):,}")

    # ── Apply normalizations ──────────────────────────────────────────────────
    print("\n🔧 Applying normalizations...")

    # Normalize each field
    df['norm_phone']   = df['agent_preferred_phone'].apply(normalize_phone)
    df['norm_license'] = df['agent_state_license'].apply(normalize_license)
    df['norm_name']    = df['agent_full_name'].apply(normalize_name)
    df['norm_email']   = df['agent_email'].apply(normalize_email)
    df['norm_office']  = df['office_name'].apply(normalize_office)

    # Extract features for blocking
    df['first_name']      = df['norm_name'].apply(extract_first_name)
    df['last_name']        = df['norm_name'].apply(extract_last_name)
    df['email_username']  = df['norm_email'].apply(extract_email_username)
    df['email_domain']    = df['norm_email'].apply(extract_email_domain)

    # ── Save normalized data ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\n💾 Saved normalized data to: {PROCESSED_DATA_PATH}")

    # ── Summary Report ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  NORMALIZATION SUMMARY")
    print("="*55)
    print(f"\n  Total Records:          {len(df):,}")
    print(f"\n  Field Coverage (non-null after normalization):")
    print(f"  norm_name:              {df['norm_name'].notna().sum():,} ({df['norm_name'].notna().mean()*100:.1f}%)")
    print(f"  norm_phone:             {df['norm_phone'].notna().sum():,} ({df['norm_phone'].notna().mean()*100:.1f}%)")
    print(f"  norm_email:             {df['norm_email'].notna().sum():,} ({df['norm_email'].notna().mean()*100:.1f}%)")
    print(f"  norm_license:           {df['norm_license'].notna().sum():,} ({df['norm_license'].notna().mean()*100:.1f}%)")
    print(f"  norm_office:            {df['norm_office'].notna().sum():,} ({df['norm_office'].notna().mean()*100:.1f}%)")

    # ── Sample Output ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SAMPLE — Before vs After Normalization")
    print("="*55)
    sample = df[['agent_full_name', 'norm_name',
                 'agent_preferred_phone', 'norm_phone',
                 'agent_state_license', 'norm_license',
                 'agent_email', 'norm_email']].head(5)

    for _, row in sample.iterrows():
        print(f"\n  Name:    {row['agent_full_name']:<30} → {row['norm_name']}")
        print(f"  Phone:   {str(row['agent_preferred_phone']):<30} → {row['norm_phone']}")
        print(f"  License: {str(row['agent_state_license']):<30} → {row['norm_license']}")
        print(f"  Email:   {str(row['agent_email']):<30} → {row['norm_email']}")
        print(f"  {'-'*50}")

    print("\n✅ Normalization Complete!\n")


if __name__ == "__main__":
    main()