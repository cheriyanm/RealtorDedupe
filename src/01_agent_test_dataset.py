"""
RealtorDedupe - Sample Dataset Generator
==========================================
Generates a realistic test dataset of real estate agents
with enough variations to cover all test cases.

Test Cases Covered:
1. Same agent, name variations (Tom/Thomas, Bob/Robert)
2. Same agent, multiple states/MLS
3. Same agent, phone number changed
4. Same agent, different email
5. Same agent, license prefix variations
6. Different agents, similar names (John Smith x multiple)
7. Exact duplicates
8. Missing/null fields
9. Truly different agents
"""

import pandas as pd
import random
import uuid

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

# ── Name Variations ───────────────────────────────────────────────────────────
NAME_VARIATIONS = {
    "Thomas":    ["Tom", "Tommy", "T.", "Thos"],
    "Robert":    ["Rob", "Bob", "Bobby", "R."],
    "Elizabeth": ["Liz", "Beth", "Eliza", "E."],
    "William":   ["Will", "Bill", "Billy", "W."],
    "Jennifer":  ["Jen", "Jenny", "J."],
    "Michael":   ["Mike", "Mick", "M."],
    "Patricia":  ["Pat", "Patty", "Tricia", "P."],
    "James":     ["Jim", "Jimmy", "J."],
    "Linda":     ["Lin", "Lindy", "L."],
    "Barbara":   ["Barb", "Babs", "B."],
}

LAST_NAMES = [
    "Johnson", "Smith", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Wilson", "Taylor",
    "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Thompson", "Young", "Allen", "King",
]

# ── MLS Systems ───────────────────────────────────────────────────────────────
MLS_SYSTEMS = {
    "NC": ["CRMLS_NC", "NWMLS_NC", "FLEX_NC"],
    "CA": ["CRMLS_CA", "NWMLS_CA", "MATRIX_CA"],
    "TX": ["NTREIS_TX", "SABOR_TX", "HAR_TX"],
    "FL": ["MFRMLS_FL", "MIAMI_FL", "BRIGHT_FL"],
    "NY": ["REBNY_NY", "MLSLI_NY", "NYSMLS_NY"],
}

STATES = list(MLS_SYSTEMS.keys())

# ── License Prefix Variations ─────────────────────────────────────────────────
def format_license(state, number):
    """Simulate how different MLS systems format the same license number."""
    formats = [
        f"{state}-{number}",        # NC-12345
        f"{state}{number}",         # NC12345
        f"{number}",                # 12345
        f"{state}-{number:06d}",    # NC-012345
        f"LIC-{state}-{number}",    # LIC-NC-12345
    ]
    return random.choice(formats)

# ── Phone Variations ──────────────────────────────────────────────────────────
def format_phone(area, exchange, number):
    """Simulate different phone formatting across MLS systems."""
    formats = [
        f"{area}-{exchange}-{number}",      # 704-555-1234
        f"({area}) {exchange}-{number}",    # (704) 555-1234
        f"{area}.{exchange}.{number}",      # 704.555.1234
        f"{area}{exchange}{number}",        # 7045551234
        f"+1{area}{exchange}{number}",      # +17045551234
    ]
    return random.choice(formats)

def generate_phone():
    """Generate a random phone number."""
    area     = random.randint(200, 999)
    exchange = random.randint(200, 999)
    number   = random.randint(1000, 9999)
    return area, exchange, number

# ── Email Variations ──────────────────────────────────────────────────────────
EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com",
    "realty.com", "kw.com", "cbunited.com",
    "remax.com", "coldwellbanker.com",
]

def generate_email(first, last, variation=0):
    """Generate email with variations for same agent."""
    first_lower = first.lower().replace(".", "")
    last_lower  = last.lower()
    domain      = random.choice(EMAIL_DOMAINS)
    formats     = [
        f"{first_lower}.{last_lower}@{domain}",
        f"{first_lower[0]}{last_lower}@{domain}",
        f"{first_lower}{last_lower[0]}@{domain}",
        f"{first_lower}_{last_lower}@{domain}",
        f"{last_lower}.{first_lower}@{domain}",
    ]
    return formats[variation % len(formats)]

# ── Core Agent Generator ──────────────────────────────────────────────────────
def generate_base_agent(first_name, last_name, state):
    """Generate a base agent record."""
    area, exchange, number = generate_phone()
    license_num            = random.randint(10000, 99999)
    mls_id                 = f"MLS{random.randint(100000, 999999)}"
    return {
        "true_agent_id":          str(uuid.uuid4())[:8],  # ground truth for evaluation
        "first_name":             first_name,
        "last_name":              last_name,
        "state":                  state,
        "area":                   area,
        "exchange":               exchange,
        "number":                 number,
        "license_num":            license_num,
        "mls_id":                 mls_id,
        "email_variation":        0,
    }

# ── Dataset Builder ───────────────────────────────────────────────────────────
records = []

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 1 — Same agent, name variations across MLS
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Test Case 1: Name variations...")
for first_name, variations in NAME_VARIATIONS.items():
    last_name = random.choice(LAST_NAMES)
    state     = random.choice(STATES)
    base      = generate_base_agent(first_name, last_name, state)
    mls_list  = MLS_SYSTEMS[state]

    # Original record
    records.append({
        "agent_full_name":      f"{first_name} {last_name}",
        "agent_mls_id":         base["mls_id"],
        "agent_state_license":  format_license(state, base["license_num"]),
        "agent_preferred_phone":format_phone(base["area"], base["exchange"], base["number"]),
        "agent_email":          generate_email(first_name, last_name, 0),
        "office_name":          f"{last_name} Realty",
        "state":                state,
        "mls_source":           mls_list[0],
        "true_agent_id":        base["true_agent_id"],
        "test_case":            "TC1_name_variation",
    })

    # Name variation records (2-3 per agent)
    for i, variation in enumerate(variations[:3], 1):
        mls = mls_list[i % len(mls_list)]
        records.append({
            "agent_full_name":      f"{variation} {last_name}",
            "agent_mls_id":         f"MLS{random.randint(100000, 999999)}",
            "agent_state_license":  format_license(state, base["license_num"]),
            "agent_preferred_phone":format_phone(base["area"], base["exchange"], base["number"]),
            "agent_email":          generate_email(variation, last_name, i),
            "office_name":          f"{last_name} Realty",
            "state":                state,
            "mls_source":           mls,
            "true_agent_id":        base["true_agent_id"],
            "test_case":            "TC1_name_variation",
        })

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 2 — Same agent, multiple states (multi-licensed)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Test Case 2: Multi-state agents...")
for i in range(20):
    first_name = random.choice(list(NAME_VARIATIONS.keys()))
    last_name  = random.choice(LAST_NAMES)
    base       = generate_base_agent(first_name, last_name, STATES[0])
    agent_id   = base["true_agent_id"]

    # Same agent in 2-3 different states
    for state in random.sample(STATES, random.randint(2, 3)):
        mls = random.choice(MLS_SYSTEMS[state])
        records.append({
            "agent_full_name":      f"{first_name} {last_name}",
            "agent_mls_id":         f"MLS{random.randint(100000, 999999)}",
            "agent_state_license":  format_license(state, base["license_num"]),
            "agent_preferred_phone":format_phone(base["area"], base["exchange"], base["number"]),
            "agent_email":          generate_email(first_name, last_name, 0),
            "office_name":          f"{last_name} Properties",
            "state":                state,
            "mls_source":           mls,
            "true_agent_id":        agent_id,
            "test_case":            "TC2_multi_state",
        })

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 3 — Same agent, phone number changed
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Test Case 3: Phone number changes...")
for i in range(20):
    first_name = random.choice(list(NAME_VARIATIONS.keys()))
    last_name  = random.choice(LAST_NAMES)
    state      = random.choice(STATES)
    base       = generate_base_agent(first_name, last_name, state)
    agent_id   = base["true_agent_id"]
    mls_list   = MLS_SYSTEMS[state]

    # Old phone number
    records.append({
        "agent_full_name":      f"{first_name} {last_name}",
        "agent_mls_id":         base["mls_id"],
        "agent_state_license":  format_license(state, base["license_num"]),
        "agent_preferred_phone":format_phone(base["area"], base["exchange"], base["number"]),
        "agent_email":          generate_email(first_name, last_name, 0),
        "office_name":          f"{last_name} Realty Group",
        "state":                state,
        "mls_source":           mls_list[0],
        "true_agent_id":        agent_id,
        "test_case":            "TC3_phone_changed",
    })

    # New phone number (different area/exchange/number)
    new_area, new_exchange, new_number = generate_phone()
    records.append({
        "agent_full_name":      f"{first_name} {last_name}",
        "agent_mls_id":         f"MLS{random.randint(100000, 999999)}",
        "agent_state_license":  format_license(state, base["license_num"]),
        "agent_preferred_phone":format_phone(new_area, new_exchange, new_number),
        "agent_email":          generate_email(first_name, last_name, 0),
        "office_name":          f"{last_name} Realty Group",
        "state":                state,
        "mls_source":           mls_list[1] if len(mls_list) > 1 else mls_list[0],
        "true_agent_id":        agent_id,
        "test_case":            "TC3_phone_changed",
    })

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 4 — Same agent, different email
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Test Case 4: Email variations...")
for i in range(20):
    first_name = random.choice(list(NAME_VARIATIONS.keys()))
    last_name  = random.choice(LAST_NAMES)
    state      = random.choice(STATES)
    base       = generate_base_agent(first_name, last_name, state)
    agent_id   = base["true_agent_id"]

    for email_var in range(3):
        records.append({
            "agent_full_name":      f"{first_name} {last_name}",
            "agent_mls_id":         f"MLS{random.randint(100000, 999999)}",
            "agent_state_license":  format_license(state, base["license_num"]),
            "agent_preferred_phone":format_phone(base["area"], base["exchange"], base["number"]),
            "agent_email":          generate_email(first_name, last_name, email_var),
            "office_name":          f"{last_name} & Associates",
            "state":                state,
            "mls_source":           random.choice(MLS_SYSTEMS[state]),
            "true_agent_id":        agent_id,
            "test_case":            "TC4_email_variation",
        })

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 5 — License prefix variations (same number, different format)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Test Case 5: License prefix variations...")
for i in range(20):
    first_name  = random.choice(list(NAME_VARIATIONS.keys()))
    last_name   = random.choice(LAST_NAMES)
    state       = random.choice(STATES)
    base        = generate_base_agent(first_name, last_name, state)
    agent_id    = base["true_agent_id"]
    license_num = base["license_num"]

    # Same license, different formats
    license_formats = [
        f"{state}-{license_num}",
        f"{state}{license_num}",
        f"{license_num}",
        f"LIC-{state}-{license_num}",
    ]

    for fmt in license_formats:
        records.append({
            "agent_full_name":      f"{first_name} {last_name}",
            "agent_mls_id":         f"MLS{random.randint(100000, 999999)}",
            "agent_state_license":  fmt,
            "agent_preferred_phone":format_phone(base["area"], base["exchange"], base["number"]),
            "agent_email":          generate_email(first_name, last_name, 0),
            "office_name":          f"{last_name} Real Estate",
            "state":                state,
            "mls_source":           random.choice(MLS_SYSTEMS[state]),
            "true_agent_id":        agent_id,
            "test_case":            "TC5_license_prefix",
        })

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 6 — Different agents, similar names (hard negatives!)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Test Case 6: Similar names, different agents...")
common_first = ["John", "Mary", "David", "Sarah", "Chris"]
common_last  = ["Smith", "Johnson", "Williams", "Brown", "Jones"]

for first in common_first:
    for last in common_last:
        # 2-3 genuinely different agents with same name
        for _ in range(random.randint(2, 3)):
            state   = random.choice(STATES)
            base    = generate_base_agent(first, last, state)
            records.append({
                "agent_full_name":      f"{first} {last}",
                "agent_mls_id":         base["mls_id"],
                "agent_state_license":  format_license(state, base["license_num"]),
                "agent_preferred_phone":format_phone(base["area"], base["exchange"], base["number"]),
                "agent_email":          generate_email(first, last, random.randint(0, 4)),
                "office_name":          random.choice(["Keller Williams", "RE/MAX", "Century 21", "Coldwell Banker"]),
                "state":                state,
                "mls_source":           random.choice(MLS_SYSTEMS[state]),
                "true_agent_id":        base["true_agent_id"],  # Each gets unique ID!
                "test_case":            "TC6_similar_name_different_agent",
            })

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 7 — Exact duplicates (should be easiest to catch)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Test Case 7: Exact duplicates...")
for i in range(20):
    first_name = random.choice(list(NAME_VARIATIONS.keys()))
    last_name  = random.choice(LAST_NAMES)
    state      = random.choice(STATES)
    base       = generate_base_agent(first_name, last_name, state)
    agent_id   = base["true_agent_id"]
    phone      = format_phone(base["area"], base["exchange"], base["number"])
    email      = generate_email(first_name, last_name, 0)
    license    = format_license(state, base["license_num"])

    # Exact same record submitted twice
    for _ in range(2):
        records.append({
            "agent_full_name":      f"{first_name} {last_name}",
            "agent_mls_id":         base["mls_id"],
            "agent_state_license":  license,
            "agent_preferred_phone":phone,
            "agent_email":          email,
            "office_name":          "Century 21",
            "state":                state,
            "mls_source":           MLS_SYSTEMS[state][0],
            "true_agent_id":        agent_id,
            "test_case":            "TC7_exact_duplicate",
        })

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 8 — Missing / null fields
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Test Case 8: Missing fields...")
for i in range(30):
    first_name = random.choice(list(NAME_VARIATIONS.keys()))
    last_name  = random.choice(LAST_NAMES)
    state      = random.choice(STATES)
    base       = generate_base_agent(first_name, last_name, state)
    agent_id   = base["true_agent_id"]

    # Randomly null out some fields
    phone   = format_phone(base["area"], base["exchange"], base["number"]) if random.random() > 0.3 else None
    email   = generate_email(first_name, last_name, 0)                     if random.random() > 0.3 else None
    license = format_license(state, base["license_num"])                   if random.random() > 0.3 else None
    mls_id  = base["mls_id"]                                               if random.random() > 0.3 else None

    records.append({
        "agent_full_name":      f"{first_name} {last_name}",
        "agent_mls_id":         mls_id,
        "agent_state_license":  license,
        "agent_preferred_phone":phone,
        "agent_email":          email,
        "office_name":          f"{last_name} Realty",
        "state":                state,
        "mls_source":           random.choice(MLS_SYSTEMS[state]),
        "true_agent_id":        agent_id,
        "test_case":            "TC8_missing_fields",
    })

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 9 — Truly different agents (true negatives)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Test Case 9: Truly different agents...")
for i in range(100):
    first_name = random.choice(list(NAME_VARIATIONS.keys()))
    last_name  = random.choice(LAST_NAMES)
    state      = random.choice(STATES)
    base       = generate_base_agent(first_name, last_name, state)

    records.append({
        "agent_full_name":      f"{first_name} {last_name}",
        "agent_mls_id":         base["mls_id"],
        "agent_state_license":  format_license(state, base["license_num"]),
        "agent_preferred_phone":format_phone(base["area"], base["exchange"], base["number"]),
        "agent_email":          generate_email(first_name, last_name, random.randint(0, 4)),
        "office_name":          random.choice(["Keller Williams", "RE/MAX", "Century 21", "Coldwell Banker", "eXp Realty"]),
        "state":                state,
        "mls_source":           random.choice(MLS_SYSTEMS[state]),
        "true_agent_id":        base["true_agent_id"],
        "test_case":            "TC9_different_agents",
    })

# ─────────────────────────────────────────────────────────────────────────────
# Assemble & Save
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)

# Add a unique record ID for reference
df.insert(0, "record_id", [f"REC{str(i).zfill(5)}" for i in range(len(df))])

# Shuffle the dataset (real data won't be neatly ordered!)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv("data/raw/agent_test_dataset.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Summary Report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  DATASET GENERATION COMPLETE!")
print("="*55)
print(f"\n  Total Records Generated: {len(df):,}")
print(f"\n  Records by Test Case:")
print("-"*55)
summary = df.groupby("test_case")["record_id"].count().reset_index()
summary.columns = ["Test Case", "Record Count"]
for _, row in summary.iterrows():
    print(f"  {row['Test Case']:<40} {row['Record Count']:>5}")
print("-"*55)
print(f"\n  Unique True Agents:  {df['true_agent_id'].nunique():,}")
print(f"  Total Records:       {len(df):,}")
print(f"  Avg Records/Agent:   {len(df)/df['true_agent_id'].nunique():.1f}")
print(f"\n  Saved to: data/raw/agent_test_dataset.csv")
print("="*55)