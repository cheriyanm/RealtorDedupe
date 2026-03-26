# config/settings.py

# File Paths
RAW_DATA_PATH        = "data/raw/agent_test_dataset.csv"
PROCESSED_DATA_PATH  = "data/processed/normalized_agents.csv"
CANDIDATE_PAIRS_PATH = "data/processed/candidate_pairs.csv"
SCORED_PAIRS_PATH    = "data/processed/scored_pairs.csv"
OUTPUT_PATH          = "data/output/unique_agents.csv"

# Matching Thresholds
HIGH_CONFIDENCE      = 0.95
LOW_CONFIDENCE       = 0.75

# Embedding Model
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"