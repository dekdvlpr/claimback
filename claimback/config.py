
"""Central configuration for Claimback."""

# --------------------------------------------------
# 1. Machine learning backend
# --------------------------------------------------

# Available options:
# "stub" = keyword-based demonstration predictions
# "tfidf" = baseline TF-IDF model
# "hf" = fine-tuned Hugging Face models

MODEL_BACKEND = "hf"

# Hugging Face model identifiers
HF_WIN_MODEL = "InuraDekumpitiya/claimback-win-v1"
HF_ISSUE_MODEL = "InuraDekumpitiya/claimback-issue-v1"

# Optional local model files
TFIDF_PATH = "models/tfidf.joblib"
CALIBRATOR_PATH = "models/calibrator.joblib"

# Optional historical outcome data
BASE_RATES_PATH = "data/base_rates.csv"


# --------------------------------------------------
# 2. Financial complaint categories
# --------------------------------------------------

ISSUE_GROUPS = [
    "Account problems",
    "Card or transaction dispute",
    "Debt collection",
    "Fees and charges",
    "Loan or mortgage servicing",
    "Scam or unauthorised payment",
]


# --------------------------------------------------
# 3. AI complaint rewriting
# --------------------------------------------------

# Gemini API configuration is managed through .env
# GEMINI_API_KEY and GEMINI_MODEL are read by llm.py.

# Maximum number of rewriting attempts.
# rewrite.py also enforces a maximum of two attempts.
REWRITE_ATTEMPTS = 2
