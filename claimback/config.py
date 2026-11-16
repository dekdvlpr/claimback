"""All the settings for Claimback live here. Change things in THIS file, not in the others."""

# ---------------------------------------------------------------------------
# 1. Which model scores complaints?
#    "stub"  = fake scores from keywords (works from minute one, no downloads)
#    "tfidf" = Person A's quick baseline model (needs models/tfidf.joblib)
#    "hf"    = the real fine-tuned models from Hugging Face (needs torch + transformers)
# ---------------------------------------------------------------------------
MODEL_BACKEND = "hf"

# Used when MODEL_BACKEND = "hf". Person A fills these in after training.
# They can be Hugging Face names ("your-org/claimback-win-v1") or local folders ("models/win").
HF_WIN_MODEL = "InuraDekumpitiya/claimback-win-v1"
HF_ISSUE_MODEL = "InuraDekumpitiya/claimback-issue-v1"

# Files Person A produces (put them in the models/ folder)
TFIDF_PATH = "models/tfidf.joblib"          # baseline model
CALIBRATOR_PATH = "models/calibrator.joblib"  # makes percentages honest (optional)
BASE_RATES_PATH = "data/base_rates.csv"     # money-back rate per company + issue (optional)

# The 6 issue groups. Person A trains the issue model with EXACTLY this list, in this order.
ISSUE_GROUPS = [
    "Account problems",
    "Card or transaction dispute",
    "Debt collection",
    "Fees and charges",
    "Loan or mortgage servicing",
    "Scam or unauthorised payment",
]

# ---------------------------------------------------------------------------
# 2. The LLM used for the checklist, questions and rewrite.
#    Needs ANTHROPIC_API_KEY in a .env file. Without a key the app still works
#    using simple offline rules (worse, but fine for building the UI).
# ---------------------------------------------------------------------------
LLM_MODEL = "claude-opus-5-5"
LLM_EFFORT = "low"        # "low" keeps it fast for a live demo
REWRITE_ATTEMPTS = 3      # tries at most 3 rewrites; stops at the first that passes both fact guards and scores higher
