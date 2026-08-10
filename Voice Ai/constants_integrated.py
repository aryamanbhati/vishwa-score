"""
ArthaSetu + Xscore — Unified Constants
=======================================
Central config for the integrated voice financial advisor.
"""

# ── API Keys — read from environment. Never hardcode. ──
# In Databricks prod: dbutils.secrets.get(scope="arthasetu", key="sarvam_api_key")
import os
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")

# ── Unity Catalog — ArthaSetu ──
ARTHASETU_CATALOG = "arthasetu"
ARTHASETU_BRONZE  = "arthasetu.bronze"
ARTHASETU_SILVER  = "arthasetu.silver"
ARTHASETU_GOLD    = "arthasetu.gold"

# ── Unity Catalog — Xscore ──
XSCORE_CATALOG = "xscore"
XSCORE_BRONZE  = "xscore.bronze"
XSCORE_SILVER  = "xscore.silver"
XSCORE_GOLD    = "xscore.gold"

# ── Volume Paths ──
ARTHASETU_UPLOADS  = "/Volumes/arthasetu/bronze/uploads"
XSCORE_UPLOADS     = "/Volumes/xscore/bronze/govt_raw"
FAISS_INDEX_PATH   = "/Volumes/arthasetu/gold/rag_files/rag.index"
FAISS_META_PATH    = "/Volumes/arthasetu/gold/rag_files/rag_meta.pkl"

# ── Xscore Tables ──
XSCORE_TABLES = {
    "bronze": [
        "calibration_stats", "home_credit_raw", "india_loan_raw",
        "paysim_raw", "synthetic_bills", "synthetic_profiles",
        "synthetic_upi_txns", "upi_raw",
    ],
    "silver": [
        "bill_payments", "upi_transactions",
        "user_features", "user_profiles",
    ],
    "gold": [
        "best_hyperparams", "credit_feature_store",
        "credit_scores", "score_explanations",
    ],
    "models": ["credit_scorer"],
}

# ── ArthaSetu Tables ──
ARTHASETU_TABLES = {
    "bronze": [
        "gov_schemes_raw", "bhashbench_finance_raw",
        "rural_loan_raw", "pmmy_state_raw",
    ],
    "silver": [
        "loan_schemes", "finance_qa",
        "rural_profiles_agg", "state_context",
    ],
    "gold": [
        "rag_corpus", "rag_embeddings", "voice_sessions",
    ],
}

# ── Sarvam AI Config ──
LANG_CODE = {
    "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "mr": "mr-IN",
    "bn": "bn-IN", "gu": "gu-IN", "pa": "pa-IN", "en": "en-IN",
    "kn": "kn-IN", "ml": "ml-IN",
}

LANG_NAMES = {
    "hi": "Hindi",    "ta": "Tamil",     "te": "Telugu",
    "mr": "Marathi",  "bn": "Bengali",   "gu": "Gujarati",
    "pa": "Punjabi",  "en": "English",   "kn": "Kannada",
    "ml": "Malayalam",
}

LANG_SPEAKER = {
    "hi": "anushka",  "ta": "arya",      "te": "manisha",
    "mr": "vidya",    "bn": "karun",     "gu": "hitesh",
    "pa": "abhilash", "kn": "anushka",   "ml": "arya",
    "en": "anushka",
}

SUPPORTED_LANGS = set(LANG_CODE.keys())

# ── Model Config ──
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
SARVAM_LLM      = "sarvam-m"
SARVAM_ASR      = "saaras:v3"
SARVAM_TTS      = "bulbul:v2"
XSCORE_MODEL    = "xscore_credit_scoring_lgbm"

# ── Scheme Eligibility Thresholds ──
SCHEME_THRESHOLDS = {
    "PM SVANidhi Tier 1":  {"min_score": 400, "max_loan": 10000},
    "PM SVANidhi Tier 2":  {"min_score": 550, "max_loan": 20000},
    "PM SVANidhi Tier 3":  {"min_score": 650, "max_loan": 50000},
    "PMMY Mudra Shishu":   {"min_score": 300, "max_loan": 50000},
    "PMMY Mudra Kishor":   {"min_score": 550, "max_loan": 500000},
    "PMMY Mudra Tarun":    {"min_score": 700, "max_loan": 1000000},
    "PM Vishwakarma T1":   {"min_score": 350, "max_loan": 100000},
    "PM Vishwakarma T2":   {"min_score": 500, "max_loan": 200000},
    "Kisan Credit Card":   {"min_score": 400, "max_loan": 300000},
}
