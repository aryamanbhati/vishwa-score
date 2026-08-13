# ─────────────────────────────────────────────
# ArthaSetu — Central config & constants
# ─────────────────────────────────────────────

# API Keys — read from environment. Never hardcode. See .env.example.
import os
SARVAM_API_KEY    = os.environ.get("SARVAM_API_KEY", "")
HF_TOKEN          = os.environ.get("HF_TOKEN", "")

# Unity Catalog
CATALOG           = "arthasetu"
BRONZE            = "arthasetu.bronze"
SILVER            = "arthasetu.silver"
GOLD              = "arthasetu.gold"

# DBFS paths
DBFS_BASE         = "/dbfs/arthasetu"
FAISS_INDEX_PATH  = "/dbfs/arthasetu/rag.index"
FAISS_META_PATH   = "/dbfs/arthasetu/rag_meta.pkl"
VOLUME_PATH       = "/Volumes/arthasetu/bronze/uploads"

# Sarvam language codes
LANG_CODE = {
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "en": "en-IN"
}

LANG_NAMES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "en": "English"
}

LANG_SPEAKER = {
    "hi": "meera",
    "ta": "pavithra",
    "te": "arvind",
    "mr": "aarav",
    "bn": "aditi",
    "gu": "manisha",
    "pa": "arjun",
    "kn": "amol",
    "ml": "priya",
    "en": "meera"
}

SUPPORTED_LANGS   = set(LANG_CODE.keys())
EMBEDDING_MODEL   = "paraphrase-MiniLM-L6-v2"
SARVAM_LLM        = "sarvam-m"
SARVAM_ASR        = "saaras:v3"
SARVAM_TTS        = "bulbul:v2"