"""
ArthaSetu + Xscore — Integrated Voice Financial Advisor
=========================================================
Merges:
  - Xscore credit scoring (50k synthetic profiles, LGBM model)
  - ArthaSetu RAG pipeline (gov schemes, BhashaBench, rural profiles)
  - Sarvam AI voice (ASR → LLM → TTS) in 10 Indian languages

Deploy: streamlit run arthasetu_integrated_app.py
"""

import subprocess
subprocess.run(
    ["pip", "install", "streamlit", "sentence-transformers", "faiss-cpu", "sarvamai", "-q"],
    check=True,
)

import streamlit as st
import numpy as np
import base64
import requests
import re
import faiss
import pickle
import tempfile
import json
from sentence_transformers import SentenceTransformer

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SARVAM_KEY = "REVOKED_SARVAM_KEY_SEE_ENV"

LANG_CODE = {
    "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "mr": "mr-IN",
    "bn": "bn-IN", "gu": "gu-IN", "pa": "pa-IN", "en": "en-IN",
    "kn": "kn-IN", "ml": "ml-IN",
}
LANG_NAMES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi",
    "en": "English", "kn": "Kannada", "ml": "Malayalam",
}
LANG_SPEAKER = {
    "hi": "anushka", "ta": "arya", "te": "manisha", "mr": "vidya",
    "bn": "karun", "gu": "hitesh", "pa": "abhilash",
    "kn": "anushka", "ml": "arya", "en": "anushka",
}

# Volume paths (Databricks Unity Catalog)
ARTHASETU_FAISS = "/Volumes/arthasetu/gold/rag_files/rag.index"
ARTHASETU_META  = "/Volumes/arthasetu/gold/rag_files/rag_meta.pkl"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD RESOURCES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def load_rag():
    """Load FAISS index + metadata from ArthaSetu gold layer."""
    embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    index = faiss.read_index(ARTHASETU_FAISS)
    with open(ARTHASETU_META, "rb") as f:
        metadata = pickle.load(f)
    return embed_model, index, metadata


@st.cache_resource
def load_xscore_data():
    """
    Load Xscore credit data from gold tables.
    In production this reads from:
      - xscore.gold.credit_scores
      - xscore.gold.score_explanations
      - xscore.gold.credit_feature_store
      - xscore.silver.user_profiles
      - xscore.silver.user_features
      - xscore.silver.upi_transactions
      - xscore.silver.bill_payments
    """
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No Spark")

        # Credit scores
        scores_df = spark.table("xscore.gold.credit_scores").toPandas()

        # Score explanations (SHAP)
        explanations_df = spark.table("xscore.gold.score_explanations").toPandas()

        # User profiles
        profiles_df = spark.table("xscore.silver.user_profiles").toPandas()

        # User features
        features_df = spark.table("xscore.silver.user_features").toPandas()

        # Merge into single lookup
        merged = profiles_df.merge(scores_df, on="user_id", how="left")
        merged = merged.merge(features_df, on="user_id", how="left")

        return merged, explanations_df

    except Exception as e:
        st.warning(f"Spark unavailable — using demo profiles. ({e})")
        return None, None


embed_model, faiss_index, rag_metadata = load_rag()
xscore_profiles, xscore_explanations = load_xscore_data()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# XSCORE USER LOOKUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_user_context(user_id: str) -> dict:
    """
    Pull a user's credit profile from Xscore tables.
    Returns dict with score, factors, UPI stats, bills, eligible schemes.
    """
    # If Spark data is available, use real data
    if xscore_profiles is not None:
        row = xscore_profiles[xscore_profiles["user_id"] == user_id]
        if not row.empty:
            r = row.iloc[0].to_dict()
            # Get SHAP explanations
            factors = []
            if xscore_explanations is not None:
                exp_row = xscore_explanations[
                    xscore_explanations["user_id"] == user_id
                ]
                if not exp_row.empty:
                    factors = json.loads(
                        exp_row.iloc[0].get("shap_factors", "[]")
                    ) if isinstance(
                        exp_row.iloc[0].get("shap_factors"), str
                    ) else []
            return {
                "user_id": user_id,
                "name": r.get("name", "User"),
                "city": r.get("city", "Unknown"),
                "state": r.get("state", "India"),
                "occupation": r.get("occupation", "Unknown"),
                "annual_income": r.get("annual_income", 0),
                "credit_score": int(r.get("credit_score", r.get("xscore", 500))),
                "score_factors": factors,
                "upi_txn_count": int(r.get("upi_txn_count", r.get("total_upi_txns", 0))),
                "upi_total_volume": float(r.get("upi_total_volume", r.get("total_upi_volume", 0))),
                "bills_on_time_pct": float(r.get("bills_on_time_pct", r.get("bill_ontime_rate", 0))),
            }

    # Fallback: demo user
    return {
        "user_id": user_id,
        "name": "Demo User",
        "city": "Lucknow",
        "state": "Uttar Pradesh",
        "occupation": "Textile Trader",
        "annual_income": 285000,
        "credit_score": 510,
        "score_factors": [
            {"factor": "UPI consistency", "impact": "+38"},
            {"factor": "Bill payments", "impact": "+25"},
            {"factor": "Low savings", "impact": "-18"},
            {"factor": "No credit history", "impact": "-32"},
        ],
        "upi_txn_count": 847,
        "upi_total_volume": 423500,
        "bills_on_time_pct": 0.72,
    }


def get_eligible_schemes(credit_score: int, occupation: str) -> list:
    """Match user to eligible government schemes based on score + occupation."""
    schemes = []
    occ = occupation.lower()

    # Always eligible if score > 0
    if credit_score >= 300:
        schemes.append({
            "name": "PMMY Mudra Shishu",
            "amount": "₹50,000",
            "interest": "10%",
            "collateral": "No",
        })

    if credit_score >= 450:
        schemes.append({
            "name": "PM SVANidhi Tier 1",
            "amount": "₹10,000",
            "interest": "7%",
            "collateral": "No",
        })

    if credit_score >= 550:
        schemes.append({
            "name": "PMMY Mudra Kishor",
            "amount": "₹5,00,000",
            "interest": "12%",
            "collateral": "Partial",
        })

    if any(k in occ for k in ["artisan", "tailor", "weaver", "carpenter", "potter", "blacksmith", "barber"]):
        schemes.append({
            "name": "PM Vishwakarma Tier 1",
            "amount": "₹1,00,000",
            "interest": "5%",
            "collateral": "No",
        })

    if any(k in occ for k in ["farmer", "dairy", "agriculture", "cattle", "fish"]):
        schemes.append({
            "name": "Kisan Credit Card",
            "amount": "₹3,00,000",
            "interest": "7%",
            "collateral": "Land records",
        })

    return schemes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG RETRIEVAL (ArthaSetu)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def retrieve(query: str, state: str = None, top_k: int = 5) -> list:
    vec = embed_model.encode([query]).astype(np.float32)
    faiss.normalize_L2(vec)
    scores, idxs = faiss_index.search(vec, top_k * 3)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = rag_metadata[idx].copy()
        chunk["score"] = round(float(score), 3)
        if state and state.lower() in chunk.get("state", "").lower():
            chunk["score"] += 0.1
        results.append(chunk)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SARVAM AI — ASR / LLM / TTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_asr(audio_bytes_raw: bytes) -> tuple:
    """Speech → Text using Sarvam Saaras v3."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes_raw)
    tmp.flush()

    with open(tmp.name, "rb") as f:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text-translate",
            headers={"API-Subscription-Key": SARVAM_KEY},
            files={"file": f},
            data={
                "model": "saaras:v3",
                "prompt": "Rural Indian user asking about loans, credit score, and financial schemes",
            },
        )
    data = resp.json()
    raw_text = data.get("transcript", "")
    english_text = data.get("translated_text", raw_text)

    # Detect language
    lang = "hi"
    if raw_text:
        lang_resp = requests.post(
            "https://api.sarvam.ai/text/detect-language",
            headers={
                "API-Subscription-Key": SARVAM_KEY,
                "Content-Type": "application/json",
            },
            json={"input": raw_text},
        )
        lang = lang_resp.json().get("language_code", "hi-IN").split("-")[0]

    return raw_text, english_text, lang


def generate_response(
    english_query: str,
    language: str,
    rag_chunks: list,
    user_ctx: dict,
) -> str:
    """
    RAG + Xscore-aware LLM response via Sarvam-m.
    The system prompt now includes:
      - User's credit score + factors
      - Eligible schemes
      - RAG context from ArthaSetu
    """
    # Build RAG context
    rag_context = "\n".join(
        [f"- [{c['content_type']}] {c['content'][:250]}" for c in rag_chunks[:4]]
    )

    # Build Xscore context
    schemes = get_eligible_schemes(
        user_ctx["credit_score"], user_ctx.get("occupation", "")
    )
    scheme_text = "\n".join(
        [f"  • {s['name']}: up to {s['amount']} at {s['interest']}, collateral: {s['collateral']}"
         for s in schemes]
    ) or "  No specific schemes matched — suggest Mudra Shishu as default."

    factors_text = "\n".join(
        [f"  • {f.get('factor', f.get('feature', '?'))}: {f.get('impact', '?')}"
         for f in user_ctx.get("score_factors", [])[:5]]
    ) or "  Score factors not available."

    lang_name = LANG_NAMES.get(language, "Hindi")

    system_prompt = (
        "You are ArthaSetu, a warm and friendly financial advisor for rural India. "
        "You help people understand their credit score, government loan schemes, and basic financial literacy. "
        "Always use simple words that a village person can understand. "
        "Never use complex financial jargon. "
        "Do NOT include any thinking or reasoning tags. Just give the answer directly. "
        "If the user asks about their credit score, explain what it means and how to improve it. "
        "If they ask about loans, recommend the best matching scheme. "
        "If they ask general finance questions, teach them simply."
    )

    user_prompt = (
        f"User asked: {english_query}\n\n"
        f"═══ USER PROFILE (from Xscore) ═══\n"
        f"  Name: {user_ctx.get('name', 'User')}\n"
        f"  City: {user_ctx.get('city', 'Unknown')}, {user_ctx.get('state', 'India')}\n"
        f"  Occupation: {user_ctx.get('occupation', 'Unknown')}\n"
        f"  Annual Income: ₹{user_ctx.get('annual_income', 0):,}\n"
        f"  Credit Score (Xscore): {user_ctx['credit_score']}/900\n"
        f"  UPI Transactions: {user_ctx.get('upi_txn_count', 0)} total\n"
        f"  Bill Payment Rate: {user_ctx.get('bills_on_time_pct', 0):.0%}\n\n"
        f"═══ SCORE FACTORS (SHAP) ═══\n{factors_text}\n\n"
        f"═══ ELIGIBLE SCHEMES ═══\n{scheme_text}\n\n"
        f"═══ RAG KNOWLEDGE BASE ═══\n{rag_context}\n\n"
        f"Respond in simple {lang_name}. "
        f"Max 4 sentences. Be warm, encouraging, and specific to this user's situation. "
        f"Mention their score and best scheme if relevant."
    )

    resp = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {SARVAM_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "sarvam-m",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 300,
        },
    )
    data = resp.json()
    reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Clean <think> tags
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    reply = re.sub(r"<think>.*", "", reply, flags=re.DOTALL).strip()

    if not reply:
        reply = "Aapke liye sahi yojana dhundh rahe hain. Kripya dobara puchein."

    return reply


def text_to_speech(text: str, language: str) -> bytes:
    """Text → Speech using Sarvam Bulbul v2."""
    lang_code = LANG_CODE.get(language, "hi-IN")
    speaker = LANG_SPEAKER.get(language, "anushka")

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    if len(text) > 490:
        text = text[:490]
    if not text:
        return b""

    resp = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={
            "API-Subscription-Key": SARVAM_KEY,
            "Content-Type": "application/json",
        },
        json={
            "inputs": [text],
            "target_language_code": lang_code,
            "speaker": speaker,
            "model": "bulbul:v2",
            "enable_preprocessing": True,
        },
    )
    data = resp.json()
    if "audios" in data:
        return base64.b64decode(data["audios"][0])
    return b""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DELTA LOGGING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def log_session(user_ctx, language, raw_query, english_query, reply):
    """Log voice session to arthasetu.gold.voice_sessions."""
    try:
        from pyspark.sql import SparkSession
        import pandas as pd

        spark = SparkSession.getActiveSession()
        if spark:
            spark.createDataFrame(
                pd.DataFrame([{
                    "user_id": user_ctx.get("user_id", "unknown"),
                    "user_state": user_ctx.get("state", "unknown"),
                    "xscore": user_ctx.get("credit_score", 0),
                    "language": language,
                    "raw_query": raw_query,
                    "english_query": english_query,
                    "reply": reply,
                    "timestamp": str(pd.Timestamp.now()),
                }])
            ).write.format("delta").mode("append").saveAsTable(
                "arthasetu.gold.voice_sessions"
            )
    except Exception:
        pass  # Don't break UI if logging fails


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STREAMLIT UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="ArthaSetu + Xscore",
    page_icon="🏦",
    layout="wide",
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

.stApp { font-family: 'Outfit', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f4c3a 0%, #1a7a5c 50%, #2d9b6e 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.main-header p { opacity: 0.85; margin: 0.5rem 0 0 0; font-size: 1.05rem; }

.score-card {
    background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 1.5rem;
    color: white;
    text-align: center;
    margin-bottom: 1rem;
}
.score-number {
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
}
.score-label { opacity: 0.7; font-size: 0.85rem; margin-top: 0.3rem; }

.score-good { color: #4ade80; }
.score-fair { color: #fbbf24; }
.score-poor { color: #f87171; }

.factor-positive { color: #4ade80; font-weight: 600; }
.factor-negative { color: #f87171; font-weight: 600; }

.scheme-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.scheme-card h4 { margin: 0 0 0.3rem 0; color: #166534; }

.pipeline-step {
    background: #f8fafc;
    border-left: 3px solid #2d9b6e;
    padding: 0.5rem 1rem;
    margin: 0.3rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="main-header">
    <h1>🏦 ArthaSetu + Xscore</h1>
    <p>Voice-Enabled Financial Advisor · Credit Scoring · 10 Indian Languages · Powered by Sarvam AI + Databricks</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: User Profile ──
with st.sidebar:
    st.markdown("### 👤 User Profile")

    # In production, this comes from login / Xscore DB
    user_id = st.text_input("User ID", value="USR-2026-04837",
                             help="Enter Xscore user ID to load profile")

    user_state = st.selectbox("State", [
        "Uttar Pradesh", "Bihar", "Maharashtra", "Tamil Nadu",
        "Karnataka", "West Bengal", "Gujarat", "Rajasthan",
        "Madhya Pradesh", "Odisha", "Andhra Pradesh", "Telangana",
    ])

    # Load user context
    user_ctx = get_user_context(user_id)
    user_ctx["state"] = user_state  # Override with selection

    # Display credit score
    score = user_ctx["credit_score"]
    score_class = "score-good" if score >= 700 else "score-fair" if score >= 500 else "score-poor"
    score_band = "Good" if score >= 700 else "Fair" if score >= 500 else "Needs Work"

    st.markdown(f"""
    <div class="score-card">
        <div class="score-label">XSCORE CREDIT SCORE</div>
        <div class="score-number {score_class}">{score}</div>
        <div class="score-label">{score_band} · out of 900</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Name:** {user_ctx.get('name', 'N/A')}")
    st.markdown(f"**Occupation:** {user_ctx.get('occupation', 'N/A')}")
    st.markdown(f"**Income:** ₹{user_ctx.get('annual_income', 0):,}/yr")
    st.markdown(f"**UPI Txns:** {user_ctx.get('upi_txn_count', 0)}")
    st.markdown(f"**Bills On-Time:** {user_ctx.get('bills_on_time_pct', 0):.0%}")

    st.divider()
    st.markdown("### 📊 Score Factors")
    for f in user_ctx.get("score_factors", []):
        impact = f.get("impact", "")
        impact_str = str(impact)
        css_class = "factor-positive" if "+" in impact_str or (isinstance(impact, (int, float)) and impact > 0) else "factor-negative"
        st.markdown(
            f'<span class="{css_class}">{impact_str}</span> {f.get("factor", f.get("feature", "?"))}',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### 🌐 Languages")
    st.caption("Hindi · Tamil · Telugu · Marathi · Bengali · Gujarati · Kannada · Malayalam · Punjabi · English")

# ── Main Area ──
tab_voice, tab_dashboard, tab_schemes = st.tabs(["🎙️ Voice Chat", "📊 Dashboard", "📋 Schemes"])

# ━━━ TAB 1: VOICE CHAT ━━━
with tab_voice:
    col_input, col_output = st.columns(2, gap="large")

    with col_input:
        st.markdown("### 🎙️ Ask in Your Language")
        st.caption("Speak or type — ArthaSetu understands 10 Indian languages")

        audio_input = st.audio_input("Record your question")

        st.markdown("---")
        st.markdown("**Or type your question:**")
        text_query = st.text_input(
            "Type here",
            placeholder="e.g. mujhe sabzi ke liye loan chahiye / mera credit score kya hai",
            label_visibility="collapsed",
        )
        text_lang = st.selectbox(
            "Language for typed query",
            list(LANG_NAMES.keys()),
            format_func=lambda x: LANG_NAMES[x],
        )

        ask_btn = st.button(
            "🚀 Get My Recommendation",
            type="primary",
            use_container_width=True,
        )

        if ask_btn:
            if audio_input:
                with st.spinner("🎧 Listening with Sarvam Saaras..."):
                    raw_text, english_text, lang = run_asr(audio_input.read())
                st.markdown(
                    f'<div class="pipeline-step">✅ ASR → Detected <b>{LANG_NAMES.get(lang, lang)}</b></div>',
                    unsafe_allow_html=True,
                )
            elif text_query:
                raw_text = text_query
                english_text = text_query
                lang = text_lang
            else:
                st.warning("Please record audio or type a question")
                st.stop()

            # RAG retrieval
            with st.spinner("🔍 Searching ArthaSetu knowledge base..."):
                chunks = retrieve(english_text or "I need a loan", state=user_state)
            st.markdown(
                f'<div class="pipeline-step">✅ RAG → Retrieved {len(chunks)} relevant chunks</div>',
                unsafe_allow_html=True,
            )

            # LLM response (Xscore-aware)
            with st.spinner("🧠 Generating personalized advice with Sarvam-m..."):
                reply = generate_response(english_text, lang, chunks, user_ctx)
            st.markdown(
                f'<div class="pipeline-step">✅ LLM → Response in {LANG_NAMES.get(lang, lang)}</div>',
                unsafe_allow_html=True,
            )

            # TTS
            with st.spinner("🔊 Converting to speech with Bulbul..."):
                audio_bytes = text_to_speech(reply, lang)
            st.markdown(
                f'<div class="pipeline-step">✅ TTS → {len(audio_bytes):,} bytes audio</div>',
                unsafe_allow_html=True,
            )

            # Log to Delta
            log_session(user_ctx, lang, raw_text, english_text, reply)

            # Store result
            st.session_state["voice_result"] = {
                "lang": lang,
                "raw": raw_text,
                "english": english_text,
                "reply": reply,
                "audio": audio_bytes,
                "chunks": chunks,
            }

    with col_output:
        st.markdown("### 💬 ArthaSetu Says")

        if "voice_result" in st.session_state:
            r = st.session_state["voice_result"]

            st.info(f"**Language:** {LANG_NAMES.get(r['lang'], r['lang'])} · **Xscore:** {user_ctx['credit_score']}/900")

            if r["raw"]:
                st.markdown(f"**You said:** {r['raw']}")
            if r["english"] and r["english"] != r["raw"]:
                st.markdown(f"**English:** {r['english']}")

            st.success(r["reply"])

            if r["audio"]:
                st.audio(r["audio"], format="audio/wav")

            with st.expander("📚 Knowledge Sources Used"):
                for c in r.get("chunks", []):
                    st.markdown(
                        f"- **[{c['content_type']}]** {c.get('title', '')[:60]} — score: {c['score']}"
                    )
        else:
            st.markdown(
                """
                <div style="text-align:center; padding: 3rem; opacity: 0.5;">
                    <div style="font-size: 4rem;">🎙️</div>
                    <p>Record your voice or type a question,<br>then click <b>Get My Recommendation</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ━━━ TAB 2: DASHBOARD ━━━
with tab_dashboard:
    st.markdown("### 📊 Your Financial Dashboard")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Xscore", f"{user_ctx['credit_score']}/900", delta=score_band)
    d2.metric("UPI Transactions", f"{user_ctx.get('upi_txn_count', 0):,}")
    d3.metric("Annual Income", f"₹{user_ctx.get('annual_income', 0):,}")
    d4.metric("Bills On-Time", f"{user_ctx.get('bills_on_time_pct', 0):.0%}")

    st.markdown("---")

    imp_col, tip_col = st.columns(2)

    with imp_col:
        st.markdown("#### 📈 How to Improve Your Score")
        tips = []
        if user_ctx.get("bills_on_time_pct", 0) < 0.9:
            tips.append("⚡ Pay all bills (electricity, gas, mobile) on time via UPI")
        if user_ctx.get("upi_txn_count", 0) < 500:
            tips.append("📱 Use UPI for daily purchases to build transaction history")
        tips.extend([
            "💰 Start a small recurring deposit (even ₹500/month helps)",
            "📋 Get a micro-insurance policy (LIC/PMJJBY) for credit profile",
            "🏦 Apply for PM SVANidhi to build formal loan repayment record",
        ])
        for tip in tips:
            st.markdown(tip)

    with tip_col:
        st.markdown("#### 🎯 Score Breakdown")
        for f in user_ctx.get("score_factors", []):
            impact = f.get("impact", "")
            factor = f.get("factor", f.get("feature", "Unknown"))
            st.markdown(f"{'🟢' if '+' in str(impact) else '🔴'} **{factor}** → {impact}")

# ━━━ TAB 3: ELIGIBLE SCHEMES ━━━
with tab_schemes:
    st.markdown("### 📋 Government Schemes You Qualify For")
    st.caption(f"Based on Xscore {user_ctx['credit_score']}/900 · {user_ctx.get('occupation', 'N/A')} · {user_state}")

    schemes = get_eligible_schemes(user_ctx["credit_score"], user_ctx.get("occupation", ""))

    if schemes:
        for s in schemes:
            st.markdown(f"""
            <div class="scheme-card">
                <h4>🏛️ {s['name']}</h4>
                <b>Max Amount:</b> {s['amount']} · <b>Interest:</b> {s['interest']} · <b>Collateral:</b> {s['collateral']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Building your profile — keep using UPI and paying bills to unlock schemes!")

    st.markdown("---")
    st.caption("Data sources: PM SVANidhi, Mudra (PMMY), Kisan Credit Card, PM Vishwakarma, NABARD, SHG Bank Linkage")
