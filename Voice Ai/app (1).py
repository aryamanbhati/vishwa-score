import subprocess
subprocess.run(["pip", "install", "streamlit", "sentence-transformers", "faiss-cpu", "sarvamai", "-q"], check=True)

import streamlit as st
import numpy as np
import scipy.io.wavfile as wavfile
import io
import base64
import requests
import re
import faiss
import pickle
import tempfile
from sentence_transformers import SentenceTransformer

# ── CONFIG ────────────────────────────────────────
SARVAM_KEY = "REVOKED_SARVAM_KEY_SEE_ENV"

LANG_CODE = {
    "hi":"hi-IN","ta":"ta-IN","te":"te-IN","mr":"mr-IN",
    "bn":"bn-IN","gu":"gu-IN","pa":"pa-IN","en":"en-IN",
    "kn":"kn-IN","ml":"ml-IN"
}
LANG_NAMES = {
    "hi":"Hindi","ta":"Tamil","te":"Telugu","mr":"Marathi",
    "bn":"Bengali","gu":"Gujarati","pa":"Punjabi",
    "en":"English","kn":"Kannada","ml":"Malayalam"
}
LANG_SPEAKER = {
    "hi":"anushka","ta":"arya","te":"manisha","mr":"vidya",
    "bn":"karun","gu":"hitesh","pa":"abhilash",
    "kn":"anushka","ml":"arya","en":"anushka"
}

# ── LOAD RAG ──────────────────────────────────────
@st.cache_resource
def load_rag():
    embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    index = faiss.read_index("/Volumes/arthasetu/gold/rag_files/rag.index")
    with open("/Volumes/arthasetu/gold/rag_files/rag_meta.pkl", "rb") as f:
        metadata = pickle.load(f)
    return embed_model, index, metadata

embed_model, index, metadata = load_rag()

def retrieve(query, state=None, top_k=5):
    vec = embed_model.encode([query]).astype(np.float32)
    faiss.normalize_L2(vec)
    scores, idxs = index.search(vec, top_k * 3)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = metadata[idx].copy()
        chunk["score"] = round(float(score), 3)
        if state and state.lower() in chunk.get("state", "").lower():
            chunk["score"] += 0.1
        results.append(chunk)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# ── LLM ───────────────────────────────────────────
def generate_response(english_query, language, chunks, xscore, user_state):
    context = "\n".join([
        f"- [{c['content_type']}] {c['content'][:250]}"
        for c in chunks[:4]
    ])
    lang_name = LANG_NAMES.get(language, "Hindi")

    resp = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {SARVAM_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "sarvam-m",
            "messages": [
                {"role": "system", "content": (
                    "You are ArthaSetu, a friendly rural financial advisor in India. "
                    "Help rural people understand government loan schemes. "
                    "Always use simple words a village person can understand. "
                    "Never use complex financial jargon. "
                    "Do NOT include any thinking or reasoning. Just give the answer directly."
                )},
                {"role": "user", "content": (
                    f"User asked: {english_query}\n"
                    f"State: {user_state} | XScore: {xscore}/900\n\n"
                    f"Relevant scheme information:\n{context}\n\n"
                    f"Respond in simple {lang_name}. "
                    f"Max 3 sentences. Mention best scheme and amount. "
                    f"Be warm and encouraging."
                )}
            ],
            "temperature": 0.3,
            "max_tokens": 250
        }
    )
    data = resp.json()
    reply = data["choices"][0]["message"]["content"]
    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
    reply = re.sub(r'<think>.*', '', reply, flags=re.DOTALL).strip()
    if not reply:
        reply = "Aapke liye sahi yojana dhundh rahe hain. Kripya dobara puchein."
    return reply

# ── TTS ───────────────────────────────────────────
def text_to_speech(text, language):
    lang_code = LANG_CODE.get(language, "hi-IN")
    speaker = LANG_SPEAKER.get(language, "anushka")
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
    if len(text) > 490:
        text = text[:490]
    if not text:
        return b""

    resp = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={
            "API-Subscription-Key": SARVAM_KEY,
            "Content-Type": "application/json"
        },
        json={
            "inputs": [text],
            "target_language_code": lang_code,
            "speaker": speaker,
            "model": "bulbul:v2",
            "enable_preprocessing": True
        }
    )
    data = resp.json()
    if "audios" in data:
        return base64.b64decode(data["audios"][0])
    return b""

# ── ASR ───────────────────────────────────────────
def run_asr(audio_bytes_raw):
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
                "prompt": "Rural Indian user asking about loans and financial schemes"
            }
        )
    data = resp.json()
    raw_text = data.get("transcript", "")
    english_text = data.get("translated_text", raw_text)

    lang = "hi"
    if raw_text:
        lang_resp = requests.post(
            "https://api.sarvam.ai/text/detect-language",
            headers={
                "API-Subscription-Key": SARVAM_KEY,
                "Content-Type": "application/json"
            },
            json={"input": raw_text}
        )
        lang = lang_resp.json().get("language_code", "hi-IN").split("-")[0]

    return raw_text, english_text, lang

# ── UI ────────────────────────────────────────────
st.set_page_config(page_title="ArthaSetu", page_icon="🏦", layout="wide")
st.title("🏦 ArthaSetu")
st.caption("Hyper-Personalised Rural Financial Advisor — Powered by Sarvam AI + Databricks")

with st.sidebar:
    st.header("👤 User Profile")
    user_state = st.selectbox("State", [
        "Uttar Pradesh","Bihar","Maharashtra","Tamil Nadu",
        "Karnataka","West Bengal","Gujarat","Rajasthan",
        "Madhya Pradesh","Odisha","Andhra Pradesh","Telangana"
    ])
    xscore = st.slider("XScore", 300, 900, 510,
        help="Credit readiness score")
    st.divider()
    st.subheader("🌐 Supported Languages")
    st.write("Hindi, Tamil, Telugu, Marathi, Bengali, Gujarati, Kannada, Malayalam, English")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("🎙️ Speak Your Query")
    st.caption("Ask in any Indian language")
    audio_input = st.audio_input("Record your question")

    # Text input as alternative
    text_query = st.text_input("Or type your question", placeholder="e.g. mujhe sabzi ke liye loan chahiye")
    text_lang = st.selectbox("Language for typed query", list(LANG_NAMES.keys()),
                              format_func=lambda x: LANG_NAMES[x])

    if st.button("Get My Recommendation", type="primary", use_container_width=True):
        if audio_input:
            with st.spinner("Listening..."):
                raw_text, english_text, lang = run_asr(audio_input.read())
        elif text_query:
            raw_text = text_query
            english_text = text_query
            lang = text_lang
        else:
            st.warning("Please record audio or type a question")
            st.stop()

        with st.spinner("Finding best schemes..."):
            chunks = retrieve(english_text or "I need a loan", state=user_state)
            reply = generate_response(english_text, lang, chunks, xscore, user_state)

        with st.spinner("Generating audio..."):
            audio_bytes = text_to_speech(reply, lang)

        st.session_state["result"] = {
            "lang": lang,
            "raw": raw_text,
            "english": english_text,
            "reply": reply,
            "audio": audio_bytes,
            "sources": list(set([c["content_type"] for c in chunks]))
        }

with col2:
    st.subheader("💬 Your Recommendation")
    if "result" in st.session_state:
        r = st.session_state["result"]
        st.info(f"**Language:** {LANG_NAMES.get(r['lang'], r['lang'])}")
        if r["raw"]:
            st.write("**You said:**", r["raw"])
        st.success(r["reply"])
        if r["audio"]:
            st.audio(r["audio"], format="audio/wav")
        with st.expander("📚 Data Sources"):
            for s in r["sources"]:
                st.write(f"• {s}")
    else:
        st.info("Record your voice or type a question, then click Get My Recommendation")