from pyspark.sql import SparkSession
import subprocess
subprocess.run(["pip", "install", "sentence-transformers", "faiss-cpu", "sarvamai", "-q"], check=True)
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
from sarvamai import SarvamAI
import numpy as np
import pandas as pd
import faiss
import pickle
import tempfile
import base64
import scipy.io.wavfile as wav
import os

spark = SparkSession.builder.getOrCreate()
spark.sql("USE CATALOG arthasetu")

# ── CONFIG ────────────────────────────────────────
SARVAM_KEY = "REVOKED_SARVAM_KEY_SEE_ENV"  # paste your key
# os.environ["SARVAM_API_KEY"] = SARVAM_KEY
client = SarvamAI(api_subscription_key=SARVAM_KEY)

print([m for m in dir(client) if not m.startswith('_')])
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
    "hi": "anushka",
    "ta": "arya",
    "te": "manisha",
    "mr": "vidya",
    "bn": "karun",
    "gu": "hitesh",
    "pa": "abhilash",
    "kn": "anushka",
    "ml": "arya",
    "en": "anushka"
}

print("✅ Sarvam client ready")

# ── LOAD FAISS ────────────────────────────────────
VOLUME_PATH = "/Volumes/arthasetu/gold/rag_files"
FAISS_PATH = f"{VOLUME_PATH}/rag.index"
META_PATH = f"{VOLUME_PATH}/rag_meta.pkl"

embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
index = faiss.read_index(FAISS_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

print(f"✅ FAISS ready — {index.ntotal} vectors")


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


# ── TRACK 1: ASR ─────────────────────────────────
def run_asr(audio_np, sr=16000):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, sr, (audio_np * 32768).astype(np.int16))

    import requests
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

    return {
        "raw_text": raw_text,
        "language": lang,
        "english_text": english_text
    }

print("✅ ASR function ready")


# ── TRACK 2: LLM ─────────────────────────────────
def generate_response(english_query, language, chunks,
                      xscore=510, user_state="India"):
    context = "\n".join([
        f"- [{c['content_type']}] {c['content'][:250]}"
        for c in chunks[:4]
    ])
    lang_name = LANG_NAMES.get(language, "Hindi")

    import requests, re
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

    # Clean <think> tags — handle both complete and incomplete tags
    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
    reply = re.sub(r'<think>.*', '', reply, flags=re.DOTALL).strip()

    if not reply:
        reply = "Aapke liye sahi yojana dhundh rahe hain. Kripya dobara puchein."

    return reply
print("✅ LLM function ready")
def text_to_speech(text, language):
    lang_code = LANG_CODE.get(language, "hi-IN")
    speaker = LANG_SPEAKER.get(language, "anushka")

    # Clean think tags here too as safety net
    import requests, re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Truncate to 490 chars (API limit 500)
    if len(text) > 490:
        text = text[:490]
    
    # Skip if empty
    if not text:
        print("  TTS skipped — empty text")
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

    print("TTS STATUS:", resp.status_code)

    if "audios" in data:
        return base64.b64decode(data["audios"][0])
    elif "error" in data:
        print("TTS ERROR:", data["error"])
        return b""
    else:
        return b""

# ── FULL PIPELINE ─────────────────────────────────
def run_full_pipeline(audio_np, user_state="Uttar Pradesh", xscore=510):
    print("\n" + "=" * 50)
    print("ArthaSetu — Full Voice Pipeline")
    print("=" * 50)

    # TRACK 1
    print("\n[1/3] ASR — Sarvam Saaras v3")
    t1 = run_asr(audio_np)
    print(f"  Raw     : {t1['raw_text'] or '(silent)'}")
    print(f"  Language: {LANG_NAMES.get(t1['language'])}")
    print(f"  English : {t1['english_text'] or '(empty)'}")

    # TRACK 2
    print("\n[2/3] RAG + Sarvam-m LLM")
    query = t1["english_text"] or "I need a loan"
    chunks = retrieve(query, state=user_state)
    print(f"  Retrieved: {len(chunks)} chunks")
    reply = generate_response(query, t1["language"],
                              chunks, xscore, user_state)
    print(f"  Reply: {reply[:80]}...")

    # TRACK 3
    print("\n[3/3] TTS — Sarvam Bulbul v2")
    audio_bytes = text_to_speech(reply, t1["language"])
    print(f"  Audio: {len(audio_bytes)} bytes")

    # LOG to Delta
    spark.createDataFrame(pd.DataFrame([{
        "user_state": user_state,
        "language": t1["language"],
        "raw_query": t1["raw_text"],
        "english_query": t1["english_text"],
        "xscore": xscore,
        "reply": reply,
        "timestamp": str(pd.Timestamp.now())
    }])).write.format("delta").mode("append") \
        .saveAsTable("arthasetu.gold.voice_sessions")

    print("\n✅ Pipeline complete!")
    return {
        "language": t1["language"],
        "raw_query": t1["raw_text"],
        "reply": reply,
        "audio_bytes": audio_bytes
    }


# ── TEXT-ONLY TEST (no mic needed) ────────────────
print("\n" + "=" * 50)
print("Running text-only tests (no microphone needed)")
print("=" * 50)

test_cases = [
    ("I need a loan for vegetables", "hi", "Uttar Pradesh"),
    ("I need a loan for farming", "ta", "Tamil Nadu"),
    ("I need money for my small shop", "te", "Andhra Pradesh"),
    ("Loan for my dairy farm", "mr", "Maharashtra"),
]

for query, lang, state in test_cases:
    print(f"\n--- {LANG_NAMES[lang]} | {state} ---")
    chunks = retrieve(query, state=state)
    reply = generate_response(query, lang, chunks, 480, state)
    audio = text_to_speech(reply, lang)
    print(f"  Reply : {reply[:70]}...")
    print(f"  Audio : {len(audio)} bytes")
# Check what methods sarvam client has
print([m for m in dir(client) if not m.startswith('_')])
print("\n" + "=" * 50)
print("✅ Voice Pipeline complete — all 4 languages tested")
print("=" * 50)
print("Next: Deploy Streamlit app")
