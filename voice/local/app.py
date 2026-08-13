"""ArthaSetu — Multilingual Voice Financial Advisor for Bharat.

Local-runnable Streamlit app. No Databricks dependency.

Usage:
    pip install -r voice/local/requirements.txt
    export SARVAM_API_KEY=sk_...
    streamlit run voice/local/app.py
"""

import io
import time

import streamlit as st

from retriever import retrieve, list_schemes
from sarvam_client import SarvamClient, LANG_NAMES

st.set_page_config(
    page_title="ArthaSetu — Voice Financial Advisor",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 ArthaSetu — Multilingual Voice Financial Advisor")
st.caption(
    "Ask about government loan schemes in **10 Indic languages**. "
    "Powered by Sarvam Saaras ASR → FAISS RAG → sarvam-m 24B LLM → Bulbul TTS."
)


@st.cache_resource
def get_client():
    return SarvamClient()


def format_context(chunks: list[dict]) -> str:
    return "\n".join(
        f"- [{c['scheme_name']}] {c['content'][:300]}" for c in chunks[:4]
    )


col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("Your question")

    input_mode = st.radio(
        "Input mode", ["Text", "Voice (microphone)"], horizontal=True
    )

    language = st.selectbox(
        "Response language",
        options=list(LANG_NAMES.keys()),
        format_func=lambda k: f"{LANG_NAMES[k]} ({k})",
        index=0,
    )

    user_state = st.selectbox(
        "Your state",
        [
            "All India", "Uttar Pradesh", "Maharashtra", "Tamil Nadu",
            "Karnataka", "Andhra Pradesh", "West Bengal", "Rajasthan",
            "Gujarat", "Madhya Pradesh", "Bihar", "Punjab", "Kerala",
            "Odisha", "Telangana", "Jharkhand", "Assam", "Chhattisgarh",
            "Haryana", "Uttarakhand",
        ],
    )

    xscore = st.slider("VishwaScore (credit score)", 300, 900, 510, step=10)

    query_text = ""
    audio_bytes = None

    if input_mode == "Text":
        query_text = st.text_area(
            "Type your question",
            placeholder="I need a loan for my vegetable cart",
            height=100,
        )
    else:
        audio_bytes = st.audio_input("Record your question")

    ask = st.button("Ask ArthaSetu", type="primary", use_container_width=True)

with col_output:
    st.subheader("ArthaSetu's answer")

    if ask:
        try:
            client = get_client()
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        detected_lang = language
        english_query = query_text

        if audio_bytes is not None:
            with st.spinner("Listening... (Sarvam Saaras v3 ASR)"):
                raw_audio = audio_bytes.read() if hasattr(audio_bytes, "read") else audio_bytes
                asr_result = client.asr(raw_audio)
            detected_lang = asr_result["language"]
            english_query = asr_result["english_text"]
            st.info(
                f"**Heard:** {asr_result['raw_text']}\n\n"
                f"**Language:** {LANG_NAMES.get(detected_lang, detected_lang)}\n\n"
                f"**English:** {english_query}"
            )

        if not english_query.strip():
            st.warning("Please enter or speak a question.")
            st.stop()

        with st.spinner("Searching schemes... (FAISS RAG)"):
            t0 = time.time()
            chunks = retrieve(english_query, state=user_state, top_k=5)
            retrieval_ms = (time.time() - t0) * 1000

        with st.expander(f"Retrieved {len(chunks)} schemes ({retrieval_ms:.0f}ms)", expanded=False):
            for c in chunks:
                st.markdown(
                    f"**{c['scheme_name']}** (score: {c['score']:.3f}) — "
                    f"target: {c['target_group']}"
                )
                st.caption(c["content"][:200] + "...")

        context = format_context(chunks)

        with st.spinner(f"Generating response in {LANG_NAMES.get(detected_lang, 'Hindi')}... (sarvam-m 24B)"):
            t0 = time.time()
            reply = client.generate(
                query=english_query,
                language=detected_lang,
                context=context,
                xscore=xscore,
                user_state=user_state,
            )
            llm_ms = (time.time() - t0) * 1000

        st.success(reply)
        st.caption(f"LLM latency: {llm_ms:.0f}ms")

        with st.spinner("Generating speech... (Sarvam Bulbul v2 TTS)"):
            t0 = time.time()
            audio_out = client.tts(reply, detected_lang)
            tts_ms = (time.time() - t0) * 1000

        if audio_out:
            st.audio(audio_out, format="audio/wav")
            st.caption(f"TTS latency: {tts_ms:.0f}ms")

        st.divider()
        st.caption(
            f"**Pipeline:** ASR → FAISS retrieval ({retrieval_ms:.0f}ms) → "
            f"sarvam-m LLM ({llm_ms:.0f}ms) → Bulbul TTS ({tts_ms:.0f}ms)"
        )

st.divider()

with st.expander("Scheme catalog (11 government loan schemes)"):
    schemes = list_schemes()
    for s in schemes:
        st.markdown(f"**{s['scheme_id']}. {s['scheme_name']}** — _{s['target_group']}_")
        st.caption(s["content"])
