"""Sarvam AI client — ASR (Saaras v3), LLM (sarvam-m 24B), TTS (Bulbul v2).

Reads SARVAM_API_KEY from the environment. All methods are synchronous and
return plain Python objects so the caller doesn't need to know HTTP details.
"""

import base64
import os
import re
import tempfile

import requests

SARVAM_BASE = "https://api.sarvam.ai"

LANG_CODE = {
    "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "mr": "mr-IN",
    "bn": "bn-IN", "gu": "gu-IN", "pa": "pa-IN", "en": "en-IN",
    "kn": "kn-IN", "ml": "ml-IN",
}
LANG_NAMES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi", "en": "English",
    "kn": "Kannada", "ml": "Malayalam",
}
LANG_SPEAKER = {
    "hi": "anushka", "ta": "arya", "te": "manisha", "mr": "vidya",
    "bn": "karun", "gu": "hitesh", "pa": "abhilash", "kn": "anushka",
    "ml": "arya", "en": "anushka",
}


class SarvamClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "SARVAM_API_KEY not set. Get one at https://www.sarvam.ai "
                "and add it to your .env file."
            )

    def _headers_sub(self) -> dict:
        return {"API-Subscription-Key": self.api_key}

    def _headers_bearer(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def asr(self, audio_bytes: bytes) -> dict:
        """Speech-to-text + translate using Sarvam Saaras v3.

        Returns {"raw_text", "english_text", "language"}.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp.close()
            with open(tmp.name, "rb") as f:
                resp = requests.post(
                    f"{SARVAM_BASE}/speech-to-text-translate",
                    headers=self._headers_sub(),
                    files={"file": f},
                    data={
                        "model": "saaras:v3",
                        "prompt": "Rural Indian user asking about loans and financial schemes",
                    },
                    timeout=30,
                )
            resp.raise_for_status()
            data = resp.json()
        finally:
            os.unlink(tmp.name)

        raw_text = data.get("transcript", "")
        english_text = data.get("translated_text", raw_text)

        lang = "hi"
        if raw_text:
            try:
                lang_resp = requests.post(
                    f"{SARVAM_BASE}/text/detect-language",
                    headers={**self._headers_sub(), "Content-Type": "application/json"},
                    json={"input": raw_text},
                    timeout=10,
                )
                lang = lang_resp.json().get("language_code", "hi-IN").split("-")[0]
            except Exception:
                lang = "hi"

        return {"raw_text": raw_text, "english_text": english_text, "language": lang}

    def generate(
        self,
        query: str,
        language: str,
        context: str,
        xscore: int = 510,
        user_state: str = "India",
    ) -> str:
        """Generate a response using sarvam-m 24B LLM."""
        lang_name = LANG_NAMES.get(language, "Hindi")
        resp = requests.post(
            f"{SARVAM_BASE}/v1/chat/completions",
            headers=self._headers_bearer(),
            json={
                "model": "sarvam-m",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are ArthaSetu, a friendly rural financial advisor in India. "
                            "Help rural people understand government loan schemes. "
                            "Always use simple words a village person can understand. "
                            "Never use complex financial jargon. "
                            "Do NOT include any thinking or reasoning tags. "
                            "Just give the answer directly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User asked: {query}\n"
                            f"State: {user_state} | XScore: {xscore}/900\n\n"
                            f"Relevant scheme information:\n{context}\n\n"
                            f"Respond in simple {lang_name}. "
                            f"Max 3 sentences. Mention the best matching scheme name "
                            f"and loan amount. Be warm and encouraging."
                        ),
                    },
                ],
                "temperature": 0.3,
                "max_tokens": 250,
            },
            timeout=30,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
        reply = re.sub(r"<think>.*", "", reply, flags=re.DOTALL).strip()
        return reply or "Aapke liye sahi yojana dhundh rahe hain. Kripya dobara puchein."

    def tts(self, text: str, language: str) -> bytes:
        """Text-to-speech using Sarvam Bulbul v2. Returns raw WAV bytes."""
        lang_code = LANG_CODE.get(language, "hi-IN")
        speaker = LANG_SPEAKER.get(language, "anushka")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if len(text) > 490:
            text = text[:490]
        if not text:
            return b""

        resp = requests.post(
            f"{SARVAM_BASE}/text-to-speech",
            headers={**self._headers_sub(), "Content-Type": "application/json"},
            json={
                "inputs": [text],
                "target_language_code": lang_code,
                "speaker": speaker,
                "model": "bulbul:v2",
                "enable_preprocessing": True,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "audios" in data and data["audios"]:
            return base64.b64decode(data["audios"][0])
        return b""
