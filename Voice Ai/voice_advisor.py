import os
import requests
import base64
import re
import tempfile

SARVAM_KEY = os.environ.get("SARVAM_API_KEY", "")

LANG_CODE = {"hi":"hi-IN","ta":"ta-IN","te":"te-IN","mr":"mr-IN","bn":"bn-IN","gu":"gu-IN","pa":"pa-IN","en":"en-IN","kn":"kn-IN","ml":"ml-IN"}
LANG_NAMES = {"hi":"Hindi","ta":"Tamil","te":"Telugu","mr":"Marathi","bn":"Bengali","gu":"Gujarati","pa":"Punjabi","en":"English","kn":"Kannada","ml":"Malayalam"}
LANG_SPEAKER = {"hi":"anushka","ta":"arya","te":"manisha","mr":"vidya","bn":"karun","gu":"hitesh","pa":"abhilash","kn":"anushka","ml":"arya","en":"anushka"}

SCHEME_CONTEXT = """
Available government loan schemes for rural India:

1. PM SVANidhi Tier 1: Street vendor loan up to Rs 10,000. No collateral. 7% interest subsidy. Need vending certificate.
2. PM SVANidhi Tier 2: Rs 20,000 for vendors who repaid Tier 1. Same interest subsidy.
3. PM SVANidhi Tier 3: Rs 50,000 for vendors with excellent repayment.
4. PMMY Mudra Shishu: Micro business loan up to Rs 50,000. No collateral. For small shops, vendors, artisans.
5. PMMY Mudra Kishor: Rs 50,000 to Rs 5 lakh for growing businesses. Need 1 year business history.
6. PMMY Mudra Tarun: Rs 5 lakh to Rs 10 lakh for established businesses.
7. Kisan Credit Card: Farm loan up to Rs 3 lakh at 7% interest (4% effective with subsidy). For crop, dairy, fishery.
8. NABARD Dairy Scheme: Up to Rs 7 lakh for dairy farms. 25% subsidy (33% for SC/ST).
9. SHG Bank Linkage: Women self-help groups get up to Rs 10 lakh collective loan. No individual collateral.
10. PM Vishwakarma Tier 1: Rs 1 lakh at 5% for artisans (blacksmith, carpenter, tailor, weaver). Includes skill training.
11. PM Vishwakarma Tier 2: Rs 2 lakh for artisans who repaid Tier 1.
"""


def run_asr(audio_bytes_raw):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes_raw)
    tmp.flush()
    with open(tmp.name, "rb") as f:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text-translate",
            headers={"API-Subscription-Key": SARVAM_KEY},
            files={"file": f},
            data={"model": "saaras:v3", "prompt": "Rural Indian user asking about loans and financial schemes"}
        )
    data = resp.json()
    raw_text = data.get("transcript", "")
    english_text = data.get("translated_text", raw_text)
    lang = "hi"
    if raw_text:
        try:
            lang_resp = requests.post(
                "https://api.sarvam.ai/text/detect-language",
                headers={"API-Subscription-Key": SARVAM_KEY, "Content-Type": "application/json"},
                json={"input": raw_text}
            )
            lang = lang_resp.json().get("language_code", "hi-IN").split("-")[0]
        except:
            lang = "hi"
    return raw_text, english_text, lang


def generate_response(query, language, xscore=510, user_state="India"):
    lang_name = LANG_NAMES.get(language, "Hindi")
    try:
        resp = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {SARVAM_KEY}", "Content-Type": "application/json"},
            json={"model": "sarvam-m", "messages": [
                {"role": "system", "content": (
                    "You are ArthaSetu, a friendly rural financial advisor in India. "
                    "Help rural people understand government loan schemes. "
                    "Always use simple words a village person can understand. "
                    "Never use complex financial jargon. "
                    "Do NOT include any thinking or reasoning tags. Just give the answer directly."
                )},
                {"role": "user", "content": (
                    f"User asked: {query}\n"
                    f"State: {user_state} | XScore: {xscore}/900\n\n"
                    f"Available schemes:\n{SCHEME_CONTEXT}\n\n"
                    f"Respond in simple {lang_name}. "
                    f"Max 3 sentences. Mention the best matching scheme name and loan amount. "
                    f"Be warm and encouraging."
                )}
            ], "temperature": 0.3, "max_tokens": 250},
            timeout=30
        )
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
        reply = re.sub(r"<think>.*", "", reply, flags=re.DOTALL).strip()
        if not reply:
            reply = "Aapke liye sahi yojana dhundh rahe hain. Kripya dobara puchein."
        return reply
    except Exception as e:
        return f"Error: {str(e)}"


def text_to_speech(text, language):
    lang_code = LANG_CODE.get(language, "hi-IN")
    speaker = LANG_SPEAKER.get(language, "anushka")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    if len(text) > 490:
        text = text[:490]
    if not text:
        return b""
    try:
        resp = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"API-Subscription-Key": SARVAM_KEY, "Content-Type": "application/json"},
            json={"inputs": [text], "target_language_code": lang_code, "speaker": speaker, "model": "bulbul:v2", "enable_preprocessing": True},
            timeout=30
        )
        data = resp.json()
        if "audios" in data:
            return base64.b64decode(data["audios"][0])
    except:
        pass
    return b""
