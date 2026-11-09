"""
CineGen AI Studio (Python / Streamlit)

Run:  streamlit run cinegen_studio.py
Install:  pip install streamlit edge-tts moviepy pillow numpy requests imageio-ffmpeg
"""
import asyncio
import base64
import io
import os
import tempfile
import urllib.parse

import edge_tts
import numpy as np
import requests
import streamlit as st
from PIL import Image

# moviepy v2 / v1 compatibility
try:
    from moviepy import VideoClip, AudioFileClip, concatenate_videoclips
    MP_V2 = True
except ImportError:
    from moviepy.editor import VideoClip, AudioFileClip, concatenate_videoclips
    MP_V2 = False

W, H = 1280, 720

PROVIDERS = {
    "Pollinations (free, no key)": {
        "id": "pollinations", "key": False, "model": "flux",
        "note": "No key. Anonymous use is rate-limited (about 1 image per 15 s) and may be watermarked.",
    },
    "Hugging Face (free token, small credit)": {
        "id": "huggingface", "key": True, "model": "black-forest-labs/FLUX.1-schnell",
        "note": "Free token from huggingface.co/settings/tokens. Free monthly credit is small.",
    },
    "Together AI (API key)": {
        "id": "together", "key": True, "model": "black-forest-labs/FLUX.1-schnell",
        "note": "Needs an API key. Check Together's model list for a free model name.",
    },
    "OpenAI (paid key)": {
        "id": "openai", "key": True, "model": "gpt-image-1",
        "note": "Paid. Needs an OpenAI API key with billing enabled.",
    },
}

VOICES = {
    "Nigerian Male (Abeo)": "en-NG-AbeoNeural",
    "Nigerian Female (Ezinne)": "en-NG-EzinneNeural",
    "US Male (Guy)": "en-US-GuyNeural",
    "US Female (Jenny)": "en-US-JennyNeural",
    "US Female (Aria)": "en-US-AriaNeural",
    "UK Male (Ryan)": "en-GB-RyanNeural",
    "UK Female (Sonia)": "en-GB-SoniaNeural",
}

PRESETS = {  # pitch offset in Hz
    "Normal": 0,
    "Child / young boy": 30,
    "Woman": 10,
    "Very old man (deep)": -30,
}

MOTIONS = ["Zoom in", "Zoom out", "Pan left", "Pan right"]

st.set_page_config(page_title="CineGen AI Studio", layout="centered")
st.title("🎬 CineGen AI Studio")

ss = st.session_state
ss.setdefault("scenes", [])
ss.setdefault("pitch", 0)
ss.setdefault("cur_img", None)
ss.setdefault("video", None)


# ===================== IMAGE PROVIDERS =====================
def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def fetch_image(pid: str, prompt: str, key: str, model: str) -> bytes:
    if pid == "pollinations":
        url = (
            "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
            + f"?model={model}&width={W}&height={H}&nologo=true"
        )
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Pollinations error {r.status_code}. Wait 15 s and retry.")
        return r.content

    if not key:
        raise RuntimeError("Enter your API key first.")

    if pid == "huggingface":
        r = requests.post(
            f"https://router.huggingface.co/hf-inference/models/{model}",
            headers={"Authorization": f"Bearer {key}"},
            json={"inputs": prompt},
            timeout=180,
        )
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
            raise RuntimeError(f"Hugging Face: {r.status_code} {r.text[:200]}")
        return r.content

    if pid == "together":
        r = requests.post(
            "https://api.together.xyz/v1/images/generations",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "prompt": prompt, "width": 1024, "height": 768,
                  "steps": 4, "n": 1, "response_format": "b64_json"},
            timeout=180,
        )
        j = r.json()
        if r.status_code != 200 or "data" not in j:
            raise RuntimeError(f"Together: {r.status_code} {str(j)[:200]}")
        return base64.b64decode(j["data"][0]["b64_json"])

    if pid == "openai":
        r = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "prompt": prompt, "size": "1536x1024"},
            timeout=180,
        )
        j = r.json()
        if r.status_code != 200 or "data" not in j:
            raise RuntimeError(f"OpenAI: {r.status_code} {str(j)[:200]}")
        d = j["data"][0]
        if d.get("b64_json"):
            return base64.b64decode(d["b64_json"])
        return requests.get(d["url"], timeout=120).content

    raise RuntimeError("Unknown provider.")


# ===================== VOICE =====================
async def _tts(text, voice, pitch, rate, path):
    comm = edge_tts.Communicate(text, voice, pitch=f"{pitch:+d}Hz", rate=f"{rate:+d}%")
    await comm.save(path)


def make_speech(text, voice, pitch, rate, path):
    asyncio.run(_tts(text, voice, pitch, rate, path))


# ===================== VIDEO =====================
def motion_clip(img_bytes: bytes, motion: str, duration: float):
    """Ken Burns style zoom / pan clip built from one image."""
    src = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    scale = max(W / src.width, H / src.height)
    big = src.resize((int(src.width * scale) + 1, int(src.height * scale) + 1), Image.LANCZOS)
    bw, bh = big.size

    def frame(t):
        p = min(1.0, t / duration) if duration > 0 else 0.0
        if motion == "Zoom in":
            z, pan = 1 + 0.2 * p, 0.5
        elif motion == "Zoom out":
            z, pan = 1.2 - 0.2 * p, 0.5
        elif motion == "Pan left":
            z, pan = 1.2, 1 - p
        else:  # Pan right
            z, pan = 1.2, p
        cw, ch = bw / z, bh / z
        x0 = (bw - cw) * pan
        y0 = (bh - ch) / 2
        out = big.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
        return np.array(out)

    return VideoClip(frame, duration=duration)


def render(scenes, voice, pitch, rate, progress):
    tmp = tempfile.mkdtemp()
    clips = []
    for i, sc in enumerate(scenes):
        progress.progress(i / len(scenes), f"Scene {i + 1} of {len(scenes)}")
        audio = None
        duration = float(sc["sec"])
        if sc["text"].strip():
            apath = os.path.join(tmp, f"voice_{i}.mp3")
            make_speech(sc["text"], voice, pitch, rate, apath)
            audio = AudioFileClip(apath)
            duration = max(duration, audio.duration + 0.3)
        clip = motion_clip(sc["image"], sc["motion"], duration)
        if audio is not None:
            clip = clip.with_audio(audio) if MP_V2 else clip.set_audio(audio)
        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose")
    out = os.path.join(tmp, "cinegen_video.mp4")
    final.write_videofile(out, fps=24, codec="libx264", audio_codec="aac", logger=None)
    progress.progress(1.0, "Done")
    with open(out, "rb") as f:
        return f.read()


# ===================== UI: VOICE =====================
st.header("1. Voice")
voice_name = st.selectbox("Voice", list(VOICES.keys()))
voice = VOICES[voice_name]

preset = st.selectbox("Preset", list(PRESETS.keys()))
if st.button("Apply preset", use_container_width=True):
    ss.pitch = PRESETS[preset]
    st.rerun()

rate = st.slider("Speed", -50, 50, 0, 5, format="%d%%")
st.write(f"**Pitch:** {ss.pitch:+d} Hz")
c1, c2, c3 = st.columns(3)
if c1.button("⬇️ Too high\n(lower it)", use_container_width=True):
    ss.pitch = max(-50, ss.pitch - 5)
    st.rerun()
if c2.button("🔄 Reset", use_container_width=True):
    ss.pitch = 0
    st.rerun()
if c3.button("⬆️ Too low\n(raise it)", use_container_width=True):
    ss.pitch = min(50, ss.pitch + 5)
    st.rerun()

if st.button("🔊 Preview voice", use_container_width=True):
    try:
        p = os.path.join(tempfile.mkdtemp(), "preview.mp3")
        make_speech("Hello, this is how my voice will sound in your video.",
                    voice, ss.pitch, rate, p)
        with open(p, "rb") as f:
            st.audio(f.read(), format="audio/mp3")
    except Exception as e:
        st.error(f"Voice preview failed: {e}")

# ===================== UI: IMAGE =====================
st.header("2. Make an image")
prov_name = st.selectbox("AI engine", list(PROVIDERS.keys()))
prov = PROVIDERS[prov_name]
st.caption(prov["note"])
key = st.text_input("API key / token", type="password") if prov["key"] else ""
model = st.text_input("Model name", value=prov["model"])
prompt = st.text_area("Describe your scene", height=90)
tweak = st.text_input("Tweak (optional)")

if st.button("⚡ Generate", use_container_width=True):
    full = ", ".join(x for x in [prompt.strip(), tweak.strip()] if x)
    if not full:
        st.warning("Describe your image first.")
    else:
        with st.spinner("Generating image..."):
            try:
                ss.cur_img = fetch_image(prov["id"], full, key.strip(), model.strip())
            except Exception as e:
                st.error(f"Failed: {e}")

up = st.file_uploader("...or use your own image", type=["png", "jpg", "jpeg", "webp"])
if up is not None:
    ss.cur_img = up.read()

if ss.cur_img:
    st.image(ss.cur_img, use_container_width=True)

# ===================== UI: ADD SCENE =====================
st.header("3. Add to storyboard")
motion = st.selectbox("Camera motion", MOTIONS)
sec = st.number_input("Minimum seconds", 1, 30, 4)
narration = st.text_area("Narration for this scene (optional)", height=80)

if st.button("➕ Add scene", type="primary", use_container_width=True):
    if not ss.cur_img:
        st.warning("Make or upload an image first.")
    else:
        ss.scenes.append({"image": ss.cur_img, "motion": motion,
                          "sec": int(sec), "text": narration})
        st.rerun()

for i, sc in enumerate(ss.scenes):
    with st.container(border=True):
        st.image(sc["image"], width=200)
        st.caption(f"{sc['motion']} · {sc['sec']}s · {sc['text'] or 'no narration'}")
        a, b, c = st.columns(3)
        if a.button("⬆️", key=f"up{i}") and i > 0:
            ss.scenes[i - 1], ss.scenes[i] = ss.scenes[i], ss.scenes[i - 1]
            st.rerun()
        if b.button("⬇️", key=f"dn{i}") and i < len(ss.scenes) - 1:
            ss.scenes[i + 1], ss.scenes[i] = ss.scenes[i], ss.scenes[i + 1]
            st.rerun()
        if c.button("🗑️", key=f"del{i}"):
            ss.scenes.pop(i)
            st.rerun()

# ===================== UI: RENDER =====================
st.header("4. Make video")
if st.button("🎬 Make video", type="primary", use_container_width=True,
             disabled=not ss.scenes):
    bar = st.progress(0.0, "Starting...")
    try:
        ss.video = render(ss.scenes, voice, ss.pitch, rate, bar)
    except Exception as e:
        st.error(f"Render failed: {e}")

if ss.video:
    st.video(ss.video)
    st.caption("Voice too high or low? Use the pitch buttons at the top, then make the video again.")
    st.download_button("⬇️ Download MP4", ss.video, file_name="cinegen_video.mp4",
                       mime="video/mp4", use_container_width=True)
