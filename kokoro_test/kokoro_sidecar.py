# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["kokoro>=0.9", "soundfile", "numpy", "fastapi", "uvicorn"]
# ///
"""
Persistent Kokoro TTS sidecar. Loads the model ONCE at boot, warms it up
(one-time kernel JIT on GPU), then serves POST /tts {text, voice}
-> audio/wav (24kHz s16le). This is the shape TTS_BACKEND=kokoro in server.ts
points /api/tts at. Provisioning, run commands per GPU vendor, and gotchas:
see README.md, "Local GPU TTS: the Kokoro sidecar".

  uv run kokoro_sidecar.py            # NVIDIA/CPU: inline deps resolve stock torch
  # AMD/ROCm boxes need the ROCm torch wheel instead — see the README section

Boot pays the one-time JIT cost (~4s); each request logs gen time + RTF so you
can see steady-state is ~200ms / 5s audio (RTF ~0.04 on a 7900 XT).
"""
import os, time, io, sys, json, traceback

# ROCm userspace (system install); harmless no-op paths on non-AMD boxes.
os.environ.setdefault("LD_LIBRARY_PATH", "/opt/rocm/lib")
os.environ.setdefault("PATH", "/opt/rocm/bin:" + os.environ["PATH"])

import numpy as np
import torch
if torch.version.hip:
    # ROCm only: the wheel's bundled MIOpen 3.4 HIPRTC clashes with GCC 16
    # <type_traits>; force native LSTM kernels (no quality loss at 82M — see
    # README ROCm note). Leave cuDNN enabled for NVIDIA.
    torch.backends.cudnn.enabled = False

from kokoro import KPipeline
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import Response
import uvicorn

PORT = int(os.environ.get("KOKORO_PORT", "7100"))
DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "am_onyx,am_michael")
# Slower delivery reads more DM-like; 0.9 is a good default. <1.0 slows, >1.0 speeds up.
DEFAULT_SPEED = float(os.environ.get("KOKORO_SPEED", "0.9"))
# cuda covers both NVIDIA and ROCm builds of torch; CPU fallback keeps a
# GPU-less box working (slower — see README for expectations).
DEVICE = os.environ.get("KOKORO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

t_boot = time.perf_counter()
print(f"[boot] loading KPipeline on {DEVICE}...", flush=True)
pl = KPipeline(lang_code="a", device=DEVICE)
print(f"[boot] pipeline load: {(time.perf_counter()-t_boot)*1000:.0f}ms", flush=True)

# warmup: trigger all HIP kernel JIT / autotune so the first real request is fast
t0 = time.perf_counter()
for _ in pl("warmup sentence for gpu kernel compilation and autotune.", voice=DEFAULT_VOICE, speed=1.0):
    pass
print(f"[boot] warmup (JIT): {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)
print(f"[boot] pre-warm total: {(time.perf_counter()-t_boot)*1000:.0f}ms — resident on GPU, ready", flush=True)

app = FastAPI()

def wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

@app.post("/tts")
async def tts(req: Request):
    body = await req.json()
    text = (body.get("text") or "").strip()[:800]
    voice = body.get("voice") or DEFAULT_VOICE
    speed = float(body.get("speed") or DEFAULT_SPEED)
    if not text:
        return Response(content=b"empty", status_code=400)
    try:
        t0 = time.perf_counter()
        chunks = [np.array(r.audio) for r in pl(text, voice=voice, speed=speed)]
        gen = (time.perf_counter() - t0) * 1000
        audio = np.concatenate(chunks)
        sr = 24000
        dur = len(audio) / sr
        data = wav_bytes(audio, sr)
        print(f"[tts] {voice:<14} spd {speed:.2f} gen {gen:5.0f}ms  audio {dur:4.2f}s  RTF {gen/1000/dur:.3f}", flush=True)
        return Response(content=data, media_type="audio/wav",
                        headers={"X-Gen-Ms": f"{gen:.0f}", "X-Audio-Sec": f"{dur:.2f}"})
    except Exception as e:
        traceback.print_exc()
        return Response(content=str(e).encode(), status_code=500)

@app.get("/health")
async def health():
    return {"ok": True, "voice": DEFAULT_VOICE, "device": DEVICE}

if __name__ == "__main__":
    print(f"[sidecar] listening on http://127.0.0.1:{PORT}/tts  (POST {{text,voice}})", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
