# 🕯️ Audiyoo — voice-only Zork

Speak into the dark; the dungeon speaks back. A browser plays **Zork** entirely by
voice: your mic streams to the server, **Silero VAD** finds your spoken commands,
**Gemma** (multimodal — it *hears* you, no speech-to-text step) narrates as the
game, and **Cartesia Sonic** speaks the reply. The UI is a phosphor-CRT
oscilloscope — green trace is you, amber is the dungeon. No text on screen.

## Quick start

```sh
bun install
cp .env.example .env          # then put your real CARTESIA_API_KEY in .env
bun start                     # -> http://localhost:3000
```

Open it in **Chrome/Edge**, click *enter the underground*, and talk.

**Prerequisites**
- [Bun](https://bun.sh)
- A running `llama-server` on `:8001` serving a **multimodal** Gemma
  (e.g. `unsloth/gemma-4-E4B-it-GGUF:Q4_K_M`), reachable at `LLAMA_URL`.
- A [Cartesia](https://cartesia.ai) API key — **or** a local
  [supertonic](https://pypi.org/project/supertonic/) sidecar with
  `TTS_BACKEND=supertonic` (see Configuration).
- `ffmpeg` is **only** needed to run the tests, not the server.

## How it works

```
 browser: AudioContext@16kHz → AudioWorklet → Int16 PCM
    │  WebSocket /ws  (mic pauses while the dungeon speaks → no echo)
    ▼
 server.ts → reframe to 512-sample windows → Silero VAD (vad.ts, onnxruntime-node)
    │                                            │ utterance boundaries → wav
    ├───────────────────────────────▶ llama-server :8001  (Gemma hears the wav)
    │   {type:"narration",text} ◀──────────────── narration
    ▼
 browser plays it via POST /api/tts → Cartesia (or local supertonic)
```

Server-side VAD (not browser-side) was chosen so the tricky segmentation logic is
**QA-able headlessly** — see `tests/`. The browser only forwards audio.

### WebSocket `/ws`
- client → server: `{type:"hello",sessionId}`, then binary Int16LE PCM @16kHz
  mono, then `{type:"resume"}` after each spoken reply finishes playing.
- server → client: `ready` · `speech_start` · `thinking` · `narration` · `error`.

### HTTP
`POST /api/start` (opening scene) · `POST /api/type` (typed fallback) ·
`POST /api/tts` (Cartesia) · `GET /` (UI).

## Configuration (env / `.env`)

| Var | Default | Notes |
|---|---|---|
| `TTS_BACKEND` | `cartesia` | `cartesia` or `supertonic` (local sidecar, see below) |
| `CARTESIA_API_KEY` | — | **required** unless `TTS_BACKEND=supertonic` |
| `CARTESIA_VOICE_ID` | `79f8b5fb-…` (Theo – Modern Narrator) | `curl -H "X-API-Key: $KEY" -H "Cartesia-Version: 2026-03-01" https://api.cartesia.ai/voices` |
| `CARTESIA_MODEL` | `sonic-3.5` | latest Sonic model |
| `CARTESIA_VERSION` | `2026-03-01` | Cartesia API version header |
| `SUPERTONIC_URL` | `http://127.0.0.1:7788` | supertonic sidecar base URL |
| `SUPERTONIC_VOICE` | `M5` | |
| `SUPERTONIC_STEPS` | `5` | diffusion steps; more = better/slower |
| `LLAMA_URL` | `http://0.0.0.0:8001` | |
| `LLAMA_MODEL` | `zork-best` | model name/alias your llama-server exposes |
| `LLAMA_API_KEY` | — | sent as `Authorization: Bearer` if set |
| `PORT` | `3000` | |
| `DEBUG_TRANSCRIPT` | off | `1` = Gemma prefixes `[heard: …]` (logged + sent to client, stripped from TTS/history) |
| `DEBUG_AUDIO` | off | `1` (= `debug/`) or a dir: dump each utterance WAV the LLM hears |

### Local TTS: the supertonic sidecar

`TTS_BACKEND=supertonic` expects a [supertonic](https://pypi.org/project/supertonic/)
server running at `SUPERTONIC_URL`. **The deployer must start it** alongside
`bun start` — it is not spawned automatically:

```sh
uv run --with 'supertonic[serve]' supertonic serve --port 7788 --log-level warning &
```

First run downloads model weights; keep the working directory stable so the
cache is reused. The server buffers whole clips (no streaming); at `steps=5`
short replies land in ~500ms once warm.

## Tests (headless, no browser)

```sh
bun run test:vad     # WAV → VAD → prints utterance segments (+ dumps seg_N.wav)
bun start &          # ws test needs the server up
bun run test:ws      # streams WAVs over /ws like the browser mic; checks turns
```

## ⚠️ Before you expose this to the world

This server is **unauthenticated**, backed by a **single GPU** (requests
serialize), and **bills your Cartesia key**. Built-in guards are modest: per-call
length caps on `/api/tts` and `/api/type`, a max-utterance cap in the VAD, and a
bounded session map. They are **not** a security boundary. Before going public:

- Put it behind **auth or a shared link token**, and a **tunnel** (cloudflared /
  ngrok) rather than a raw port-forward.
- Set a **hard spend cap** on the Cartesia key.
- Add **rate limiting** and a **concurrency=1–2 queue** so the GPU degrades
  gracefully instead of melting.

## Tuning the feel

- VAD sensitivity lives in `vad.ts` (`DEFAULTS`): `positiveSpeechThreshold`,
  `redemptionFrames` (how long a pause ends your turn), `minSpeechFrames`.
- The dungeon's personality is the `ZORK_SYSTEM` prompt in `server.ts`.

## Layout

```
server.ts            Bun + Hono: /ws, /api/*, serves the UI
vad.ts               streaming Silero VAD state machine + WAV encoder
models/              vendored silero_vad_v5.onnx
public/index.html    voice-only CRT-oscilloscope UI (self-contained)
tests/               headless VAD + WS harnesses and audio fixtures
```
