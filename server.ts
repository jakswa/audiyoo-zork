// server.ts — Bun + Hono glue for the ZORK voice demo.
//
// The player's RAW MIC AUDIO goes straight into Gemma (multimodal) as the game
// command — no separate speech-to-text step. Gemma narrates back as the Zork
// engine; Cartesia Sonic speaks the narration. Audio in -> audio out.
//
//   POST /api/start  { sessionId }                  -> opening scene text
//   POST /api/say     multipart: sessionId, audio    -> Gemma hears it -> text
//   POST /api/type    { sessionId, text }            -> typed fallback  -> text
//   POST /api/tts     { text }                        -> Cartesia audio/wav
//
// Env: CARTESIA_API_KEY (required), LLAMA_URL, LLAMA_MODEL, CARTESIA_VOICE_ID,
//      CARTESIA_MODEL, CARTESIA_VERSION, PORT
import { Hono } from "hono";
import { createBunWebSocket } from "hono/bun";
import { SileroVAD, encodeWav, FRAME_SAMPLES } from "./vad";

const { upgradeWebSocket, websocket } = createBunWebSocket();

const TTS_BACKEND = (process.env.TTS_BACKEND ?? "supertonic") as "cartesia" | "supertonic" | "kokoro";
const CARTESIA_API_KEY = process.env.CARTESIA_API_KEY ?? "";
// Supertonic local sidecar (supertonic serve) — native /v1/tts endpoint lets us
// pass `steps` (the OpenAI-compatible /v1/audio/speech does not).
const SUPERTONIC_URL = process.env.SUPERTONIC_URL ?? "http://127.0.0.1:7788";
const SUPERTONIC_VOICE = process.env.SUPERTONIC_VOICE ?? "M5";
const SUPERTONIC_STEPS = Number(process.env.SUPERTONIC_STEPS ?? 5);
// Kokoro-82M local GPU sidecar (kokoro_sidecar.py, 7900XT/ROCm). Persistent +
// pre-warmed, ~200ms per utterance at RTF ~0.04. Default voice `am_onyx,am_michael`
// blends depth + clarity at speed 0.9 for DM gravitas — audition in
// kokoro_test/out/audition/.
const KOKORO_URL = process.env.KOKORO_URL ?? "http://127.0.0.1:7100";
const KOKORO_VOICE = process.env.KOKORO_VOICE ?? "am_onyx,am_michael";
const KOKORO_SPEED = Number(process.env.KOKORO_SPEED ?? "0.9");  // <1 = slower, more DM-like
const LLAMA_URL = process.env.LLAMA_URL ?? "http://0.0.0.0:8001";
const LLAMA_MODEL = process.env.LLAMA_MODEL ?? "zork-best";
const LLAMA_API_KEY = process.env.LLAMA_API_KEY ?? "";  // optional; llama-server may require it
// When true, the system prompt asks Gemma to prefix each reply with
// `[heard: <transcript>]` so you can see what it thinks you said. The tag is
// stripped before TTS and before storing in session history. See parseNarration().
const DEBUG_TRANSCRIPT = /^(1|true|yes)$/.test(process.env.DEBUG_TRANSCRIPT ?? "");
// When set to a directory ("1" means "debug/"), each utterance WAV that reaches
// the LLM is written to disk so you can hear exactly what Gemma hears. Use to
// diagnose mic capture / sample-rate / VAD-truncation issues.
const DEBUG_AUDIO = (process.env.DEBUG_AUDIO ?? "").replace(/^1$/, "debug");
// "Theo - Modern Narrator" — steady, enunciating narrator voice, fits a DM.
const VOICE_ID = process.env.CARTESIA_VOICE_ID ?? "79f8b5fb-2cc8-479a-80df-29f7a7cf1a3e";
const CARTESIA_MODEL = process.env.CARTESIA_MODEL ?? "sonic-3.5";   // latest Sonic
const CARTESIA_VERSION = process.env.CARTESIA_VERSION ?? "2026-03-01"; // latest API version
const PORT = Number(process.env.PORT ?? 3000);

if (TTS_BACKEND === "cartesia" && !CARTESIA_API_KEY) {
  console.error("Set CARTESIA_API_KEY before starting (or use the default TTS_BACKEND=supertonic).");
  process.exit(1);
}

const ZORK_SYSTEM = `You are the parser and narrator of ZORK, the classic text adventure set in the ruins of the Great Underground Empire.

The player speaks their commands aloud; you receive their spoken audio as each turn's input. Interpret what they said as a game command and respond as Zork would.

Voice and rules:
- Speak in terse, atmospheric second person ("You are standing...", "It is pitch black...").
- Your reply is spoken aloud, so use plain prose only: NO markdown, asterisks, bullets, or stage directions. 1 to 4 sentences.
- Stay fully in character as the game. Never mention being an AI, a model, or audio.
- Honor classic commands: look, go north/south/east/west/up/down, take/drop <thing>, open, read, inventory, examine. Accept loose natural speech too.
- If you cannot make out the audio, nudge in character ("Speak up, adventurer — the cavern swallows your words.").
- You may invent rooms, items, and a light plot to keep the adventure flowing, but keep Zork's tone: dry, slightly mischievous, dangerous.
- If the player does something fatal or absurd, narrate the consequences with wit.

The adventure opens at "West of House". On the first turn, set that scene.`;

// Debug variant: same narrator, but echo a transcript of the player's heard
// command as a `[heard: ...]` tag on its own first line, stripped before use.
const ZORK_SYSTEM_DEBUG = ZORK_SYSTEM + `

DEBUG MODE: Begin every reply with one line exactly of the form \`[heard: <transcript>]\` — your best guess at what the player said aloud, lowercased, brief. Then a newline, then your narration as normal. If you cannot make out the audio at all, write \`[heard: ???]\`.`;

const SYS_PROMPT = DEBUG_TRANSCRIPT ? ZORK_SYSTEM_DEBUG : ZORK_SYSTEM;

type Msg = { role: "system" | "user" | "assistant"; content: any };
const sessions = new Map<string, Msg[]>();
const MAX_SESSIONS = 200; // bound memory if this is exposed publicly
const MAX_HISTORY = 60;   // messages kept per session (besides the system prompt)

function freshSession(id: string): Msg[] {
  // simple LRU-ish eviction: Map preserves insertion order, drop the oldest
  while (sessions.size >= MAX_SESSIONS) sessions.delete(sessions.keys().next().value);
  const h: Msg[] = [{ role: "system", content: SYS_PROMPT }];
  sessions.set(id, h);
  return h;
}

// Split a debug-tagged reply into the heard transcript and the clean narration.
// Fallback: no tag -> whole string is narration, heard unknown. The tag is
// expected on the first line; anything after the first newline is narration.
function parseNarration(raw: string): { heard: string | undefined; text: string } {
  const m = raw.match(/^\[heard:\s*([^\]]*)\]\s*[\r\n]+([\s\S]*)/);
  if (!m) return { heard: undefined, text: raw };
  return { heard: m[1].trim() || "???", text: m[2].trim() };
}
function getSession(id: string): Msg[] {
  return sessions.get(id) ?? freshSession(id);
}

// Run the current session history through Gemma, store + return the narration.
async function narrate(history: Msg[]): Promise<{ heard: string | undefined; text: string }> {
  // keep the system prompt + the most recent turns so context can't grow forever
  if (history.length > MAX_HISTORY + 1) history.splice(1, history.length - (MAX_HISTORY + 1));
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (LLAMA_API_KEY) headers["Authorization"] = `Bearer ${LLAMA_API_KEY}`;
  const r = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
    method: "POST",
    headers,
    // Gemma 3 it models emit chain-of-thought in `reasoning_content`; disable it
    // so the narration budget goes to `content` (and latency stays low).
    body: JSON.stringify({
      model: LLAMA_MODEL, messages: history, temperature: 0.8, max_tokens: 320,
      chat_template_kwargs: { enable_thinking: false },
    }),
  });
  if (!r.ok) throw new Error(`llama-server ${r.status}: ${await r.text()}`);
  const data = await r.json();
  const raw: string = data.choices?.[0]?.message?.content?.trim() ?? "";
  const { heard, text } = DEBUG_TRANSCRIPT ? parseNarration(raw) : { heard: undefined as string | undefined, text: raw };
  history.push({ role: "assistant", content: text });  // store clean narration, not the debug tag
  if (heard !== undefined) console.log(`[heard] ${heard}`);
  return { heard, text };
}

const app = new Hono();

app.post("/api/start", async (c) => {
  const { sessionId } = await c.req.json<{ sessionId: string }>();
  const history = freshSession(sessionId);
  history.push({ role: "user", content: "Begin the adventure." });
  try {
    const { text } = await narrate(history);
    return c.json({ text });
  } catch (e: any) {
    return c.json({ error: String(e.message ?? e) }, 502);
  }
});

app.post("/api/type", async (c) => {
  const { sessionId, text } = await c.req.json<{ sessionId: string; text: string }>();
  const command = (text ?? "").slice(0, 500); // bound the prompt fed to the LLM
  const history = getSession(sessionId);
  history.push({ role: "user", content: command });
  try {
    const { text } = await narrate(history);
    return c.json({ text });
  } catch (e: any) {
    return c.json({ error: String(e.message ?? e) }, 502);
  }
});

app.post("/api/tts", async (c) => {
  const { text } = await c.req.json<{ text: string }>();
  // cap length so a public endpoint can't run up a bill or stall a local synth
  const speakable = (text ?? "").replace(/[*_#`>]/g, "").trim().slice(0, 800);
  if (!speakable) return c.text("empty", 400);

  if (TTS_BACKEND === "kokoro") {
    // Local Kokoro-82M GPU sidecar (kokoro_sidecar.py). One forward pass, no
    // streaming, but RTF ~0.04 so a 5s reply lands in ~200ms warm — comfortably
    // under conversational latency. Voice is configurable per-request.
    const r = await fetch(`${KOKORO_URL}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: speakable, voice: KOKORO_VOICE, speed: KOKORO_SPEED }),
    });
    if (!r.ok) return c.text(`kokoro ${r.status}: ${await r.text()}`, 502);
    return new Response(r.body, { headers: { "Content-Type": "audio/wav" } });
  }

  if (TTS_BACKEND === "supertonic") {
    // Local supertonic sidecar (supertonic serve). Buffers the whole clip — no
    // streaming — but at steps=5 short replies land in ~500ms warm.
    const r = await fetch(`${SUPERTONIC_URL}/v1/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: speakable,
        voice: SUPERTONIC_VOICE,
        lang: "en",
        steps: SUPERTONIC_STEPS,
        response_format: "wav",
      }),
    });
    if (!r.ok) return c.text(`supertonic ${r.status}: ${await r.text()}`, 502);
    return new Response(r.body, { headers: { "Content-Type": "audio/wav" } });
  }

  const r = await fetch("https://api.cartesia.ai/tts/bytes", {
    method: "POST",
    headers: {
      "X-API-Key": CARTESIA_API_KEY,
      "Cartesia-Version": CARTESIA_VERSION,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model_id: CARTESIA_MODEL,
      transcript: speakable,
      voice: { mode: "id", id: VOICE_ID },
      output_format: { container: "wav", encoding: "pcm_s16le", sample_rate: 44100 },
    }),
  });
  if (!r.ok) return c.text(`cartesia ${r.status}: ${await r.text()}`, 502);
  return new Response(r.body, { headers: { "Content-Type": "audio/wav" } });
});

// --- hands-free: stream mic PCM, server-side Silero VAD segments utterances ---
// Protocol:
//   client -> server:  {"type":"hello","sessionId":...}   (first, ties to session)
//                      <binary>  Int16LE PCM, 16kHz mono   (continuous mic)
//                      {"type":"resume"}                   (after TTS playback ends)
//   server -> client:  {"type":"ready"} | {"type":"speech_start"}
//                      {"type":"narration","text":...} | {"type":"error",...}
app.get("/ws", upgradeWebSocket(() => {
  let vad: SileroVAD | null = null;
  const vadReady = SileroVAD.create().then((v) => { vad = v; });
  let sessionId = "";
  let busy = false;                 // true from utterance-end until client "resume"
  let leftover = new Float32Array(0);
  const queue: Float32Array[] = [];
  let pumping = false;
  let ws: any = null;
  const send = (o: any) => ws?.send(JSON.stringify(o));

  async function pump() {
    if (pumping) return;
    pumping = true;
    await vadReady;
    while (queue.length) {
      const frame = queue.shift()!;
      if (busy) continue;           // ignore mic while a turn is being handled (anti-echo)
      const ev = await vad!.process(frame);
      if (ev?.type === "start") send({ type: "speech_start" });
      else if (ev?.type === "end") { busy = true; send({ type: "thinking" }); await handleUtterance(ev.audio); }
    }
    pumping = false;
  }

  async function handleUtterance(audio: Float32Array) {
    try {
      const wav = encodeWav(audio);
      if (DEBUG_AUDIO) {
        const path = `${DEBUG_AUDIO}/utt_${Date.now()}.wav`;
        Bun.write(path, wav)
          .then(() => console.log(`[debug-audio] wrote ${path} (${(audio.length / 16000).toFixed(2)}s)`))
          .catch((e) => console.warn(`[debug-audio] write failed: ${e.message}`));
      }
      const b64 = Buffer.from(wav).toString("base64");
      const history = getSession(sessionId);
      history.push({ role: "user", content: [{ type: "input_audio", input_audio: { data: b64, format: "wav" } }] });
      const { heard, text } = await narrate(history);
      send({ type: "narration", text, ...(heard !== undefined && { heard }) });
    } catch (e: any) {
      busy = false;                 // let the player try again
      send({ type: "error", message: String(e.message ?? e) });
    }
  }

  return {
    onOpen(_e: any, w: any) { ws = w; send({ type: "ready" }); },
    onMessage(e: any, w: any) {
      ws = w;
      const d = e.data;
      if (typeof d === "string") {
        const m = JSON.parse(d);
        if (m.type === "hello") {
          sessionId = m.sessionId;
          if (m.sampleRate && m.sampleRate !== 16000)
            console.warn(`[mic] client AudioContext at ${m.sampleRate} Hz (expected 16000) — audio will be mis-pitched!`);
        }
        else if (m.type === "resume") { busy = false; queue.length = 0; leftover = new Float32Array(0); vad?.reset(); }
        return;
      }
      if (busy) return;             // drop inbound audio while speaking/thinking
      // binary Int16LE PCM -> Float32, reframed to exact 512-sample windows
      const bytes = d instanceof ArrayBuffer ? new Uint8Array(d.slice(0)) : Uint8Array.from(d as Uint8Array);
      const i16 = new Int16Array(bytes.buffer, 0, bytes.byteLength >> 1);
      const incoming = new Float32Array(i16.length);
      for (let k = 0; k < i16.length; k++) incoming[k] = i16[k] / 32768;
      const merged = new Float32Array(leftover.length + incoming.length);
      merged.set(leftover); merged.set(incoming, leftover.length);
      let off = 0;
      for (; off + FRAME_SAMPLES <= merged.length; off += FRAME_SAMPLES) queue.push(merged.slice(off, off + FRAME_SAMPLES));
      leftover = merged.slice(off);
      pump();
    },
    onClose() { queue.length = 0; },
  };
}));

app.get("/", async (c) => {
  return c.html(await Bun.file(new URL("./public/index.html", import.meta.url)).text());
});

console.log(`Zork voice demo on http://localhost:${PORT}  (voice ${VOICE_ID}, llama ${LLAMA_URL})`);
export default { port: PORT, fetch: app.fetch, websocket };
