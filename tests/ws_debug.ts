// Test the WS audio path with DEBUG_TRANSCRIPT: feed a command WAV as PCM,
// expect a {type:"narration", text, heard} back.
import { SAMPLE_RATE } from "../vad";

const url = process.argv[2] ?? "ws://localhost:3102/ws";
const wavPath = process.argv[3] ?? "supertonic_test/out/cmd_go_north.wav";
const sessionId = "wsdbg-" + Date.now();

// Decode the WAV to PCM Float32, then re-encode as Int16LE chunks to stream.
const wav = await Bun.file(wavPath).arrayBuffer();
const view = new DataView(wav);
// skip 44-byte header
const samples = new Float32Array((wav.byteLength - 44) / 2);
for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(44 + i * 2, true) / 32768;

const ws = new WebSocket(url);
ws.binaryType = "arraybuffer";
let gotNarration = false;
ws.onopen = () => {
  ws.send(JSON.stringify({ type: "hello", sessionId }));
  // stream the audio in chunks (e.g. 4096 samples = 256ms)
  const CHUNK = 4096;
  for (let i = 0; i < samples.length; i += CHUNK) {
    const slice = samples.slice(i, i + CHUNK);
    const i16 = new Int16Array(slice.length);
    for (let k = 0; k < slice.length; k++) i16[k] = Math.max(-1, Math.min(1, slice[k])) * (slice[k] < 0 ? 0x8000 : 0x7fff);
    ws.send(i16.buffer);
  }
  // send ~1.5s of trailing silence so VAD fires the end-of-utterance
  const SILENCE = Math.floor(SAMPLE_RATE * 1.5);
  const z = new Int16Array(SILENCE);
  ws.send(z.buffer);
  console.log(`sent ${samples.length} samples (${(samples.length / SAMPLE_RATE).toFixed(2)}s) + 1.5s silence`);
};
ws.onmessage = (e) => {
  const m = JSON.parse(e.data as string);
  console.log("<<", JSON.stringify(m));
  if (m.type === "narration") {
    gotNarration = true;
    console.log(`\n=== RESULT ===`);
    console.log(`heard: ${m.heard ?? "(not present)"}`);
    console.log(`text:  ${m.text}`);
    ws.close();
  }
  if (m.type === "error") { console.error("server error:", m.message); ws.close(); }
};
ws.onclose = () => { if (!gotNarration) console.log("closed without narration"); process.exit(0); };
setTimeout(() => { console.log("TIMEOUT"); process.exit(1); }, 30000);
