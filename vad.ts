// vad.ts — streaming Silero VAD (server-side). The "tricky logic", isolated so it
// can be QA'd headlessly against WAV files (see vad_qa.ts) with no browser.
//
// Silero v5 wants 16kHz mono, exactly 512 samples per frame (~32ms). We keep the
// LSTM state running across frames and apply the standard hysteresis state machine
// (positive/negative thresholds + redemption + min-speech + pre-roll padding).
import * as ort from "onnxruntime-node";

export const SAMPLE_RATE = 16000;
export const FRAME_SAMPLES = 512;
// Silero v5 wants each frame prepended with the last 64 samples of the previous
// frame (model input is [1, 576] at 16kHz, like the official wrapper). Feeding a
// bare [1, 512] "works" but probabilities collapse mid-speech, ending turns early.
const CONTEXT_SAMPLES = 64;

export interface VADOptions {
  positiveSpeechThreshold: number; // enter "speaking" at/above this prob
  negativeSpeechThreshold: number; // candidate end below this prob
  redemptionFrames: number;        // consecutive below-neg frames that end an utterance
  minSpeechFrames: number;         // shorter than this => discard as a misfire
  preSpeechPadFrames: number;      // frames of audio to prepend before speech onset
  maxSpeechFrames: number;         // force-end a runaway utterance (bounds memory)
}

export const DEFAULTS: VADOptions = {
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.35,
  redemptionFrames: 15,     // ~480ms of quiet ends the turn
  minSpeechFrames: 3,       // ~96ms minimum, filters clicks/coughs
  preSpeechPadFrames: 3,    // ~96ms of lead-in so onsets aren't clipped
  maxSpeechFrames: 1500,    // ~48s hard cap on a single utterance
};

export type VADEvent =
  | { type: "start" }
  | { type: "end"; audio: Float32Array; frames: number }
  | { type: "misfire"; frames: number };

let modelPromise: Promise<ort.InferenceSession> | null = null;
export function loadModel(path = new URL("./models/silero_vad_v5.onnx", import.meta.url).pathname) {
  if (!modelPromise) modelPromise = ort.InferenceSession.create(path);
  return modelPromise;
}

export class SileroVAD {
  private opt: VADOptions;
  private session: ort.InferenceSession;
  private state = new Float32Array(2 * 1 * 128);
  private readonly sr = new ort.Tensor("int64", BigInt64Array.from([BigInt(SAMPLE_RATE)]), []);

  private speaking = false;
  private redemption = 0;
  private context = new Float32Array(CONTEXT_SAMPLES); // tail of the previous frame
  private speechFrames: Float32Array[] = []; // frames kept for the current utterance
  private preRoll: Float32Array[] = [];       // ring buffer of recent pre-speech frames

  constructor(session: ort.InferenceSession, opt: Partial<VADOptions> = {}) {
    this.session = session;
    this.opt = { ...DEFAULTS, ...opt };
  }

  static async create(opt: Partial<VADOptions> = {}) {
    return new SileroVAD(await loadModel(), opt);
  }

  private async prob(frame: Float32Array): Promise<number> {
    // Build the [context ++ frame] input the v5 model expects. Copying into a
    // fresh buffer also sidesteps ort's byteOffset-ignoring tensor reads.
    const data = new Float32Array(CONTEXT_SAMPLES + FRAME_SAMPLES);
    data.set(this.context);
    data.set(frame, CONTEXT_SAMPLES);
    this.context.set(frame.subarray(FRAME_SAMPLES - CONTEXT_SAMPLES));
    const input = new ort.Tensor("float32", data, [1, CONTEXT_SAMPLES + FRAME_SAMPLES]);
    const state = new ort.Tensor("float32", this.state, [2, 1, 128]);
    const out = await this.session.run({ input, state, sr: this.sr });
    // onnxruntime-node reuses the output buffer on the next run(), so we must
    // copy the recurrent state out before feeding it back — otherwise it gets
    // aliased/poisoned and every subsequent probability collapses to ~0.
    this.state = Float32Array.from(out.stateN.data as Float32Array);
    return (out.output.data as Float32Array)[0];
  }

  /** Reset to a clean listening state (fresh LSTM state, no in-progress utterance). */
  reset() {
    this.state = new Float32Array(2 * 1 * 128);
    this.context = new Float32Array(CONTEXT_SAMPLES);
    this.speaking = false;
    this.redemption = 0;
    this.speechFrames = [];
    this.preRoll = [];
  }

  get isSpeaking() { return this.speaking; }

  /** Feed exactly one 512-sample frame; returns an event when an utterance starts/ends. */
  async process(frame: Float32Array): Promise<VADEvent | null> {
    const p = await this.prob(frame);

    if (!this.speaking) {
      // keep a short rolling pre-roll so we don't clip the start of speech
      this.preRoll.push(frame);
      if (this.preRoll.length > this.opt.preSpeechPadFrames) this.preRoll.shift();

      if (p >= this.opt.positiveSpeechThreshold) {
        this.speaking = true;
        this.redemption = 0;
        this.speechFrames = [...this.preRoll, frame];
        this.preRoll = [];
        return { type: "start" };
      }
      return null;
    }

    // currently speaking
    this.speechFrames.push(frame);
    if (this.speechFrames.length >= this.opt.maxSpeechFrames) return this.endUtterance();
    if (p < this.opt.negativeSpeechThreshold) {
      this.redemption++;
      if (this.redemption >= this.opt.redemptionFrames) {
        return this.endUtterance();
      }
    } else {
      this.redemption = 0;
    }
    return null;
  }

  private endUtterance(): VADEvent {
    // drop the trailing redemption frames (they're silence) from the utterance
    const kept = this.speechFrames.slice(0, Math.max(0, this.speechFrames.length - this.redemption));
    const nFrames = kept.length;
    this.speaking = false;
    this.redemption = 0;
    this.speechFrames = [];
    if (nFrames < this.opt.minSpeechFrames) return { type: "misfire", frames: nFrames };
    const audio = new Float32Array(nFrames * FRAME_SAMPLES);
    kept.forEach((f, i) => audio.set(f, i * FRAME_SAMPLES));
    return { type: "end", audio, frames: nFrames };
  }

  /** Flush any in-progress utterance (e.g. stream closed). */
  flush(): VADEvent | null {
    if (!this.speaking) return null;
    // treat remaining frames as the utterance, nothing to trim
    this.redemption = 0;
    return this.endUtterance();
  }
}

/** Encode mono Float32 [-1,1] @16kHz to a 16-bit PCM WAV (what Gemma accepts). */
export function encodeWav(samples: Float32Array, sampleRate = SAMPLE_RATE): Uint8Array {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const wr = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  wr(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); wr(8, "WAVE");
  wr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  wr(36, "data"); v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return new Uint8Array(buf);
}
