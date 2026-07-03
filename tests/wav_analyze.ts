// Analyze a dumped utterance WAV: report per-segment RMS energy to see whether
// VAD cut mid-word (energy at tail) or fired after a clean pause (silent tail).

function rms(buf: Float32Array): number {
  let s = 0;
  for (const x of buf) s += x * x;
  return Math.sqrt(s / buf.length);
}

for (const path of process.argv.slice(2)) {
  const ab = await Bun.file(path).arrayBuffer();
  const v = new DataView(ab);
  const n = (ab.byteLength - 44) / 2;
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) samples[i] = v.getInt16(44 + i * 2, true) / 32768;

  const dur = (n / 16000).toFixed(2);
  // split into 50ms windows
  const win = Math.floor(16000 * 0.05);
  const segs: { t: number; rms: number }[] = [];
  for (let i = 0; i + win <= n; i += win) {
    segs.push({ t: i / 16000, rms: rms(samples.slice(i, i + win)) });
  }
  const bars = segs.map((s) => {
    const lvl = Math.min(40, Math.round(s.rms * 800));
    return `${s.t.toFixed(2).padStart(5)}s ${"█".repeat(lvl).padEnd(40)} ${s.rms.toFixed(3)}`;
  });
  // tail energy: last 100ms
  const tail = samples.slice(Math.max(0, n - 1600));
  const tailRms = rms(tail);
  const verdict = tailRms > 0.01 ? "ENERGY at tail → cut MID-WORD (speech ongoing)" : "silent tail → fired after a pause";
  console.log(`\n=== ${path}  (${dur}s, ${n} samples) ===`);
  console.log(bars.join("\n"));
  console.log(`tail RMS (last 100ms): ${tailRms.toFixed(4)}  → ${verdict}`);
}
