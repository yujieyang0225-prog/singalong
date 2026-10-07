#!/usr/bin/env python3
"""Visualise a training run of train_chroma.py: curves, example plot and listenable audio.

Writes into the run folder:
  curves.png           loss + val F1 / exact over steps, baseline as dashed lines
  example_<id>.png     input bands / target chord pitch classes / SNN output rate
  <id>_input.wav       the singer alone (input)
  <id>_snn.wav         singer + SNN output played as tones (amplitude = spike rate)
  <id>_target.wav      singer + MIDI target chords played the same way
  report.html          everything on one page, open it in a browser

Two steps, because the SDK container has no ffmpeg and the host has no Talamo:
  1. host (decode the audio of the validation singers to WAV, once):
       PYTHONPATH=.pydeps python3 jan/train/visualise_run.py --prepare-audio
  2. container (from the repo root):
       python3 jan/train/visualise_run.py data/runs/<run> [--rec 0] [--start 9 --secs 30]
"""
import argparse
import csv
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "damp100"
WAVS = DATA / "wav"
DAMP = REPO / "data" / "DAMP-S-AG" / "vocals" / "amazing_grace"
SR, HOP, FPS = 16000, 256, 62.5
NAMES = "C C# D D# E F F# G G# A A# B".split()


# ------------------------------------------------------------------- audio --
def write_wav(path, y):
    y = np.clip(y, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((y * 32767).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32767


def prepare_audio():
    """Host only: decode val + test singers to 16 kHz WAV (needs ffmpeg)."""
    WAVS.mkdir(exist_ok=True)
    rows = [r for r in csv.DictReader(open(DATA / "manifest.csv")) if r["split"] in ("val", "test")]
    for r in rows:
        out = WAVS / f"{r['id']}.wav"
        if not out.exists():
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(DAMP / f"{r['id']}.m4a"), "-ac", "1",
                                  "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
            write_wav(out, np.frombuffer(raw, np.float32))
        print("ok", out.name)


def render_chroma(act, n_samples, octave_midi=60):
    """Play 12 pitch classes as soft violin-like tones; act [frames, 12] in 0..1 (amplitude per frame)."""
    t = np.arange(n_samples) / SR
    fr = np.arange(len(act)) * HOP + HOP // 2
    k = np.ones(5) / 5                                      # smooth the envelope a little (80 ms)
    out = np.zeros(n_samples)
    for pc in range(12):
        a = np.convolve(np.pad(act[:, pc], 2, mode="edge"), k, "valid")
        if a.max() < 0.05:
            continue
        env = np.interp(np.arange(n_samples), fr, a)
        f0 = 440 * 2 ** ((octave_midi + pc - 69) / 12)
        vib = 1 + 0.003 * np.sin(2 * np.pi * 5.5 * t)       # light vibrato
        ph = 2 * np.pi * np.cumsum(f0 * vib) / SR
        tone = sum(np.sin(h * ph) / h for h in range(1, 6))
        out += env * tone
    return out / max(np.abs(out).max(), 1e-6) * 0.5


def mix(voice, acc):
    v = voice / max(np.abs(voice).max(), 1e-6) * 0.7
    return v + acc[:len(v)]


# ------------------------------------------------------------------- plots --
def plot_curves(run, metrics, path):
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(run / "log.csv")))
    st = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    loss = [float(r["loss"]) for r in rows]
    if not all(np.isnan(loss)):
        ax[0].plot(st, loss, "o-", color="#4a7ab5")
    else:
        ax[0].text(0.5, 0.5, "loss not logged by Talamo fit", ha="center", transform=ax[0].transAxes)
    ax[0].set_title("training loss (BCE, one batch)", loc="left", fontsize=10)
    ax[0].set_xlabel("step")
    ax[1].plot(st, [float(r["val_f1"]) for r in rows], "o-", color="#e8590c", label="val F1 (chroma)")
    ax[1].plot(st, [float(r["val_exact"]) for r in rows], "o-", color="#2b8a3e", label="val exact chord")
    b = metrics["baseline"]
    ax[1].axhline(b["f1"], color="#e8590c", ls="--", lw=1, label=f"baseline F1 (always {b['chord']})")
    ax[1].axhline(b["exact"], color="#2b8a3e", ls="--", lw=1, label="baseline exact")
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("step / epoch"); ax[1].legend(fontsize=8)
    ax[1].set_title("validation (8 unseen singers)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def plot_example(rec, rate, t0, n, path):
    import matplotlib.pyplot as plt
    ext = [t0 / FPS, (t0 + n) / FPS]
    fig, ax = plt.subplots(4, 1, figsize=(14, 8.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.3, 1, 1, 0.7]))
    ax[0].imshow(rec["X"][t0:t0 + n].T, origin="lower", aspect="auto", cmap="Blues", extent=ext + [40, 88])
    ax[0].set_title(f"input: 48 bands, singer {rec['id']}", loc="left", fontsize=9)
    ax[1].imshow(rec["Y"][t0:t0 + n].T, origin="lower", aspect="auto", cmap="Greys", extent=ext + [-.5, 11.5])
    ax[1].set_title("target: MIDI chord pitch classes", loc="left", fontsize=9)
    ax[2].imshow(rate.T, origin="lower", aspect="auto", cmap="Oranges", vmin=0, vmax=1, extent=ext + [-.5, 11.5])
    ax[2].set_title("SNN output: spike rate per chroma neuron (causal)", loc="left", fontsize=9)
    for a in ax[1:3]:
        a.set_yticks(range(12), NAMES, fontsize=7)
    y, p = rec["Y"][t0:t0 + n], rate >= 0.5
    tt = np.arange(t0, t0 + n) / FPS
    ax[3].plot(tt, (p == y).mean(1), color="#888", lw=0.8, label="pitch classes correct (of 12)")
    ax[3].plot(tt, (p == y).all(1).astype(float), color="#2b8a3e", lw=1.2, label="exact chord")
    ax[3].set_ylim(-0.05, 1.05); ax[3].legend(fontsize=7, loc="lower right")
    ax[3].set_title("frame-wise correctness (threshold 0.5)", loc="left", fontsize=9)
    ax[3].set_xlabel("time in song (s)")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", type=Path)
    ap.add_argument("--prepare-audio", action="store_true")
    ap.add_argument("--rec", type=int, default=0, help="index of the validation singer")
    ap.add_argument("--start", type=float, default=9.0, help="seconds into the song")
    ap.add_argument("--secs", type=float, default=30.0)
    ap.add_argument("--quantized", action="store_true", help="use trainable_params_quantized.json (Talamo runs)")
    a = ap.parse_args()
    if a.prepare_audio:
        return prepare_audio()

    import matplotlib
    matplotlib.use("Agg")
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    if not (run_arg := a.run).joinpath("trainable_params.json").exists() and \
            not (REPO / run_arg / "trainable_params.json").exists():
        from train_chroma import ChromaSNN, load_split

    run = a.run if a.run.is_absolute() else REPO / a.run
    metrics = json.load(open(run / "metrics.json"))
    plot_curves(run, metrics, run / "curves.png")

    t0, n = int(a.start * FPS), int(a.secs * FPS)
    if (run / "trainable_params.json").exists():          # Talamo pipeline run (jan/singalong)
        sys.path.insert(0, str(REPO / "jan" / "singalong"))
        from singalong import data as sdata, model as smodel
        rec = sdata.load_recordings("val")[a.rec]
        ar = metrics["args"]
        pipe, _ = smodel.build_pipeline(ar["hidden"], recurrent=not ar.get("feedforward", False),
                                        enc_threshold=ar.get("enc_threshold", 120.0))
        if a.quantized:
            pipe.quantize(quantizers=smodel.make_quantizers(pipe))
        pipe.trainable_params.load(str(run / ("trainable_params_quantized.json" if a.quantized
                                              else "trainable_params.json")))
        rate = smodel.predict(pipe, torch.tensor(rec["X"][None, t0:t0 + n]))[0].numpy()
    else:                                                  # old run of train_chroma.py
        ref = dict(np.load(DATA / "reference.npz"))
        rec = load_split("val", ref)[a.rec]
        model = ChromaSNN(metrics["args"]["hidden"])
        model.load_state_dict(torch.load(run / "model.pt"))
        model.eval()
        with torch.no_grad():
            rate = model(torch.tensor(rec["X"][None, t0:t0 + n]))[0][0].numpy()
    plot_example(rec, rate, t0, n, run / f"example_{rec['id']}.png")

    audio_html = "<p><i>No audio: run <code>--prepare-audio</code> on the host first.</i></p>"
    wav = WAVS / f"{rec['id']}.wav"
    if wav.exists():
        voice = read_wav(wav)[t0 * HOP:(t0 + n) * HOP]
        write_wav(run / f"{rec['id']}_input.wav", voice / max(np.abs(voice).max(), 1e-6) * 0.8)
        write_wav(run / f"{rec['id']}_snn.wav", mix(voice, render_chroma(rate, len(voice))))
        write_wav(run / f"{rec['id']}_target.wav", mix(voice, render_chroma(rec["Y"][t0:t0 + n], len(voice))))
        audio_html = "".join(
            f"<p><b>{lab}</b><br><audio controls src='{rec['id']}_{k}.wav'></audio></p>"
            for k, lab in [("input", "🎤 Input: singer alone"),
                           ("snn", "🎻 Singer + SNN output (tones follow the output neurons' spike rate)"),
                           ("target", "✅ Singer + target chords from the MIDI")])

    b, best = metrics["baseline"], metrics["best"]
    html = f"""<!doctype html><meta charset="utf-8"><title>{run.name}</title>
<body style="font-family:sans-serif;max-width:1450px;margin:auto">
<h2>Run {run.name}</h2>
<p>{json.dumps(metrics['args'])}</p>
<table border=1 cellpadding=5 style="border-collapse:collapse">
<tr><th></th><th>val F1</th><th>val exact chord</th></tr>
<tr><td>baseline (always {b['chord']})</td><td>{b['f1']}</td><td>{b['exact']}</td></tr>
<tr><td>SNN best (step {best['step']})</td><td>{best['f1']}</td><td>{best['exact']}</td></tr>
{"<tr><td>SNN quantized int8 (chip)</td><td>%s</td><td>%s</td></tr>" % (metrics['quantized']['f1'], metrics['quantized']['exact']) if 'quantized' in metrics else ""}</table>
<h3>Training</h3><img src="curves.png" width="100%">
<h3>Example: {rec['id']}, {a.start:.0f}–{a.start + a.secs:.0f} s</h3>{audio_html}
<img src="example_{rec['id']}.png" width="100%">
</body>"""
    (run / "report.html").write_text(html)
    print(f"wrote {run / 'report.html'}")


if __name__ == "__main__":
    main()
