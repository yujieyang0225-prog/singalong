#!/usr/bin/env python3
"""Expression facts from our song recordings (beyond "which note"): tuning, loudness, vibrato.

Per recording (first 60 s, 16 kHz, YIN with a 32 ms window, 16 ms hop, confident voiced frames only):
  * tuning offset  = circular mean of each frame's deviation from the nearest equal-tempered semitone (A4 = 440 Hz)
  * drift          = tuning offset in the last 20 s minus the first 20 s
  * note spread    = median |frame deviation - local reference| (local reference = circular median over ±2 s)
  * loudness range = 95th - 5th percentile of frame level (dBFS) over voiced frames
  * loudness speed = median |level change| per 100 ms within voiced stretches
  * vibrato        = share of stable notes (>= 600 ms) whose pitch wobbles 4.5-8 Hz (clear spectral peak) with >= ±15 cents, plus median rate/extent

Run from repo root:  PYTHONPATH=.pydeps:jan/scripts python3 jan/scripts/analyze_expression.py
"""
import numpy as np

import frontend_check as fc

SR, WIN, HOP = 16000, 512, 256
FPS = SR / HOP  # 62.5 frames per second


def circ_mean_cents(c):
    a = np.exp(2j * np.pi * np.asarray(c) / 100)
    return 100 * np.angle(a.mean()) / (2 * np.pi)


def wrap(c):
    return (np.asarray(c) + 50) % 100 - 50


def analyse(path):
    x = fc.load(path, SR)[: SR * 60]
    f0, conf = fc.yin(x, SR, WIN, HOP)
    n = len(f0)
    lvl = 10 * np.log10(np.mean(np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP][:n] ** 2, axis=1) + 1e-12)
    voiced = conf & (lvl > np.percentile(lvl, 95) - 40)
    midi = 69 + 12 * np.log2(f0 / 440)
    dev = wrap(100 * (midi - np.round(midi)))  # cents from nearest ET semitone
    t = np.arange(n) / FPS
    v = np.where(voiced)[0]
    if len(v) < 200:
        return None
    off = circ_mean_cents(dev[v])
    first, last = v[t[v] < 20], v[t[v] > t[v[-1]] - 20]
    drift = wrap(circ_mean_cents(dev[last]) - circ_mean_cents(dev[first]))
    ref = np.array([circ_mean_cents(dev[v[(t[v] > t[i] - 2) & (t[v] < t[i] + 2)]]) for i in v])
    spread = np.median(np.abs(wrap(dev[v] - ref)))
    lr = np.percentile(lvl[v], 95) - np.percentile(lvl[v], 5)
    k = int(0.1 * FPS)
    both = voiced[:-k] & voiced[k:]
    lspeed = np.median(np.abs(lvl[k:] - lvl[:-k])[both]) if both.any() else np.nan
    # stable notes: runs of voiced frames staying within ±70 cents of their median semitone for >= 400 ms
    vib, rates, exts, run = 0, [], [], []
    notes = 0
    for i in range(n + 1):
        ok = i < n and voiced[i] and (not run or abs(midi[i] - np.median(midi[run])) < 0.7)
        if ok:
            run.append(i)
            continue
        if len(run) >= 0.6 * FPS:
            notes += 1
            c = 100 * (midi[run] - np.median(midi[run]))
            c = c - np.polyval(np.polyfit(np.arange(len(c)), c, 1), np.arange(len(c)))  # remove slow glide
            spec = np.abs(np.fft.rfft(c * np.hanning(len(c)), 512))
            fr = np.fft.rfftfreq(512, 1 / FPS)
            band = (fr >= 3.5) & (fr <= 9)
            peak = fr[band][spec[band].argmax()]
            ext = (np.percentile(c, 95) - np.percentile(c, 5)) / 2
            if 4.5 <= peak <= 8 and ext >= 15 and spec[band].max() > 3 * np.median(spec[(fr > 1) & (fr < 15)]):
                vib += 1
                rates.append(peak)
                exts.append(ext)
        run = [i] if (i < n and voiced[i]) else []
    return dict(off=off, drift=drift, spread=spread, lr=lr, lspeed=lspeed, notes=notes, vib=vib,
                rate=np.median(rates) if rates else np.nan, ext=np.median(exts) if exts else np.nan,
                voiced_s=len(v) / FPS, t=t, lvl=lvl, dev=dev, voiced=voiced, midi=midi)


KIND = {  # from 08 Songs/Song sources (EN).md
    "521217": "voice a cappella", "624721": "voice a cappella (maybe processed)", "574342": "voice + ukulele",
    "371903": "harmonica", "sung_with": "voice (sung, full lyrics)", "plain": "plain melody (instrument)",
    "little_star.webm": "voice (video)", "(vocal)": "voice a cappella (male)", "575367": "children + clapping",
    "Akustik": "guitar", "Frei_martinho": "voice (Portuguese)", "Irm": "voice (Portuguese)", "canon": "melody as round",
    "sujet": "plain melody", "90077": "shellac 1928, voice + quartet", "wienstadt": "shellac 1917, voice + piano",
    "Sig_m": "Danish version", "seifert": "choir, synthesised"}


def kind(f):
    return next((v for k, v in KIND.items() if k in f.name), "?")


def main():
    files = sorted(p for p in fc.SONGS.glob("*/audio/*") if p.suffix in (".wav", ".ogg", ".flac", ".mp3", ".webm"))
    print("| Recording | type | voiced s | tuning offset (cents vs A440) | drift last−first 20 s | note spread | loudness range | loudness change /100 ms | stable notes | with vibrato | vibrato rate / extent |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in files:
        r = analyse(f)
        if r is None:
            print(f"| {f.parent.parent.name[:2]} {f.stem[:40]} | {kind(f)} | too little confident pitch |||||||||")
            continue
        vr = f"{r['rate']:.1f} Hz / ±{r['ext']:.0f} c" if r["vib"] else "—"
        print(f"| {f.parent.parent.name[:2]} {f.stem[:40]} | {kind(f)} | {r['voiced_s']:.0f} | {r['off']:+.0f} | {r['drift']:+.0f} | {r['spread']:.0f} c | "
              f"{r['lr']:.0f} dB | {r['lspeed']:.1f} dB | {r['notes']} | {r['vib']} | {vr} |")




def fft_tuning(x, win=1024, hop=256, sr=SR):
    """Tuning offset as the chip could compute it: per frame, take the 3 strongest peaks of the 1024-pt FFT
    (100 Hz - 2 kHz), refine each by parabolic interpolation on log magnitude, express as cents from the
    nearest equal-tempered semitone, and average (circularly, weighted by peak level)."""
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop] * np.hanning(win)
    mag = np.log(np.abs(np.fft.rfft(frames, win)) + 1e-9)
    lo, hi = int(100 * win / sr), int(2000 * win / sr)
    out = np.full(len(mag), np.nan)
    for i, m in enumerate(mag):
        seg = m[lo:hi]
        pk = [k for k in range(1, len(seg) - 1) if seg[k] > seg[k - 1] and seg[k] >= seg[k + 1]]
        pk = sorted(pk, key=lambda k: -seg[k])[:3]
        if not pk or seg[pk[0]] < m.max() - 3:
            continue
        cs, ws = [], []
        for k in pk:
            a, b, c = seg[k - 1], seg[k], seg[k + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0
            f = (lo + k + d) * sr / win
            cs.append(wrap(1200 * np.log2(f / 440)))
            ws.append(np.exp(b))
        out[i] = 100 * np.angle(np.sum(np.array(ws) * np.exp(2j * np.pi * np.array(cs) / 100))) / (2 * np.pi)
    return out


def check_fft_tuning():
    """Does the cheap FFT-peak estimate agree with YIN on the global tuning offset of the voice recordings?"""
    print("\n| Voice recording | offset YIN | offset FFT peaks | difference |")
    print("|---|---|---|---|")
    for f in sorted(fc.SONGS.glob("*/audio/*")):
        if not kind(f).startswith("voice"):
            continue
        x = fc.load(f, SR)[: SR * 60]
        r = analyse(f)
        if r is None:
            continue
        ft = fft_tuning(x)
        n = min(len(ft), len(r["voiced"]))
        ok = r["voiced"][:n] & ~np.isnan(ft[:n])
        o2 = circ_mean_cents(ft[:n][ok])
        print(f"| {f.stem[:40]} | {r['off']:+.0f} | {o2:+.0f} | {wrap(o2 - r['off']):+.0f} |")


if __name__ == "__main__" and "--fft-tuning" in __import__("sys").argv:
    check_fft_tuning()


if __name__ == "__main__" and "--fft-tuning" not in __import__("sys").argv:
    main()
