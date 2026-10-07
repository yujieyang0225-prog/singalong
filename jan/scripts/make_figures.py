#!/usr/bin/env python3
"""Generate the explanatory figures for the vault (Hackathon/Attachments/Figures/*.png).

Data figures are computed from our own songs with the same code as the experiments
(jan/scripts/frontend_check.py, jan/scripts/analyze_song_harmony.py). Schematic figures draw the
current proposals from 01 Project/Key numbers.md; if a number there changes, edit it below and rerun.

Run from repo root:  PYTHONPATH=.pydeps:jan/scripts python3 jan/scripts/make_figures.py
"""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

import analyze_expression as ae  # noqa: E402
import analyze_song_harmony as ash  # noqa: E402
import frontend_check as fc  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "Hackathon" / "Attachments" / "Figures"
OUT.mkdir(parents=True, exist_ok=True)

# Reference palette (dataviz skill, light mode); validated: adjacent CVD ΔE >= 9.1
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#b9b8b2"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def title(fig, main, sub):
    """Title + wrapped subtitle, placed in inches from the top so they never collide."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.12 / h, main, ha="left", va="top", fontsize=14, weight="bold", color=INK)
    fig.text(0.012, 1 - 0.42 / h, textwrap.fill(sub, int(fig.get_figwidth() * 13.2)), ha="left", va="top",
             fontsize=9.5, color=INK2, linespacing=1.4)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", name)


def note_name(m):
    return f"{NOTE[m % 12]}{m // 12 - 1}"


# ---------------------------------------------------------------- 1. what the SNN sees
def fig_bands():
    fc.MIDI_LO, fc.N_BANDS = 40, 48  # D02 layout updated 2026-10-06 (82 Hz - 1.25 kHz)
    f = next(fc.SONGS.glob("02_Twinkle/audio/Twinkle_twinkle_little_star_(vocal).ogg"))
    sr, win, nfft, hop = fc.CONFIGS["16k win1024 (plan)"]
    x = fc.load(f, sr)[int(sr * 4.2): int(sr * 15.2)]  # the recording starts with 4 s of near-silence
    e = fc.bands(x, sr, win, nfft, hop)  # frames x 64, log2, 0 = loudest band, -13.3 = 40 dB down
    rms = np.sqrt((np.lib.stride_tricks.sliding_window_view(x, win)[::hop] ** 2).mean(axis=1))
    gate = rms > 0.1 * np.percentile(rms, 90)  # voicing gate (D02): silent frames -> all bands at floor
    e[~gate] = -13.3
    f0, conf = fc.yin(x, sr, win, hop)
    n = min(len(e), len(f0))
    e, f0, conf, gate = e[:n], f0[:n], conf[:n], gate[:n]
    mel = 69 + 12 * np.log2(f0 / 440)
    t = np.arange(n) * hop / sr

    fig = plt.figure(figsize=(11, 5.4))
    ax = fig.add_axes([0.12, 0.12, 0.75, 0.68])
    cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    im = ax.imshow((e.T + 13.3) * 3.01, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=40,
                   extent=[t[0], t[-1] + hop / sr, fc.MIDI_LO - 0.5, fc.MIDI_LO + fc.N_BANDS - 0.5], interpolation="nearest")
    v = conf & gate
    ax.scatter(t[v], mel[v], s=10, color=S[1], linewidths=0, label="sung melody (YIN pitch tracker, for reference)", zorder=3)
    ticks = [m for m in range(fc.MIDI_LO, fc.MIDI_LO + fc.N_BANDS) if m % 12 == 0]
    ax.set_yticks(ticks, [f"{note_name(m)}  {fc.hz(m):.0f} Hz" for m in ticks])
    ax.set_ylim(39.5, fc.MIDI_LO + fc.N_BANDS - 0.5)
    ax.axhspan(39.5, fc.MIDI_LO - 0.5, color="#efeee9", lw=0, zorder=0)
    if fc.MIDI_LO > 41:
        ax.text(t[-1] - 0.05, 41, "not in the input (below the bands): this singer's fundamental. The SNN must read it from the harmonics above",
                ha="right", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(59.5, color=INK2, lw=0.8)
    ax.text(t[0] + 0.1, 58.8, "below C4: neighbouring bands share FFT bins (blurred)", ha="left", va="top", fontsize=8.5,
            color=INK, bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec="none", alpha=0.85))
    ax.set_xlabel("time (s)  ·  one column = one 16 ms SNN frame  ·  white = silence (voicing gate)")
    ax.legend(loc="upper right", fontsize=9, markerscale=2.2, framealpha=0.9, edgecolor="none", facecolor=SURF)
    cax = fig.add_axes([0.89, 0.12, 0.015, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("level below loudest band (dB)", color=INK2)
    cb.set_ticks([0, 10, 20, 30, 40], labels=["−40", "−30", "−20", "−10", "0"])
    cb.outline.set_visible(False)
    title(fig, "What the SNN sees: 48 semitone bands (82 Hz – 1.25 kHz), one column every 16 ms",
          "\"Twinkle twinkle little star (vocal)\" from 08 Songs, 11 s · 16 kHz, 1024-pt FFT, 64 ms Hann window, level relative to the loudest band per frame. The dark stripes stacked above the melody are its harmonics (×2, ×3, ×4 …): they move in parallel, so a key change shifts the whole picture up or down.")
    save(fig, "frontend-semitone-bands.png")


# ---------------------------------------------------------------- 2. bin width vs semitone spacing
def fig_resolution():
    f = np.geomspace(80, 1100, 400)
    semi = f * (2 ** (1 / 12) - 1)
    fig = plt.figure(figsize=(9, 4.8))
    ax = fig.add_axes([0.09, 0.13, 0.86, 0.66])
    ax.axvspan(85, 180, color=S[0], alpha=0.07, lw=0)
    ax.axvspan(165, 520, color=S[1], alpha=0.07, lw=0)
    ax.text(124, 41, "typical male\nsinging F0", ha="center", va="top", fontsize=8.5, color=INK2)
    ax.text(300, 41, "female / child\nsinging F0", ha="center", va="top", fontsize=8.5, color=INK2)
    ax.plot(f, semi, color=S[0], lw=2)
    ax.text(1000, 1000 * (2 ** (1 / 12) - 1) + 1.5, "distance to the next semitone", ha="right", color=INK, fontsize=9.5)
    ax.axhline(15.625, color=S[1], lw=2)
    ax.text(84, 16.6, "FFT bin width at 16 kHz / 1024 pt = 15.6 Hz (our plan)", fontsize=9.5, color=INK)
    ax.axhline(7.8125, color=GRAY, lw=2)
    ax.text(84, 8.6, "8 kHz / 1024 pt = 7.8 Hz (old plan: sharper, but 128 ms window)", fontsize=9, color=INK2)
    ax.plot([262], [262 * (2 ** (1 / 12) - 1)], "o", ms=8, color=S[1], mec=SURF, mew=2, zorder=4)
    ax.annotate("crossover ≈ 262 Hz (C4):\nabove it every semitone gets its own bin", (262, 15.6), (420, 25),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xscale("log")
    ax.set_xticks([100, 131, 200, 262, 400, 523, 800, 1000], ["100", "131\nC3", "200", "262\nC4", "400", "523\nC5", "800", "1000"])
    ax.set_xlim(80, 1100)
    ax.set_ylim(0, 42)
    ax.set_xlabel("note frequency (Hz, log scale)")
    ax.set_ylabel("Hz")
    ax.grid(axis="y")
    title(fig, "Why the lowest octave is blurred at 16 kHz — and why it barely matters",
          "Below C4 two neighbouring semitones fall into one FFT bin. But a sung note's 2nd–4th harmonics lie above C4, so the pitch survives (98 % on low synthetic notes).")
    save(fig, "fft-bin-vs-semitone.png")


# ---------------------------------------------------------------- 3. front-end measurement
def fig_frontend_acc():
    # from 07 Experiments/2026-10-06 Front-end pitch check 16 kHz (real recordings, pitch class vs YIN)
    rows = [("16 kHz · 1024 window · 64 ms  (plan)", 83), ("8 kHz · 512→1024 · 64 ms  (old plan)", 83),
            ("16 kHz · 512 window · 32 ms  (low latency)", 78), ("8 kHz · 1024 window · 128 ms", 74)]
    fig = plt.figure(figsize=(9, 3.4))
    ax = fig.add_axes([0.36, 0.12, 0.58, 0.56])
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=S[0] if "plan)" in lab and "old" not in lab else GRAY, lw=0)
        ax.text(v + 1, yi, f"{v} %", va="center", fontsize=10, color=INK, weight="bold" if "(plan)" in lab else "normal")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100 %"])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    title(fig, "Pitch survives the front-end: 83 % on real recordings, before any learning",
          "Pitch-class agreement with a YIN reference, 18 song recordings (19,967 voiced frames), fixed harmonic-sum readout. Longer windows smear note changes.")
    save(fig, "frontend-accuracy.png")


# ---------------------------------------------------------------- 4. chords in our songs
def fig_chords():
    res = [ash.analyse(*s) for s in ash.SONGS]
    cats = ["I", "V", "IV", "vi", "ii"]
    fig = plt.figure(figsize=(10, 4.0))
    ax = fig.add_axes([0.17, 0.2, 0.79, 0.6])
    names = [r["name"].replace(" (Gerloff)", "") for r in res]
    for i, r in enumerate(res[::-1]):
        tot = sum(r["vocab"].values())
        shares = [sum(v for k, v in r["vocab"].items() if k.replace("7", "") == c) / tot for c in cats]
        shares.append(1 - sum(shares))
        left = 0
        for j, s in enumerate(shares):
            ax.barh(i, s, left=left, height=0.64, color=(S[j] if j < len(cats) else GRAY), edgecolor=SURF, lw=2)
            if s >= 0.08:
                ax.text(left + s / 2, i, f"{(cats + ['other'])[j]} {s:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if j in (0, 1) else INK)
            left += s
    ax.set_yticks(range(len(res)), names[::-1])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100 %"])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in S[:5] + [GRAY]]
    ax.legend(handles, ["I (tonic)", "V (dominant)", "IV (subdominant)", "vi", "ii", "other"], ncol=6, frameon=False,
              loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=9, handlelength=1, columnspacing=1.4)
    title(fig, "Our songs live on a few chords: I + IV + V cover 64 % of beats, + vi + ii 80 %",
          "Chord per beat in the 6 harmonised song MIDIs, relative to the key (7ths folded into triads). 99.7 % of melody notes are in the key's scale.")
    save(fig, "song-chord-share.png")


# ---------------------------------------------------------------- 5. chip budget
def fig_chip():
    role = ["hidden"] * 64 + ["out"] * 8 + ["free"] * 312
    colors = {"hidden": S[0], "out": S[2], "free": "#efeee9"}
    fig = plt.figure(figsize=(11, 5.8))
    ax = fig.add_axes([0.07, 0.31, 0.62, 0.52])
    for n, r in enumerate(role):
        plane, k = divmod(n, 32)
        ax.add_patch(Rectangle((k, 11 - plane), 0.82, 0.82, color=colors[r], lw=0))
    ax.set_xlim(-0.2, 32)
    ax.set_ylim(-0.2, 12)
    ax.set_yticks([11.41 - p for p in range(12)], [f"plane {p}" for p in range(12)], fontsize=8)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal")
    ax.text(32.3, 10.9, "← hidden layer (64), trainable leak", va="center", fontsize=9, color=INK)
    ax.text(32.3, 9.4, "← ~8 chord-evidence outputs\n    (read as spike counts by the selector)", va="center", fontsize=9, color=INK)
    ax.text(32.3, 4.4, "← ~310 neurons free: a 2nd layer,\n    wider hidden, or the optional\n    key latches (D05)", va="center", fontsize=9, color=INK2)
    bx = fig.add_axes([0.07, 0.08, 0.72, 0.1])
    segs = [("48 band inputs", 48, "#86b6ef", INK), ("32 recurrence (optional)", 32, S[0], "white"), ("48 free", 48, "#efeee9", INK)]
    left = 0
    for lab, w, c, tc in segs:
        bx.barh(0, w, left=left, color=c, edgecolor=SURF, lw=2, height=0.8)
        bx.text(left + w / 2, 0, lab, ha="center", va="center", fontsize=8.5, color=tc)
        left += w
    bx.set_xlim(0, 128)
    bx.set_xticks([0, 32, 64, 96, 128])
    bx.set_yticks([])
    for sp in ("left", "top", "right"):
        bx.spines[sp].set_visible(False)
    fig.text(0.07, 0.205, "Inputs of ONE hidden neuron (limit 128 = 4 sectors × 32): feedforward needs only the bands", fontsize=10, weight="bold", color=INK)
    title(fig, "The chip budget: the short-context SNN needs ~72 of 384 neurons",
          "Pulsar C1 = 12 planes × 32 integrate-and-fire neurons. Proposal 2026-10-06 (D04): long context lives in the chord selector on the RISC-V (D20), so no context plane. Feedforward vs recurrent still to be tested; the mapper decides the real placement.")
    save(fig, "chip-neuron-budget.png")

# ---------------------------------------------------------------- 6. neuron dynamics on C1
def c1(inputs, thr, bias=0, w_self=0, inhib=None):
    """Talamo C1 DigitalNeuron: U(t+1) = 0 if U(t) > thr else U(t) + Σws + bias; floor clamped at 0."""
    U, spk, u = np.zeros(len(inputs)), np.zeros(len(inputs), bool), 0.0
    for t in range(len(inputs)):
        U[t] = u
        if u > thr:
            spk[t], u = True, 0.0
        else:
            u = max(0.0, u + inputs[t] + bias + (w_self if t and spk[t - 1] else 0) + (inhib[t] if inhib is not None else 0))
    return U, spk


def fig_neurons():
    T, dt = 250, 0.016
    t = np.arange(T) * dt
    fig = plt.figure(figsize=(11, 7.2))
    axs = [fig.add_axes([0.13, 0.6 - i * 0.25, 0.83, 0.165]) for i in range(3)]
    # a) plain IF: evidence pulses, then silence while half-full -> potential stays
    x = np.zeros(T)
    x[[8, 20, 27, 41, 50, 62, 70, 85, 99, 104, 118, 131]] = 90
    Ua, sa = c1(x, thr=400)
    # b) slow integrator: sustained evidence crosses a high threshold once, pauses drain it (negative bias)
    xb = np.zeros(T)
    xb[10:95] = 13
    xb[150:185] = 13
    Ub, sb = c1(xb, thr=700, bias=-4)
    # c) two key latches, self-weight 220 > threshold 200, mutual inhibition 100 delivered twice
    #    (direct, and 1 step later via a relay): a firing C1 neuron ignores input in its reset step and a
    #    latch fires every 2nd step, so with direct inhibition only, two latches in antiphase never see each other.
    ea, eb = np.zeros(T), np.zeros(T)
    ea[20:32], eb[140:150] = 45, 130
    sA, sB, ua, ub = np.zeros(T, bool), np.zeros(T, bool), 0.0, 0.0
    for k in range(T):
        sA[k], sB[k] = ua > 200, ub > 200
        pa, pb = bool(k and sA[k - 1]), bool(k and sB[k - 1])
        ra, rb = bool(k > 1 and sA[k - 2]), bool(k > 1 and sB[k - 2])
        ua = 0.0 if sA[k] else max(0.0, ua + ea[k] + 220 * pa - 100 * (pb + rb))
        ub = 0.0 if sB[k] else max(0.0, ub + eb[k] + 220 * pb - 100 * (pa + ra))

    def trace(ax, U, s, thr, c):
        ax.plot(t, U, color=c, lw=1.6)
        ax.vlines(t[s], thr * 1.12, thr * 1.32, color=c, lw=1.6)
        ax.axhline(thr, color=INK2, lw=0.8)
        ax.text(0.02, thr * 1.0, "threshold ", va="bottom", ha="left", fontsize=8, color=INK2)
        ax.set_ylim(0, thr * 1.36)
        ax.set_yticks([])
        ax.set_ylabel("membrane U", fontsize=8.5)

    trace(axs[0], Ua, sa, 400, S[0])
    axs[0].axvspan(t[132], t[-1], color=GRAY, alpha=0.15, lw=0)
    axs[0].text(t[190], 400 * 0.62, "input stops → U just stays here", ha="center", fontsize=8.5, color=INK)
    trace(axs[1], Ub, sb, 700, S[0])
    for a, b in ((10, 95), (150, 185)):
        axs[1].axvspan(t[a], t[b], color=S[0], alpha=0.08, lw=0)
    axs[1].text(t[52], 700 * 0.12, "singing", ha="center", fontsize=8.5, color=INK2)
    axs[1].text(t[122], 700 * 0.62, "pause:\ndrains", ha="center", fontsize=8.5, color=INK2)
    ax = axs[2]
    for row, (spk, ev, c, lab) in enumerate(((sB, eb, S[2], "latch \"G major\""), (sA, ea, S[1], "latch \"C major\""))):
        ax.vlines(t[spk], row + 0.15, row + 0.75, color=c, lw=1)
        on = np.where(ev > 0)[0]
        ax.add_patch(Rectangle((t[on[0]], row + 0.05), t[on[-1]] - t[on[0]] + dt, 0.8, color=c, alpha=0.18, lw=0))
    ax.set_yticks([0.45, 1.45], ["key latch\n\"G major\"", "key latch\n\"C major\""])
    ax.set_ylim(0, 2)
    ax.text(t[26], 1.92, "brief evidence for C", ha="center", va="top", fontsize=8.5, color=INK2)
    ax.text(t[90], 1.92, "C keeps itself on (self-weight > threshold)", ha="center", va="top", fontsize=8.5, color=INK)
    ax.text(t[145], 0.92, "strong evidence for G", ha="center", va="top", fontsize=8.5, color=INK2)
    ax.text(t[205], 0.92, "G takes over, inhibits C", ha="center", va="top", fontsize=8.5, color=INK)
    heads = ["① Plain IF neuron (C1 default): no leak, so it never forgets; it only resets when it fires",
             "② Slow integrator: high threshold + negative bias = linear leak (fires only after sustained evidence)",
             "③ Key latches: self-excitation keeps firing, mutual inhibition lets only one key be on (spike raster)"]
    for a, h in zip(axs, heads):
        a.set_xlim(0, t[-1])
        a.text(0, 1.1, h, transform=a.transAxes, fontsize=10.5, weight="bold", color=INK)
    for a in axs[:-1]:
        a.set_xticklabels([])
    axs[-1].set_xlabel("time (s) at 16 ms per SNN step  ·  short ticks = spikes")
    title(fig, "Memory without a time constant: three ways a C1 neuron can remember",
          "Simulated with Talamo's C1 DigitalNeuron equation U(t+1) = 0 if U(t) > threshold, else U(t) + Σ w·s + bias (floor 0). Weights are made up for illustration. ③ needs inhibition on two consecutive steps (direct + 1-step relay): a C1 neuron ignores input in the step it fires, so latches firing in alternate steps would never inhibit each other. Untested on the real chip (D05).")
    save(fig, "c1-neuron-memory.png")


# ---------------------------------------------------------------- 7. training crop
def fig_crop():
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.74])
    ax.set_xlim(-1, 61)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 8), 60, 0.9, boxstyle="round,pad=0,rounding_size=0.3", color="#efeee9", lw=0))
    ax.text(0, 9.3, "one song (≈ 60 s ≈ 3,750 frames): too long and too unstable for full BPTT", fontsize=9.5, color=INK2)
    ax.add_patch(Rectangle((22, 8), 6, 0.9, color=S[0], lw=0))
    ax.plot([22, 0], [8, 5.9], color=GRAY, lw=0.8)
    ax.plot([28, 60], [8, 5.9], color=GRAY, lw=0.8)
    ax.text(25, 7.6, "random 6 s crop", ha="center", va="top", fontsize=8.5, color=INK2)
    ax.add_patch(Rectangle((0, 3.9), 20, 2, color="#dcdbd5", lw=0))
    ax.add_patch(Rectangle((20, 3.9), 40, 2, color=S[0], lw=0))
    ax.text(10, 4.9, "burn-in 1–2 s\nno loss: state & key latch settle", ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(40, 4.9, "scored window 2–5 s\nloss on every 16 ms frame, BPTT through the whole crop", ha="center",
            va="center", fontsize=9.5, color="white", weight="bold")
    # frame ticks + target shift (3 frames ≈ 50 ms), drawn zoomed
    x0, w = 22, 1.6
    for k in range(10):
        ax.add_patch(Rectangle((x0 + k * w, 2.3), w * 0.86, 0.7, color="#86b6ef" if k < 7 else "#efeee9", lw=0))
    ax.annotate("", (x0 + 3 * w + 0.7, 3.05), (x0 + 0.7, 3.05), arrowprops=dict(arrowstyle="-|>", color=S[1], lw=1.6,
                connectionstyle="arc3,rad=-0.5"))
    ax.text(x0 + 0.7, 2.0, "input t", ha="center", va="top", fontsize=8.5, color=INK)
    ax.text(x0 + 3 * w + 0.7, 2.0, "target t+3", ha="center", va="top", fontsize=8.5, color=S[1], weight="bold")
    ax.text(x0 + 10 * w + 1, 2.65, "each input frame is trained against the\nchord ≈ 50 ms (3 frames) later: this mostly\ncompensates our own latency (D07)", va="center", fontsize=9, color=INK)
    ax.text(0, 2.65, "Curriculum: windows\n0.5 → 2 → 5 s, then int8\nquantisation-aware fine-tuning", fontsize=9, color=INK2, va="center")
    title(fig, "How we train with full BPTT only: short crops with a burn-in",
          "Proposal from D06 (option B). A 2 s burn-in + 4 s window ≈ 375 SNN steps instead of 3,750. Talamo resets the state between batches, so every crop starts from rest; the burn-in is where the network catches up.")
    save(fig, "training-crop.png")


# ---------------------------------------------------------------- 8. system at a glance
def fig_system():
    fig = plt.figure(figsize=(12, 4.3))
    ax = fig.add_axes([0.01, 0.0, 0.98, 0.8])
    ax.set_xlim(0, 120)
    ax.set_ylim(-1, 33)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, -0.5), 92, 30.5, boxstyle="round,pad=0,rounding_size=1.5", fill=False, ec=INK2, lw=1))
    ax.text(2.5, 27.3, "Innatera Pulsar C1: everything in this box runs on the chip, live, no PC in the loop", fontsize=9.5, color=INK2)
    boxes = [
        (3, 11, "Mic", "16 kHz\nboard mic", "#efeee9", INK),
        (15, 11, "HW FFT", "1024 pt\n64 ms window\nevery 16 ms", "#efeee9", INK),
        (27, 11, "Semitone\nbands", "48 bands\n82 Hz–1.25 kHz\nlog · C kernel", "#efeee9", INK),
        (39, 11, "IF encoder", "48 spike\nlines", "#efeee9", INK),
        (51, 13, "Recurrent\nSNN", "64 hidden\n→ chord evidence\n(latches optional)", S[0], "white"),
        (65, 11, "Chord\nselector", "HMM (later GRU)\non RISC-V: key,\nchord history", "#d7f0e6", INK),
        (78, 11, "Voice\nleading", "2–3 violin\nvoices, nearest\nchord tone", "#efeee9", INK),
        (99, 11, "Violin\nsynth", "violin samples\n+ reverb (D19)", "#fde3d6", INK),
    ]
    for x, w, head, sub, fill, tc in boxes:
        ax.add_patch(FancyBboxPatch((x, 10), w, 15, boxstyle="round,pad=0,rounding_size=1", color=fill, lw=0))
        ax.text(x + w / 2, 21.5, head, ha="center", va="center", fontsize=10, weight="bold", color=tc)
        ax.text(x + w / 2, 14.3, sub, ha="center", va="center", fontsize=8, color=tc if tc == "white" else INK2)
    for x0, x1 in ((14, 15), (26, 27), (38, 39), (50, 51), (64, 65), (76, 78), (89, 99)):
        ax.annotate("", (x1, 17.5), (x0, 17.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    ax.text(96, 19, "MIDI over\nUART/USB", ha="center", fontsize=8, color=INK2)
    ax.text(57.5, 7.4, "SNN: chord evidence,\nshort context (+50 ms)", ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(70.5, 7.4, "long context\n(D20)", ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(84.3, 7.4, "rules:\n'beautiful'", ha="left", va="center", fontsize=8.5, color=INK)
    ax.text(104.5, 7.4, "the chip has no\naudio output", ha="center", va="center", fontsize=8, color=INK2)
    # expression lane: front-end features straight to the renderer (no SNN)
    ax.plot([32.5, 32.5, 83.5], [10, 3.2, 3.2], color=S[1], lw=1.6)
    ax.annotate("", (83.5, 10), (83.5, 3.2), arrowprops=dict(arrowstyle="-|>", color=S[1], lw=1.6))
    ax.text(58, 1.4, "expression path (no SNN): singer level · onsets · tuning  →  dynamics, timing, pitch bend (D15–D17)",
            ha="center", va="center", fontsize=8.5, color=INK)
    title(fig, "The system at a glance (current proposal)",
          "Sing → the SNN gives chord evidence → a chord selector adds long context → rules turn it into violin lines. Numbers from 01 Project/Key numbers; each box has a decision page (D02–D20).")
    save(fig, "system-at-a-glance.png")


# ---------------------------------------------------------------- 9. expression from the singer (D15, D17)
def violin_level(level_db, voiced, fps, attack=0.15, release=1.2, ratio=2.0, below=8.0):
    """D15 proposal: follow the singer's level, smoothed (fast attack, slow release), compressed 2:1 around
    its running median, 8 dB below the singer. In pauses the envelope just releases slowly (no hard stop)."""
    a_up, a_dn = np.exp(-1 / (attack * fps)), np.exp(-1 / (release * fps))
    env, out, hist = level_db[0], np.zeros(len(level_db)), []
    for i, (l, v) in enumerate(zip(level_db, voiced)):
        tgt = l if v else env - 0.5  # unvoiced: drift down slowly
        a = a_up if tgt > env else a_dn
        env = a * env + (1 - a) * tgt
        if v:
            hist.append(l)
        med = np.median(hist[-int(10 * fps):]) if hist else env
        out[i] = med + (env - med) / ratio - below
    return out


def running_reference(dev, voiced, fps, span=4.0):
    """D17 proposal: the singer's tuning reference = circular mean of the last `span` s of voiced deviations,
    UNWRAPPED (allowed beyond ±50 cents), so a singer drifting by more than half a semitone keeps a smooth
    reference instead of jumping to the neighbouring semitone mid-song."""
    ref, buf, prev = np.full(len(dev), np.nan), [], None
    for i in range(len(dev)):
        if voiced[i]:
            buf.append(dev[i])
        buf = buf[-int(span * fps):]
        if len(buf) > fps:
            c = ae.circ_mean_cents(buf)
            if prev is not None:
                c += 100 * np.round((prev - c) / 100)  # pick the copy closest to the previous value
            ref[i] = prev = c
    return ref


def fig_expression():
    f = next(fc.SONGS.glob("01_Happy_Birthday/audio/624721*"))
    r = ae.analyse(f)
    t, fps = r["t"], ae.FPS
    vl = violin_level(r["lvl"], r["voiced"], fps)
    ref = running_reference(r["dev"], r["voiced"], fps)
    fig = plt.figure(figsize=(11, 6.6))
    a1 = fig.add_axes([0.08, 0.53, 0.88, 0.27])
    a2 = fig.add_axes([0.08, 0.1, 0.88, 0.3])
    top = np.percentile(r["lvl"][r["voiced"]], 99)
    a1.plot(t, r["lvl"] - top, color=GRAY, lw=1, label="singer level (16 ms frames)")
    a1.plot(t, vl - top, color=S[0], lw=2, label="violin level (proposal: smoothed, 2:1, 8 dB below)")
    a1.set_ylim(-45, 3)
    a1.set_ylabel("dB re singer peak")
    a1.grid(axis="y")
    a1.legend(frameon=False, loc="lower right", fontsize=9, ncol=2)
    a1.text(0, 1.06, "① Dynamics (D15): the violin breathes with the singer but stays under her and never drops out abruptly",
            fontsize=10.5, weight="bold", color=INK, transform=a1.transAxes)
    v = r["voiced"]
    near = np.where(np.isnan(ref), 0, ref)
    dev_u = r["dev"] + 100 * np.round((near - r["dev"]) / 100)  # show each frame next to the reference
    a2.scatter(t[v], dev_u[v], s=5, color=GRAY, linewidths=0, label="sung pitch, cents from A440 notes")
    a2.axhline(0, color=INK2, lw=0.8)
    a2.text(t[-1], 1.5, "standard tuning A440 ", ha="right", va="bottom", fontsize=8.5, color=INK2)
    a2.plot(t, ref, color=S[1], lw=2.2, label="singer's tuning reference = violin pitch bend (proposal)")
    a2.set_ylim(-60, 90)
    a2.set_yticks([-50, 0, 50])
    a2.axhline(50, color=GRID, lw=0.8)
    end = ref[~np.isnan(ref)][-1]
    a2.annotate(f"drift ≈ {end - ref[~np.isnan(ref)][0]:+.0f} cents in 30 s:\nmore than half a semitone", (t[-1] - 1, end), (20.5, 72),
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a2.set_ylabel("cents")
    a2.set_xlabel("time (s)")
    a2.grid(axis="y")
    a2.legend(frameon=False, loc="lower left", fontsize=9, ncol=2)
    a2.text(0, 1.06, "② Tuning (D17): she starts in tune and drifts sharp; an A440 violin would end up more than half a semitone off",
            fontsize=10.5, weight="bold", color=INK, transform=a2.transAxes)
    for a in (a1, a2):
        a.set_xlim(0, t[-1])
    title(fig, "Making it sound good beyond the right note: follow the singer's loudness and tuning",
          "Freesound 624721 (Happy Birthday, female voice a cappella, CC0). Measured with jan/scripts/analyze_expression.py; the blue and orange lines are what the proposed rules (D15, D17) would send as MIDI CC11 and pitch bend.")
    save(fig, "expression-from-singer.png")


# ---------------------------------------------------------------- 10. where the voice lives (D02)
def fig_voice_map():
    rows = [  # label, lo, hi, colour, note
        ("Sung F0, adult men (approx.)", 85, 330, S[0], "our male singer: 110–125 Hz"),
        ("Sung F0, women (approx.)", 175, 520, S[0], ""),
        ("Sung F0, children (approx.)", 250, 600, S[0], ""),
        ("Harmonics 3–5 dominate pitch (F0 100–400)", 300, 2000, S[1], "Ritsma 1967"),
        ("Vowel formant F1 (men → children)", 270, 1030, S[2], "Peterson & Barney 1952"),
        ("Vowel formant F2 (men → children)", 840, 3200, S[2], "Peterson & Barney 1952"),
        ("Singer's formant (trained voices only)", 2500, 3200, S[3], "Sundberg 1974"),
        ("Consonants /s/, /f/ … (approx.)", 3000, 8000, GRAY, ""),
    ]
    layouts = [("A. plan: 64 semitone bands", 131, 4978), ("B. 64 semitone bands", 65, 2489), ("C. 48 semitone bands", 82, 1245),
               ("Violin (G3 up)", 196, 3000)]
    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_axes([0.34, 0.1, 0.6, 0.72])
    ys = list(range(len(rows) + len(layouts) + 1))[::-1]
    for (lab, lo, hi, c, note), y in zip(rows, ys):
        ax.barh(y, hi - lo, left=lo, height=0.6, color=c, alpha=0.85 if c != GRAY else 1, lw=0)
        ax.text(hi * 1.06, y, note, va="center", fontsize=8.5, color=INK2)
    y0 = ys[len(rows)]
    ax.axhline(y0, color=GRID, lw=1)
    for (lab, lo, hi), y in zip(layouts, ys[len(rows) + 1:]):
        ax.barh(y, hi - lo, left=lo, height=0.6, color="none", edgecolor=INK if "Violin" not in lab else S[4], lw=1.6, hatch="" if "Violin" not in lab else "//")
    ax.set_yticks([y for y in ys[:len(rows)]] + ys[len(rows) + 1:], [r[0] for r in rows] + [l[0] for l in layouts])
    ax.set_yticks([y0], ["input layouts compared ↓"], minor=True)
    ax.set_xscale("log")
    ax.set_xlim(50, 8000)
    ticks = [65, 131, 262, 523, 1047, 2093, 4186, 8000]
    ax.set_xticks(ticks, ["65\nC2", "131\nC3", "262\nC4", "523\nC5", "1047\nC6", "2093\nC7", "4186\nC8", "8000\nNyquist"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("frequency (Hz, log scale)")
    for s_ in ("left",):
        ax.spines[s_].set_visible(False)
    title(fig, "Where the singing voice lives, and which part the SNN needs",
          "Harmony needs the pitch, which lives in F0 and its harmonics up to ~2 kHz. Above that are vowel colour, trained-singer brilliance and consonants: they carry what is sung, not which chord fits. Ranges approximate; F0 ranges are textbook values (not verified per source).")
    save(fig, "voice-frequency-map.png")


# ---------------------------------------------------------------- 11. semitone bands vs mel vs MFCC (D02)
def fig_mfcc_example():
    import mfcc_check as mc
    import band_layout_check as bl
    f = next(fc.SONGS.glob("02_Twinkle/audio/Twinkle_twinkle_little_star_(vocal).ogg"))
    x = fc.load(f, 16000)[int(16000 * 4.2): int(16000 * 15.2)]
    f0, conf = fc.yin(x, 16000, 1024, 256)
    Sp = mc.power_spec(x, 1024, 1024)
    semi = bl.features(x, bl.LAYOUTS["C. 48 semitone, 82 Hz–1.25 kHz (MIDI 40–87)"])
    fbm = mc.mel_bank(64, 1024)
    mel = mc.log_mel(Sp, fbm)
    mf = mc.mfcc(mc.power_spec(x, 512, 512, 256), mc.mel_bank(128, 512), 32)
    n = min(len(semi), len(mel), len(mf), len(f0))
    semi, mel, mf, f0, conf = semi[:n], mel[:n], mf[:n], f0[:n], conf[:n]
    rms = np.sqrt((np.lib.stride_tricks.sliding_window_view(x, 1024)[::256][:n] ** 2).mean(axis=1))
    gate = rms > 0.1 * np.percentile(rms, 90)
    t = np.arange(n) * 256 / 16000
    v = conf & gate
    seq = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig = plt.figure(figsize=(11, 8.6))
    axs = [fig.add_axes([0.1, 0.66 - i * 0.29, 0.86, 0.2]) for i in range(3)]
    # a) semitone bands
    a = axs[0]
    a.imshow(np.where(gate[:, None], semi, 0).T, origin="lower", aspect="auto", cmap=seq, vmin=0, vmax=1,
             extent=[0, t[-1], -0.5, semi.shape[1] - 0.5], interpolation="nearest")
    mi = 69 + 12 * np.log2(f0 / 440) - 40
    a.scatter(t[v], mi[v], s=6, color=S[1], linewidths=0)
    a.set_yticks([0, 8, 20, 32, 44], ["82 Hz", "131", "262", "523", "1047"])
    a.set_ylabel("band")
    # b) log-mel
    b = axs[1]
    b.imshow(np.where(gate[:, None], mel, 0).T, origin="lower", aspect="auto", cmap=seq, vmin=0, vmax=1,
             extent=[0, t[-1], -0.5, mel.shape[1] - 0.5], interpolation="nearest")
    centres = (np.arange(513) * 16000 / 1024)[fbm.argmax(1)]
    b.scatter(t[v], np.interp(f0[v], centres, np.arange(len(centres))), s=6, color=S[1], linewidths=0)
    ticks = [0, 4, 10, 20, 32, 48, 63]
    b.set_yticks(ticks, [f"{centres[k]:.0f}" for k in ticks])
    b.set_ylabel("mel band (Hz)")
    # c) MFCC (z-scored per coefficient, gated)
    c = axs[2]
    z = (mf - mf[gate].mean(0)) / (mf[gate].std(0) + 1e-9)
    c.imshow(np.where(gate[:, None], np.clip(z, -2.5, 2.5), 0).T, origin="lower", aspect="auto", cmap=div, vmin=-2.5, vmax=2.5,
             extent=[0, t[-1], -0.5, 31.5], interpolation="nearest")
    c.set_ylabel("MFCC index")
    c.set_yticks([0, 8, 16, 24, 31])
    c.set_xlabel("time (s)  ·  orange dots = sung melody (YIN)  ·  white / grey = silence")
    heads = ["① Our input: 48 semitone bands (82 Hz – 1.25 kHz). Harmonics are separate stripes that move together with the melody",
             "② 64 log-mel bands at 16 kHz: below 1 kHz the bands are ~47 Hz apart (≈ 4 semitones at 200 Hz), so nearby notes merge",
             "③ 32 MFCC as in Innatera's audio example: keeps the spectral envelope (vowel), not the pitch stripes"]
    for ax, h in zip(axs, heads):
        ax.text(0, 1.06, h, transform=ax.transAxes, fontsize=10, weight="bold", color=INK)
        ax.set_xlim(0, t[-1])
    for ax in axs[:2]:
        ax.set_xticklabels([])
    title(fig, "The same sung phrase, three ways: what the SNN would see",
          "\"Twinkle twinkle little star (vocal)\", 11 s, 16 kHz. MFCC was designed to describe *what* is said or which scene it is (timbre, vowel) and deliberately smooths away pitch detail; our task needs exactly that detail. Measured comparison: D02 / jan/scripts/mfcc_check.py.")
    save(fig, "semitone-vs-mel-vs-mfcc.png")


if __name__ == "__main__":
    fig_system()
    fig_bands()
    fig_resolution()
    fig_frontend_acc()
    fig_chords()
    fig_chip()
    fig_neurons()
    fig_crop()
    fig_expression()
    fig_voice_map()
    fig_mfcc_example()
