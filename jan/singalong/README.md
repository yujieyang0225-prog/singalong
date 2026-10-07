# singalong — Amazing Grace live demo

**Goal (pitch Friday 9 Oct, 10:00):** someone sings *Amazing Grace* into the Pulsar EVK microphone. The chip turns
the audio into 48 semitone bands, a 48 → 64 → 24 recurrent spiking network on the SNN fabric reports per 16 ms
frame *which pitch class is sung* and *which chord pitch classes fit*. Those 24 spike counts stream to a laptop
over USB; the laptop follows the song (HMM), picks the chord, voice-leads a violin and plays it live.

```
 board (Pulsar C1)                                  │ laptop
 PDM mic 16 kHz → bands kernel (C) → IF encoder →   │ → selector (HMM: key / song / score follower)
 SNN 48→64 rec→24 → spike counts, 62.5/s, JSON/UART │ → violin renderer (voice leading, dynamics) → MIDI → FluidSynth
```

Everything on the laptop side **works and is tested here** (`python3 -m pytest -q tests`, 12 tests).
The chip side is prepared but **untested** (no SDK image, Docker daemon, board or DAMP data in the cloud
session): see *Untested* below. Each stage has a fallback so a demo exists at every point:

| Stage | What runs where | Status | Command |
|---|---|---|---|
| 0 | laptop only: audio file or mic → bands → **baseline evidence** (no SNN) → HMM → violin | ✅ works | `live.py --source mic --evidence baseline` |
| 1 | laptop only: the **int8 SNN** simulated in numpy from `weights_int8.npz` | ✅ runs; the included net is synthetic-trained and **below the baseline on real voices** (see below) | `live.py --evidence snn:<npz>` |
| 2 | **SNN on the chip (HIL)**, front-end on the laptop, spike counts back | 📝 script written | `Pulsar/IF_DNR/deploy_live.py <run> --mode hil` |
| 3 | **all on the chip**: mic → C kernel → encoder → SNN → counts over UART | 📝 kernel in C ✅ matches Python; app glue untested | `deploy_live.py <run> --mode standalone --build` |

## Quick start (laptop, no SDK)
```bash
cd jan/singalong
pip install numpy scipy torch mido python-rtmidi sounddevice pyfluidsynth pytest      # torch: CPU wheel is fine
python3 -m singalong.synth --n 40 --out ../../data/synth40 --wav       # synthetic singers (any key/tempo/voice)
python3 -m pytest -q tests                                              # ~2 min
python3 live.py --source wav:../../data/synth40/synth01_000.wav --evidence baseline --selector score --sink print
python3 eval_chain.py --data ../../data/synth40 --evidence baseline --selector score   # numbers, not sound
```
Sound: install FluidSynth and a GM SoundFont (e.g. `FluidR3_GM.sf2`, violin = program 40) and use
`--sink fluidsynth:/path/to.sf2`, or run FluidSynth as a MIDI device and use `--sink midi:fluid`.
Microphone: `--source mic` (16 kHz, uses `sounddevice`). Board: `--source board:/dev/ttyUSB1`.

## Train, primary path (chip-exact, Innatera's): `Pulsar/IF_DNR/train.py`
As before (Talamo `Pipeline.fit` → `quantize` → `sim_check.py`), now with the fixes and options from this
session: `--out 24 --loss dual --aug --data … --lr 0.1`. **This is the path to trust for the chip**; everything the
torch path below produces must come back through it (`import_torch_weights.py`) and be checked there.
Export for the laptop: `Pulsar/IF_DNR/export_npz.py data/runs/<run>` → `weights_int8.npz`.
Compare the three implementations on one input (also explains the 73.7 % simulator agreement of run 2):
`Pulsar/IF_DNR/compare_torch_talamo.py data/runs/<run> --sim`.

> **Risk, stated plainly.** The torch copy of the C1 neuron (`torchsnn.py`) was written from the vault's notes on
> the SDK, not from the SDK source, and could differ in details (encoder reset, one-step layer delay, refractory,
> accumulator width). Talamo's own software neuron already disagrees with the SoC simulator in 26 % of spike
> slots, so *both* paths need the comparison. Until `import_torch_weights.py --check` reports high agreement,
> treat torch-trained weights as a fast way to find hyper-parameters, not as the demo network.

## Train, fast path (GPU, no SDK): `train_torch.py`
Plain-PyTorch implementation of the C1 network (`singalong/torchsnn.py`): IF encoder (reset by subtraction),
integrate-and-fire neurons with reset to zero, floor at 0, trainable bias (= linear leak), int8 weights,
thresholds 127, one step per frame, recurrent spikes from the previous step. Trained in **chip units** (the
weights *are* the int8 numbers), last epochs with straight-through rounding (QAT), export = round.
```bash
# on gpusrv03 (or any CUDA box): plain torch, no Innatera image needed
python3 train_torch.py --data ../../data/damp100 --data ../../data/synth200 --hidden 80 --epochs 40 --qat-epochs 8 --name damp_synth_rnn80
# outputs: data/runs/<name>/{weights_int8.npz, checkpoint.pt, log.csv, metrics.json}
python3 eval_chain.py --data ../../data/damp100 --evidence snn:../../data/runs/<name>/weights_int8.npz --selector score
```
Sizes that fit (`torchsnn.fits_c1`): hidden ≤ 80 without plane pairing (48 + 80 = 128 fan-in), up to ~128 hidden
with pairing (neuron budget 192 then), 24 outputs ≤ 64 readable counters.
- Augmentation (`singalong/augment.py`, D11) is on by default: transposition ±6 semitones by shifting the band
  rows (labels rolled), tempo 0.75–1.35×, rubato, noise, band dropout. **Transposition is the fix for "the network
  learned the average chord"** (run 2): after shifting, no single chord is the average.
- Dual head (D07): outputs 0–11 chord pitch classes (BCE on the 128 ms spike rate), 12–23 melody pitch class
  (cross-entropy, 13th class "no note"). The melody head is what the score follower uses; the chord head is the
  "free mode" evidence.
- Learning rate is in chip units: `--lr 0.3` (default). `1e-3` does nothing visible in 10 epochs; this is the
  likely reason Talamo runs 0–2 barely moved (see the experiment note).
- CPU here: ~1.5 min/epoch for 160 recordings at batch 128 (python loop over 125 steps); a GPU should be 10–50×.

Then back into Innatera's path (mandatory before the demo): `run.sh python3 Pulsar/IF_DNR/import_torch_weights.py data/runs/<name>/weights_int8.npz
--out data/runs/<name>_talamo --check data/damp100` → `sim_check.py` → `deploy_live.py`.

## What was found while building this (details: vault, 2026-10-07 Chord SNN training → *Diagnosis*)
1. `RateBCE` clamped the rate to [1e-3, 1−1e-3]: **a silent output neuron gets zero gradient** and can never
   start firing. Fixed in `model.py` (affine squeeze) and `torchsnn.DualLoss`.
2. Learning rate 1e-3 on int8-scale weights (std ≈ 20) moves nothing. Use ~0.3 (AdamW).
3. Surrogate gradient must be wide (±1 threshold) and the initial weights large enough that neurons fire.
4. The HMM's per-frame evidence must be weak (β ≈ 0.25 per frame): 62.5 frames/s are not independent
   observations. With β ≈ 6 the selector followed every frame and changed chord 3× too often.
5. Following the **melody** of the known song (score follower, 12 keys × 37 notes) tracks position reliably from
   raw band evidence; following chords alone floats between identical "I" bars.

## Measured here (synthetic Amazing Grace singers, 4 voice types, random key/tempo; `eval_chain.py`)
> **Synthetic only.** DAMP is not reachable from the cloud session (network policy), so every SNN number below is
> on synthetic vowel singers (`synth.py`). The chain and the selector are validated; the SNN's *real-voice*
> performance is not, except for the YIN cross-check in `real_voice_check.py` (results in the vault note).
> First job on a machine with `data/damp100`: `eval_chain.py --data ../../data/damp100 --evidence snn:…`.
>
> **Real-voice check (`real_voice_check.py`, SNN pitch class vs YIN on the `08 Songs` a cappella files):** the
> synthetic-only network gets 0.36–0.62 exact (0.72–0.95 within a semitone), the hand-written baseline 0.65–0.79
> (0.84–0.96). Adding YIN-pseudo-labelled real files (4 files, 8× weight) made it worse (0.32) and unstable.
> **So `pretrained/synth200_rnn80_int8.npz` is a smoke-test artefact, not the demo network.** For real singers
> today use `--evidence baseline`; the DAMP-trained network (Talamo, `train.py`) must beat the baseline in this
> check before it replaces it.
| Evidence → selector | exact chord | sung note in chord (ref 0.81) | chord changes/min (ref 19) |
|---|---|---|---|
| perfect pitch → score follower | 0.76 | 0.82 | 24 |
| baseline (harmonic sum, no SNN) → score follower | 0.71 | 0.77 | 22 |
| baseline → key HMM (any song) | 0.09 | 0.80 | 51 |
| **trained int8 SNN 48→80→24 (torch, 40 epochs, synthetic; `pretrained/synth200_rnn80_int8.npz`) → score follower** | **0.76** | **0.80** | 24 |
| trained int8 SNN 48→64→24 (15 epochs) → score follower | 0.70 | 0.73 | 24 |

## Files
```
singalong/frontend.py     the one front-end definition (offline + streaming), == jan/prep/precompute_damp.py
singalong/amazing_grace.py lead sheet in degrees; reads data/damp100/reference.npz if present (amazing_grace.md)
singalong/synth.py        synthetic singers → dataset in the damp100 format (python3 -m singalong.synth)
singalong/data.py         recordings → crops; labels [chord 12 | mask | melody class]; multi-dataset
singalong/augment.py      D11 augmentation on recordings (X and labels together)
singalong/torchsnn.py     the C1 network in torch (GPU), QAT, export; fits_c1()
singalong/snn_numpy.py    integer reference of the exported network (laptop fallback, 3-way comparison)
singalong/selector.py     KeyChordHMM (any song) · SongHMM (chord progression) · ScoreFollowerHMM (melody)
singalong/renderer.py     voice leading, dynamics (CC11), tuning (pitch bend); sinks: MIDI / FluidSynth / print
singalong/evidence.py     baseline (no SNN) · numpy SNN · torch SNN · board JSON lines
singalong/metrics.py      F1 / exact / pitch acc / note-in-chord / changes per minute
train_torch.py            GPU trainer ·  live.py  host app ·  eval_chain.py  offline scoring
kernels_c/bands/          C front-end (bands_core.c) + PC test harness; == Python to ±1
Pulsar/IF_DNR/train.py    Talamo trainer (chip path)  · export_npz.py · import_torch_weights.py
Pulsar/IF_DNR/sim_check.py simulator check (existing) · compare_torch_talamo.py · deploy_live.py (board)
tests/test_chain.py       12 tests, no SDK needed
```

## Untested (needs the SDK container, the board, or the DAMP download) — in the order to do them
1. `Pulsar/IF_DNR/train.py --out 24 --loss dual --aug --lr 0.1 --epochs 10` on `data/damp100` (does the Talamo
   path learn now?). Then `export_npz.py --list` to confirm the parameter names `export_npz.MAP` assumes.
2. `train_torch.py --data data/damp100 --data data/synth200` (synth: `python3 -m singalong.synth --n 200 --seed 1
   --out data/synth200`), then `import_torch_weights.py … --check data/damp100`: the spike agreement between
   Talamo's software neuron and `snn_numpy` tells whether `torchsnn.py` models the C1 neuron right
   (`enc_reset`, `layer_delay` variants are printed). Fix `torchsnn.py` to the best variant and retrain.
3. `compare_torch_talamo.py <run> --sim` for the simulator; `sim_check.py <run> --build` (mapper accepts 24 outputs?).
4. `deploy_live.py <run> --mode hil` with the board, laptop `live.py --source board:<port>`.
5. Standalone app: the `bands` custom kernel needs the Talamo wrapper (`kernel_bands_core.c` calling
   `bands_frame()`, HW FFT instead of the float FFT, `register_custom_kernels`), mic at 16 kHz without ALC.
6. `data/damp100`: `jan/prep/precompute_damp.py --n 300` for more singers (80 usable of 100 so far).
7. Real-voice transfer: `real_voice_check.py` (SNN vs YIN on the `08 Songs` a cappella files) after every training
   run; `pseudo_label_real.py` turns those files into YIN-labelled training data for the melody head (pass the
   folder several times to `--data` to weight it). Record the team on Thursday → more of this.
