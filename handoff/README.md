# Singalong pitch model for the demo: A_vocalset_P

Handoff from Simon's side to Yujie, 2026-10-08, branch `singalong-pitch-handoff`. Everything the demo laptop needs
for the singalong SNN is in this folder or in the code commit below. Nothing here needs the cluster or `data/`.

| Path | What |
|---|---|
| `runs/A_vocalset_P/weights_int8.npz` | **the model** (md5 `64623170b4e5537306a89e7319e3df7d`) |
| `runs/A_vocalset_P/` | the whole run folder: `checkpoint.pt`, `log.csv`, `metrics.json`, `eval_pitch.json` |
| `ranking.txt` | the evaluation table as printed on the cluster |
| `testdata/damp250/` | the 24 DAMP validation recordings as band features + `manifest.csv` + `reference.npz` (23 MB) |
| `testdata/vocalset/` | the 7 held-out VocalSet singers as band features + `manifest.csv` (41 MB, 913 recordings) |

> [!warning] Replace the old handoff weights
> `data/handoff/singalong_int8.npz` (md5 `4c5e8d27232faa74cded8b1c58e06181`), passed around since 11:52 today, is
> **B_damp_lr0.03**: chord + pitch head, trained on DAMP, 0.226 val score. It is not the demo model. Use
> `runs/A_vocalset_P/weights_int8.npz` instead.

## 1. Ranking

`simon/eval_pitch.py` scores the int8 weights (the file that goes to the chip) on the same fixed crops for every
run. The values are melody (pitch-class) accuracy. The rule is: rank by VocalSet songs; VocalSet exercises break
ties within 0.02; DAMP is a sanity check (a model clearly below the others there is suspect).

| run               | VocalSet songs | VocalSet exercises | DAMP      | chord outputs |
|-------------------|---------------:|-------------------:|----------:|--------------:|
| **A_vocalset_P**  | **0.530**      | **0.599**          | **0.364** | 0.0000        |
| A_vocalset_lr0.03 | 0.378          | 0.486              | 0.263     | 0.0192        |
| A_vocalset_lr0.1  | 0.322          | 0.466              | 0.233     | 0.0002        |

Crops: songs 427, exercises 512, DAMP 512. Chance is about 0.08.

**Winner: A_vocalset_P.** It leads on songs by 0.15, so the tie-break never applies, and it is not low on DAMP (it is
best there too).

- **Scope:** Simon chose the VocalSet (A) model. B_damp_wd0_lr0.1, B_damp_P and C_both_P are not ranked here.
- **Pending:** a search around A_vocalset_P is still running (cluster job 16643, `simon/dataprep/search_vocalset.sh`,
  ranked by the same script). If a search run wins under the same rule, a new `runs/<name>/` is added on this branch
  and this README changes. Until then, A_vocalset_P is the model.
- **Reproduced:** running the eval on only the test data in this folder, on CPU, gives the same row: 0.530 / 0.599 /
  0.364.

```bash
cd jan/singalong && python3 ../../simon/eval_pitch.py ../../handoff/runs/A_vocalset_P --vocalset ../../handoff/testdata/vocalset --damp ../../handoff/testdata/damp250 --device cpu
```

The eval needs torch. It overwrites `eval_pitch.json` in the run folder with the same numbers.

## 2. Run folder

**`weights_int8.npz`, `metrics.json` and `checkpoint.pt` are all from epoch 40.**

- `train_torch.py` writes `weights_int8.npz` and `metrics.json` at the best rounded-weights (QAT) epoch, and
  `checkpoint.pt` after every epoch.
- This run had 40 epochs and its best epoch was the last one, so all three coincide.
- Validation pitch accuracy at epoch 40 was 0.638 (on the run's own validation split, not the table above). It was
  still rising slightly over the last 4 epochs (0.631 to 0.638).

## 3. Code that produced it

- **Branch:** `singalong-pitch-handoff`.
- **Code commit:** `b7da7bb4bcd59cd1b0543a862471d81a05180c49`.
- **Base:** `origin/main` at `fd3e4cb`.

What the commit contains:

| File | Change |
|---|---|
| `jan/singalong/train_torch.py` | `--wd`, `--compact`, `--chunk`, `--pitch-only` |
| `jan/singalong/singalong/data.py`, `augment.py`, `metrics.py` | compact dtypes; datasets without chord labels |
| `simon/eval_pitch.py`, `simon/plot_runs.py` | new |
| `simon/dataprep/train_slurm.sh`, `build_datasets.sh`, `check_datasets.py` | changed |
| `simon/dataprep/search_vocalset.sh` | new |

`jan/singalong/singalong/torchsnn.py` and `frontend.py` are **unchanged** from `main`; nothing in them needed
committing.

**Command line that trained A_vocalset_P.** It is rebuilt from `metrics.json` `args`. Every other setting is the
script default (batch 64, lr-enc 1.0, qat-epochs 4, crop 125, shift 6, seed 0), and `w_chord 0` follows from
`--pitch-only`.

```bash
cd jan/singalong && python3 train_torch.py --data ../../data/vocalset --out 24 --hidden 80 --epochs 40 --lr 0.1 --wd 0 --pitch-only --compact --chunk 64 --device cuda --name A_vocalset_P
```

The same run on the cluster, through the SLURM script:

```bash
sbatch --array=0 --export=ALL,TAG=_P,EXTRA="--compact --chunk 64 --lr 0.1 --wd 0 --pitch-only" simon/dataprep/train_slurm.sh
```

**How we know this is the code that ran (verified 2026-10-08).**
- **Same files.** The md5 hashes of `train_torch.py`, `torchsnn.py`, `frontend.py`, `data.py`, `augment.py`,
  `metrics.py` and `eval_pitch.py` on the cluster (`~/MNH26SVR`) are identical to this commit.
- **The run needed this version.**
  - `--pitch-only` was added to `train_torch.py` at 12:54, and the file has not changed since.
  - The run's `metrics.json` has `pitch_only: true`, so it ran with this version.
  - The other five files were last changed before 12:54.

## 4. Model facts

- **Pitch-only.** Chord outputs 0–11 are silent: weights 0, bias −128, measured rate 0.0000 on all three test sets.
  Outputs 12–23 are the melody pitch classes: output 12 + k is pitch class k, with 0 = C, 1 = C#, … 11 = B. On the
  laptop, run `live.py` with `--use melody`. Free (chord) mode does not work with this model.
- **Network unchanged.** From `metrics.json` `cfg`: 48 inputs, 80 recurrent hidden, 24 outputs. Encoder threshold
  starts at 120, neuron threshold 127, weights int8 (−128..127).
  - The encoder thresholds are trained per input band. They are the 48 integers `enc_threshold` in
    `weights_int8.npz`, not one scalar.
- **Front-end** is `jan/singalong/singalong/frontend.py`, unchanged since commit `d012431` (2026-10-07):
  - 16 kHz audio, 1024-sample Hann window, hop 256 (16 ms, 62.5 frames/s)
  - 48 triangular semitone bands, MIDI 40–87 (82 Hz – 1.245 kHz)
  - log2 band energy, normalised per frame over a 40 dB range
  - voicing gate: a frame counts as sung when its level is at least 15 dB above the noise floor; otherwise all 48
    bands are 0
  - output: integers 0–100 per band
  - The training features were built with its `log_bands` (`simon/dataprep/make_damp_format.py`), and `data.py`
    applies `normalise` at load time.
- **Readout used in training and on the laptop:** the spike rate of each output as a causal moving average over the
  last 8 frames (128 ms). It is `torchsnn.spike_rate`.

## 5. Test data

64 MB in total, in `testdata/`. The two manifests list only the included recordings, so `data.load_recordings` works
on these folders as they are.

- **`damp250`:** the 24 recordings of the DAMP validation split. The labels need `reference.npz` (the Amazing Grace
  score).
- **`vocalset`:** every recording of the 7 singers no model trained on: female2, female8, female9, male10, male11,
  male3, male5.
  - 266 val, 602 test and 45 songs.
  - The eval leaves out the spoken takes.

## 6. Chip import check on the torch-trained weights

Run on Simon's machine in the Innatera SDK image 4.7.1. This is the first run of the import on torch-trained weights.
It is reported as-is; the chip side is Yujie's.

```bash
jan/singalong/run.sh python3 Pulsar/IF_DNR/import_torch_weights.py ../../handoff/runs/A_vocalset_P/weights_int8.npz --out ../../data/runs/A_vocalset_P_talamo --check ../../handoff/testdata/damp250
```

- **Mapping:** every parameter found its Talamo counterpart: encoder threshold (48), W_in (48×80), W_rec (80×80), b_h,
  thr_h, W_out (80×24), b_o, thr_o.
- **Spike agreement:** 0.912 over 600 frames × 24 outputs. But Talamo fired **895** output spikes where the numpy
  reference (`singalong.snn_numpy`) fired **416**.
  - Only 3–6 % of frames have a spike, so the agreement figure is mostly frames where both stay silent.
  - The two implementations do not behave the same yet.
- **Variants:** the script's own variants (layer_delay 0/1, encoder reset subtract/zero) give 0.912–0.916, so neither
  setting explains the gap.
- **Not a scale factor:** the older torch model B_damp_lr0.03 gave 0.801, with 832 vs 2315 spikes. That is the
  opposite direction.
- **VocalSet check crashes:** `--check` on the VocalSet folder fails. The script always takes frames 600–1200 of the
  first validation recording, which is shorter than that.
- **Observation for the chip side:** torch stores weights as [out, in] and Talamo as [in, out].
  - The import transposes W_in and W_out because their shapes differ, but copies the square W_rec unchanged.
  - Simon's side tried transposing W_rec once; the B model's numbers did not change at all. Either W_rec is not what
    the check exercises, or the values written are not the ones the quantized pipeline runs.
  - The numbers above were measured with that one-line transpose in place. It is not in this commit.

## For the board pipeline: what this model expects

These are facts about the model, compared with `yujie/live-mic-voice` (`pulsar_live_voice`). They are not a design.

| | This model | `pulsar_live_voice` voice baseline |
|---|---|---|
| Audio | 16 kHz mono, int16 | 16 kHz 16-bit stream (from 19531.25 Hz, resampled) |
| Features | 48 semitone bands, 1024 window, 256 hop, 0–100 (above); **not MFCC** | Talamo MFCC, 512/512, 128 mel |
| C reference | `jan/singalong/kernels_c/bands/bands_core.c`: one 1024-sample int16 frame → 48 × uint8 0–100 | — |
| Network state | recurrent, trained on 2 s crops after a 0.5 s (31-frame) burn-in; state carries from frame to frame | reset once per 1 s window |
| Outputs | 24, of which 12–23 are used (pitch class); 0–11 always silent | 3 classes |
| Readout | rate over the last 8 frames (128 ms), every frame | max rate per window |

After a reset, the model's outputs are not trained to be meaningful for about the first 0.5 s.

## Notes that are out of date

Many notes in the vault, and several docs on other branches, predate today's results. The list, with what
is current, is in
`Hackathon/00 Inbox/2026-10-08 Out-of-date notes after the cluster runs.md`.
