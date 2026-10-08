# Feedback from the chip side for the next training session

Model looked at: `handoff/runs/A_vocalset_P/weights_int8.npz` (md5 `64623170…`).
Section 7 covers its successor trained under the chip's rules, `A_vsP_chip_shift3`.
Branch `yujie/singalong-chip`, started from `singalong-pitch-handoff` at `2a612c3`.
Kept up to date as the chip work proceeds; the log at the end says what changed and when.

Sections 1 to 4 were measured in **Talamo 4.7.1's software model** of the chip and in an integer
reference. Section 5 has the first board measurement.

## Short version

| | |
| --- | --- |
| Can the trained weights be put into Talamo correctly? | **Yes, since 2026-10-08.** The import script was broken; it is fixed and its check passes. |
| Does Talamo's model then behave like the trained network? | Not spike for spike: two rules differ (below). **By decision these are kept as they are.** |
| What do those differences cost? | About 1 point of pitch accuracy at most. No retraining is needed because of them. |
| What should the next training session change? | Nothing is required. Section 4 lists what would make the model safer on the chip. |
| Does the network fit and run on the real chip? | **Yes** (2026-10-08): it builds, the chip accepts it, and all 1,451 test windows came back identical to the reference. |
| Is the chip fast enough for a live loop? | Yes, at one update every 0.5 s: 0.37 to 0.39 s per 2 s window on the board. The live loop runs, with about 1 s from sound to note; see section 6. |
| Does training under the chip's rules help? | Yes on songs. `A_vsP_chip_shift3` scores 0.577 / 0.591 / 0.378 on the board, against 0.516 / 0.589 / 0.364 for `A_vocalset_P`; see section 7. |

## 1. Accuracy to expect on the chip from the same weights

Same crops and scoring as `simon/eval_pitch.py`. The first row reproduces that script's numbers
exactly, which is the check that the reference used here is right.

| Rules the network runs under | VocalSet songs | VocalSet exercises | DAMP |
| --- | ---: | ---: | ---: |
| As trained (`torchsnn.py`) | 0.530 | 0.599 | 0.364 |
| Recurrent input as on the chip, no layer delay | 0.532 | 0.601 | 0.368 |
| **Chip rules, scored as the output arrives** | **0.516** | **0.589** | **0.364** |
| Chip rules, labels shifted by the one-frame delay | 0.524 | 0.596 | 0.366 |
| `snn_numpy.py` default (encoder reset "subtract") | 0.549 | 0.581 | 0.367 |

Reproduce: `python3 ../../yujie/chip/eval_modes.py ../../handoff/runs/A_vocalset_P` from `jan/singalong`
in the SDK container. Result file: `yujie/chip/chip_expectation.json`.

Reading: the model is robust to the rule differences. The small loss under chip rules is mostly the
16 ms delay, not a change in what the network computes.

## 2. The two differences that are kept

With these two rules, the integer reference in `yujie/chip/chipref.py` reproduces Talamo's output
spike for spike (same count, same positions) on DAMP and VocalSet recordings.

| | Trainer (`torchsnn.py`) | Talamo's model of the chip |
| --- | --- | --- |
| **Recurrent input** | added together with input and bias, then one floor at 0: `U = max(0, U + W·s_in + W_rec·s_prev + b)` | applied after the neuron has fired, with its own floor: `U = max(0, U + W·s_in + b)`, fire, reset, then `U = max(0, U + W_rec·s_now)` |
| **Layer delay** | the output layer sees the hidden spikes of the same step | the output layer sees them one step later: every output is 16 ms late |

Why the first matters in principle: a strong inhibitory recurrent kick is cut off at 0 before the
next input arrives, so inhibition cannot "carry over" into the next frame the way it does in training.
In practice it changed the hidden firing rate from 0.156 to 0.161 and accuracy by less than 0.005.

Everything else agrees between trainer and chip model: the encoder resets to zero, thresholds are
strict (`>`), neurons floor at 0 and reset to zero, weights are int8, thresholds 127.

**If a later session wants exact agreement,** `C1Layer.forward` in `torchsnn.py` would apply
`s @ Wr.t()` after the reset with a second clamp, and feed the output layer the hidden spikes of
the previous step. This is optional.

## 3. What was wrong with the import (fixed, for the record)

`jan/singalong/Pulsar/IF_DNR/import_torch_weights.py` as handed over:

1. **The imported values were never used.** They were copied into the live tensors, but `pipe(...)`
   and the saved parameter file use a stored parameter set. Every earlier check (0.912 "agreement",
   895 against 416 spikes), and any `sim_check.py` or `deploy_live.py` run from that file, used a
   **randomly initialised network**. Fix: `step.save_params()` inside the quantized context.
2. **The encoder was scaled by 128.** `quantize()` leaves the IF encoder multiplying its input by
   2⁷ with a threshold scaled to match. The trained thresholds are unscaled. Fix: multiplier 2⁰,
   sign +1, bias 0.
3. **`W_rec` was not transposed.** It is square, so the shape test that transposes `W_in` and `W_out`
   never fired. torch stores `[to, from]`, Talamo `[from, to]`.
4. **The check crashed on short recordings** and measured the wrong thing (next section).

The check now compares `pipe(...)`, the steps called directly, the saved file reloaded into a new
pipeline, and the chip-rule reference. All four are identical on DAMP and on VocalSet:

```
python3 Pulsar/IF_DNR/import_torch_weights.py ../../handoff/runs/A_vocalset_P/weights_int8.npz \
    --out ../../data/runs/A_vocalset_P_talamo --check ../../handoff/testdata/damp250
... IMPORT CHECK PASSED
```

## 4. What would help the next training session

Ordered by how much it should matter on the chip. Items marked *suggestion* are judgement, not measurement.

1. **Do not use "share of equal entries" to compare spike trains.** About 95 % of entries are silent
   in both, so two unrelated trains score above 0.9. The old 0.912 corresponded to an F1 of 0.02 on
   spike positions. Use spike counts plus F1 of spike positions, or better, the task accuracy.
2. **Run the import check after every export.** It takes a minute and catches a wrong export format.
   A new model is only "chip-ready" when it prints `IMPORT CHECK PASSED`.
3. **Use `eval_modes.py` as the yardstick for the chip**, next to `eval_pitch.py`. If a future model
   is less robust than this one, the "chip rules" row will show it before any board is involved.
4. **Fix three stale statements, so nobody trains or debugs against the wrong rule:**
   - the docstring at the top of `torchsnn.py` says the encoder resets by subtraction with `>=`;
     the code resets to zero with `>`, which is what the chip does;
   - `jan/singalong/README.md` says the same;
   - `singalong/snn_numpy.py` defaults to `enc_reset="subtract"`, so the laptop fallback
     (`live.py --evidence snn:…`) runs a different encoder from the one trained. On this model the
     accuracy is similar (last row of the table), but it is a different network: 4,570 input spikes
     against 3,798 on the same recording.
5. **Outputs are sparse and the readout is coarse** (*suggestion*). The twelve pitch outputs fire
   0.56 spikes per frame in total, and the rate is taken over 8 frames, so each output's rate is one
   of nine values. Ties and one-spike differences will decide many frames. On another model on this
   board, software and board picked a different class on 13.5 % of identical inputs for exactly this
   reason. A loss that rewards a margin between the best and second-best output, or a longer rate
   window, should make the result steadier.
6. **Keep the 31-frame burn-in in mind for deployment** (*suggestion*). The chip side plans to send
   2 s windows and discard the first 0.5 s of each, which matches how the crops were trained.

## 5. On the chip

**Built and run on a Pulsar EVK, 2026-10-08** (`yujie/chip/build_chip.py`; RAM boot, nothing flashed).
The application is host-fed: the PC sends 125 frames of 48 bands, the chip runs the encoder and the
recurrent network, and returns the output spikes of every frame.

| | |
| --- | --- |
| Network | 48 inputs, 80 recurrent hidden, 24 outputs: 104 neurons |
| Synapses in use | 10,917 non-zero (3,766 input, 6,215 recurrent, 936 output) |
| Largest fan-in | 128 (48 inputs + 80 recurrent), which is the limit the trainer's `fits_c1` assumes: **no headroom** |
| Talamo's mapper | accepted it without a warning; serialized configuration 15,685 bytes |
| Application RAM | 91,424 bytes of 384 KB (23 %) |
| Time on the board | **0.34 s per 2 s window**, plus 6.6 s once per call for the boot (`board_timing.json`) |

**Board against the reference, whole handoff test set** (`yujie/chip/board_check.py`, `board_check.json`):

| Test set | Windows | Identical to the reference | Spikes, board / reference | Pitch accuracy on the board | Reference, chip rules | As trained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VocalSet songs | 427 | 427 | 28,807 / 28,807 | **0.516** | 0.516 | 0.530 |
| VocalSet exercises | 512 | 512 | 29,423 / 29,423 | **0.589** | 0.589 | 0.599 |
| DAMP | 512 | 512 | 42,617 / 42,617 | **0.364** | 0.364 | 0.364 |

The chip reproduces the chip-rule reference spike for spike on every window: 100,847 spikes, none
missing, none extra. This validates `chipref.py` with `CHIP` rules for this checkpoint and these tested inputs.
Software predictions remain software results; future checkpoints and architectures require verification. The chord outputs stayed silent.

Consequences for training:

- **80 recurrent hidden neurons is the largest that fits** with 48 inputs under this layout. A wider
  hidden layer needs fewer inputs, sparser recurrence, or the "plane pairing" the trainer mentions,
  which has not been tried on the board.
- Outputs 0 to 11 are silent on the board as intended, so a pitch-only model costs nothing extra.
- **A new model can be screened without a board:** run `eval_modes.py`; its "chip rules" row is a
  prediction, not a physical-board measurement. Verify each new exported checkpoint before deployment.
- The one-point gap between "as trained" and the board is the price of the two kept rule
  differences. Training with the chip's rules would remove it.

Still open:

- The live loop now exists (section 6): the board stays booted and gets the latest 2 s window from
  the laptop front-end every 0.5 s.
- These windows start from a reset state, as in training. Whether carrying the state on from window
  to window would be better has not been tested.

## 6. Live integration progress — 2026-10-08

**Milestone A passed on the physical board.** `persistent_board.py` reuses one Talamo 4.7.1
connection. `persistent_check.json` records one RAM boot (5.923 s) and six separate calls:
0.783 s for the first call; 0.335–0.368 s subsequently. Random input, silence after activity,
full-scale input, and repeated input all match the zero-state reference exactly. Silence returns
zero spikes. No model or encoder rules were changed.

The generated application returns 126 frames including a flush frame. Retaining the first 125,
with all 24 channels, matches the existing `board_check.py`. SDK source confirms an acknowledged
internal-state reset after each sample; this is not an application reboot.

**Milestone B passed software checks.** The 500 ms schedule alternates 31/32 new frames,
uses the existing StreamingFrontEnd, calculates eight-frame rates within each replayed window,
and delivers each new frame to the existing selector once. Startup omits 93 frames; subsequent
frames remain contiguous. A one-second playout buffer preserves 16 ms selector/renderer pacing.
Its delay must not be described as the 0.34 s board inference time.

**Milestone C passed on the physical board with communication diagnostics enabled.**
`prerecorded_check.json` records 12 seconds of deterministic audio through the existing streaming
frontend, physical chip, score follower and violin WAV renderer: 20/20 windows match `chipref.CHIP`
exactly, 626 unique frames delivered, zero skipped updates, duplicates, or outputs >50 ms late.
Median SDK inference round trip was 0.369 s (maximum 0.399 s); median Mac-to-worker round trip,
including reference verification, was 0.390 s (maximum 0.420 s). One RAM boot took 5.891 s.
Measured audio-frame-end to renderer completion: median 1.0046 s, p95 1.0082 s, maximum 1.0248 s,
including the intentional 1 s playout delay. This is not acoustic/DAC latency.

**Startup remains a deployment risk.** Several earlier startup attempts returned an empty sample
list, including after RAM clearing and with active input first. A standalone silence-first test
failed, whereas active-first standalone checks passed twice; this does not establish silence as the
root cause. The successful worker path enables SDK communication diagnostics (`--worker-debug`),
checks deterministic active input against the reference, then checks silence/reset before accepting
audio. Three consecutive diagnostic worker starts passed (four-window, 20-window, and live-microphone runs).
The exact cause of the earlier failures is unresolved; logging is not claimed as a root-cause fix.
Warnings about protocol timeout/reset are now surfaced even though the container suppresses warnings.
There are no hidden reboots/retries during inference. Preserve startup failure records for diagnosis.

**Milestone D passed on the physical board and live MacBook microphone.** `chip_microphone.json`:
15 seconds, 26/26 windows exactly match the chip-rule reference, 814 unique frames delivered to the
existing score follower and GarageBand MIDI sink. Capture ran at 16 kHz; 131 feature frames were
nonzero, no clipping or capture errors. No skipped updates, duplicates, or outputs >50 ms late.
Boot: 5.920 s; median board round trip 0.360 s (maximum 0.384 s); median host round trip 0.381 s.
Measured audio-frame-end to renderer/MIDI-send completion: median **1.2547 s**, p95 1.2599 s,
maximum 1.2920 s, including 1.25 s playout. CoreAudio reported 0.2386 s input buffering.
This was a functional microphone test, not a singing-quality or pitch-accuracy evaluation; downstream
GarageBand/DAC/acoustic latency and instrument sound were not measured. No microphone audio saved.

**Milestone E failed at first, and the failure was reproducible (fixed, see below).** The connection stopped after about 338 windows: **measured twice, at the same count.**

| Run | Windows before it stopped | All exact until then | How it stopped |
| --- | ---: | --- | --- |
| 185 s stability run, prerecorded audio at 500 ms (`chip_stability.json`) | 337, plus 2 start-up calls | yes | `wait_events()` returned `None` inside the SDK |
| Back-to-back calls, no audio pacing (`check_call_limit.py`, `call_limit.json`) | 338 | yes | the same error |

Because the count repeats with and without audio pacing, the limit belongs to the connection, not to
timing or to the audio. It is consistent with a ceiling of about 1,024 transport events at three
events per window, which the previous session saw in its trace; that reading is not proven.
At one window every 500 ms it means **about 2 minutes 49 seconds of continuous chip operation per boot.**
After that the loop stops with an error. It does not fall back silently.
Until then nothing was wrong: median board round trip 0.364 s, no skipped or duplicated frames;
18 outputs were more than 50 ms late, all at the point of failure.

This did not concern training. It limited how long one chip session could run, whatever the model.

**The 338-window limit is fixed (board).** The cause, from the SDK's source: every window left three
operating-system event handles open, because Talamo creates a new MocktcpManager per pipeline call and
never collects its last ping replies. The native library waits on these handles with `select()`, which
only works for handle numbers below 1,024. With about 11 handles open at rest, (1,024 − 11) / 3 ≈ 338.
A repository-local `TrackedEvents` proxy (`transport_events.py`) now collects the leftover ping replies
after each window. It neither reopens the connection nor reboots the board, and changes no SDK file or
model rule.

| Evidence, all on the board | Result |
| --- | --- |
| Six deterministic windows (`persistent_event_cleanup_check.json`) | all exact, 0.359–0.443 s per call |
| 185 s live run (`chip_stability_event_cleanup.json`) | 366 of 366 windows exact; open handles constant at 11 |
| 4-minute microphone run (`results/live_2026-10-08/chip_microphone_4min_synth.json`) | 476 windows, completed; not compared with the reference |
| Whole test set on one boot (`results/A_vsP_chip_shift3/banner_testset_check.json`) | 1,451 of 1,451 windows exact |

In the 185 s run, 195 of 11,439 frames were more than 50 ms late with a 1.0 s playout delay.
Three software tests cover bounded handle reuse, unexpected leftover data and native no-completion errors.
One caution: a 1,451-window run with the SDK's communication logging switched off stalled between
window 300 and 399 (nothing was saved; cause unknown). The same run with logging on, as in every live
run, passed. That is one run each way.

**Start-up, later the same day.** The failure is in the SDK's start-up handshake between host and board
(the worker reports `waiting for sync`). The host's reply in that handshake is sent once and never
repeated, so a boot whose handshake fails stays dead. During the evening it happened in roughly one
start in three. `live_chip.py` now clears the board (RAM only) and starts a fresh worker, up to
`--start-attempts` times (default 3), before any audio is processed; every failure is printed and
counted in the metrics. The cause is not fixed.

**Independent fallback checks passed:** 12 seconds of prerecorded baseline -> score follower -> violin
WAV (747 frames); 12 seconds of MacBook microphone -> baseline -> GarageBand MIDI (747 frames,
133 nonzero-feature frames, no clipping/capture errors); and 185 seconds of prerecorded baseline
stability (11,559 frames, no duplicates, skips, or outputs >50 ms late). Baseline stability median
latency was 1.0046 s; p95 1.0082 s. These are software-only, not physical-chip results. Acoustic output
quality was not assessed. No microphone audio is saved.

Usage, interface details and reproduction commands: `yujie/chip/LIVE_CHIP.md`.
Nothing has been flashed; the model, frozen voice demo and original SDK container remain unchanged.

## 7. A_vsP_chip_shift3 on the board — 2026-10-08, evening

Simon trained `A_vsP_chip_shift3` under the chip's rules (`--chip-rules --shift 3`), which is the change
section 2 offers. Same network: 48 → 80 recurrent → 24, int8, fan-in 128. Weight file md5
`6b47d452dddcd1cdddbe8d4f2a152035`, copied from `origin/singalong-pitch-handoff` at `4917b31`.
Import: `IMPORT CHECK PASSED`, 420 output spikes in all four comparisons.

| Rules, and where measured | VocalSet songs | VocalSet exercises | DAMP |
| --- | ---: | ---: | ---: |
| **Board** (`results/A_vsP_chip_shift3/board_check.json`) | **0.577** | **0.591** | **0.378** |
| Software, chip rules, scored as the output arrives | 0.577 | 0.591 | 0.378 |
| Software, chip rules, labels shifted by the one-frame delay | 0.587 | 0.597 | 0.381 |
| Software, the laptop trainer's original rules | 0.593 | 0.619 | 0.375 |
| For comparison: `A_vocalset_P` on the board | 0.516 | 0.589 | 0.364 |

- Board against reference: all 1,451 windows identical, 91,343 spikes, none missing or extra. The chord
  outputs stayed silent.
- Build: RAM 91,360 bytes (23.23 %), largest fan-in 128, accepted by the mapper.
- Time on the board: 0.37 to 0.39 s per 2 s window, the same as for `A_vocalset_P`.
- Output activity on the board: 0.50 spikes per frame over the twelve pitch outputs (0.56 for `A_vocalset_P`).
- A second build that also shows a fixed text on the EVK display (`build_banner.py`) uses 25.59 % RAM and
  matches the reference on all 1,451 windows as well, on a single boot.

Reading: training under the chip's rules gave about 6 points on VocalSet songs on the board. The other two
sets moved by less than the seed spread Simon reports (0.02 to 0.04). The row "the laptop trainer's original
rules" is what earlier tables call "as trained"; for this model it is not how it was trained.

`snn_numpy.py`, the laptop's SNN fallback, runs the trainer's original rules. Under them this model gives
different spikes (393 against 420 on the import-check recording, F1 0.335). Use it on the chip, not in
`live.py --evidence snn:`.

All figures of this section in one place: `HARDWARE_BENCHMARK.md`.

## 8. What the live note-tracking demo asks of the next model

On 2026-10-08 the demo goal changed: the synth plays the note that is sung, in any song
(`--selector pitch`), instead of accompanying Amazing Grace. Everything here is *suggestion*, not measurement,
unless a number is given.

1. **The output has no octave.** The twelve pitch outputs name a pitch class. The live tracker takes the
   class from the chip and picks the octave on the laptop from the 48 bands. A model that also reports
   the octave would let the chip decide the whole note.
2. **There is no accuracy figure for live singing.** Dataset frame accuracy is 0.38 to 0.59. In three
   3-minute live runs on the board (speakers on, so the synth fed back into the microphone) the median
   tracked note lasted 0.30 to 0.34 s and 38 to 45 % of notes lasted under 0.3 s. Nobody scored the notes.
   One short sung take with known notes, recorded once and replayed through every model, would give a number.
3. **Steady output matters more than frame accuracy here.** The tracker averages 12 frames and switches
   when another class has led for 6 frames. A loss that rewards a margin between the best and the
   second-best output (section 4, item 5) would help this demo directly.
4. **The delay comes from the window scheme, not from the network.** Sending a 2 s window every 0.5 s
   costs about 1 s from sound to note. A model trained to keep its state from frame to frame, without
   the reset, would allow feeding single frames. This has not been tried.

## Log

- **2026-10-08, late evening** `A_vsP_chip_shift3`, trained under the chip's rules, run on the board: all
  1,451 windows identical to the reference, pitch accuracy 0.577 / 0.591 / 0.378 (section 7). The 338-window
  limit is explained and fixed; start-up handshake failures are retried automatically but not fixed
  (section 6). The demo goal changed to direct note tracking; section 8 lists what that asks of the next model.

- **2026-10-08, event cleanup** Added a repository-local pending-ping drain; six physical windows
  still match exactly. Long-run validation started with descriptor/event counters.

- **2026-10-08, Milestone E** Long run failed after 337 exact windows; native event failure, timing
  outliers, and pending-event investigation recorded. No stability success claimed.

- **2026-10-08, recheck** Milestone E failed after 337 windows; a back-to-back test stopped after 338.
  The connection has a fixed lifetime of about 338 windows. Software tests and the one-boot check
  were rerun and pass. Singing through the chip loop has still not been tried by a person.

- **2026-10-08, Milestone D** Physical-chip microphone/MIDI integration passed: 26/26 exact windows;
  measured median audio-to-renderer latency 1.2547 s. Multi-minute chip stability test started.

- **2026-10-08, Milestone C** Prerecorded physical-chip integration passed: 20/20 exact windows;
  timing and startup reliability caveat updated. Microphone hardware test started.

- **2026-10-08, live integration** Persistent connection validated on six deterministic physical-board calls;
  worker startup blocker and independent baseline tests recorded in section 6.

- **2026-10-08, evening** Timing measured (0.34 s per window). The whole test set was run on the
  board: all 1,451 windows identical to the chip-rule reference; pitch accuracy on the board
  0.516 / 0.589 / 0.364.
- **2026-10-08, later** Network built for the chip and run on the board: it fits, and one window
  matches the reference exactly (section 5). Noted while reading, not yet tested: `sim_check.py`'s
  `load_quantized` builds the pipeline without `out_neurons`, so it would create 12 outputs for a
  24-output run.
- **2026-10-08** Import fixed; rule differences identified and measured; this file written.
  An earlier verbal report said the trainer's encoder resets by subtraction. That was wrong: it came
  from the README and the `snn_numpy.py` default. The trainer's code resets to zero, like the chip.
