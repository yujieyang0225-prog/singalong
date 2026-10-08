# Feedback from the chip side for the next training session

Model looked at: `handoff/runs/A_vocalset_P/weights_int8.npz` (md5 `64623170…`).
Branch `yujie/singalong-chip`, started from `singalong-pitch-handoff` at `2a612c3`.
Kept up to date as the chip work proceeds; the log at the end says what changed and when.

Everything here was measured in **Talamo 4.7.1's software model** of the chip and in an integer
reference. Nothing here is a board measurement yet.

## Short version

| | |
| --- | --- |
| Can the trained weights be put into Talamo correctly? | **Yes, since 2026-10-08.** The import script was broken; it is fixed and its check passes. |
| Does Talamo's model then behave like the trained network? | Not spike for spike: two rules differ (below). **By decision these are kept as they are.** |
| What do those differences cost? | About 1 point of pitch accuracy at most. No retraining is needed because of them. |
| What should the next training session change? | Nothing is required. Section 4 lists what would make the model safer on the chip. |

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

## 5. Open on the chip side

- Build the network for the chip and confirm it fits (104 neurons, fan-in 128).
- Board against the reference on the handoff test data.
- Round-trip time for a 125-frame window; live loop with the laptop front-end.

## Log

- **2026-10-08** Import fixed; rule differences identified and measured; this file written.
  An earlier verbal report said the trainer's encoder resets by subtraction. That was wrong: it came
  from the README and the `snn_numpy.py` default. The trainer's code resets to zero, like the chip.
