# Keep improving the sing-along without the board

Written 2026-10-09, after the pitch. Paths are relative to `~/TUM/NeuroTUM/agent-pitch-handoff` unless they start
with `/workspace` (inside the `singalong_sdk` container). Python on the Mac means
`python`.

## Short answer

**Yes, most of the work does not need the board.** What does not: the final proof on the chip, and anything about
timing. One piece is missing and has to be built first: running a *new network* inside the live loop (section 3).

| Work | Board needed? | How |
| --- | --- | --- |
| Train a new model | no | Simon's trainer (`jan/singalong`, `simon/`) on the cluster; weights come back as `weights_int8.npz` |
| Import the weights into Talamo and check them | no | container, section 2 step 1 |
| Know what accuracy the chip will give | no | `eval_modes.py`, section 2 step 2 |
| Know whether it fits and builds for the chip | no | `build_chip.py`, `build_banner.py`, section 2 steps 3 and 4 |
| Note tracking, synth, the live page, the front-end | no | `--mode baseline`, tests, recordings |
| Run a new *network* in the live loop | **not yet possible** | section 3, one small script to build |
| Whole test set on the chip, RAM %, round trip, drift, the start-up handshake | **yes** | section 5, once at the end |

## 1. Why a software result is a chip result

`chipref.py` computes the network in integers with the chip's rules (`chipref.CHIP`). It reproduces Talamo's
software model of the C1 spike for spike (`diagnose_import.py`). On the physical chip the output was identical to this
reference on 1,451 of 1,451 test windows (plain and banner application) and on 2,396 of 2,396 windows of the
20-minute run (both measured on the board, 2026-10-08 and 09). So accuracy computed with `chipref` is the accuracy the
board gives, for the network itself.

What the reference does **not** tell: round-trip time, late frames, microphone timestamp drift, the start-up handshake,
RAM use of the built application, and anything about sound and room.

## 2. The loop for a new model, no board

`M` is the model's folder name under `handoff/runs/`. Steps 1, 3 and 4 run in the container from
`/workspace/jan/singalong`; step 2 also runs on the Mac.

| Step | Command | Result to look for |
| --- | --- | --- |
| 1. Import and check | `python3 Pulsar/IF_DNR/import_torch_weights.py ../../handoff/runs/$M/weights_int8.npz --out ../../data/runs/${M}_talamo --check ../../handoff/testdata/damp250` | ends with `IMPORT CHECK PASSED` |
| 2. Accuracy on the chip | `python3 ../../yujie/chip/eval_modes.py ../../handoff/runs/$M` | row "CHIP, scored as it arrives" = what the board gives |
| 3. Plain application, does it fit | `python3 ../../yujie/chip/build_chip.py ../../data/runs/${M}_talamo --npz ../../handoff/runs/$M/weights_int8.npz --check ../../handoff/testdata/damp250` | builds, RAM figure (23.23 % for the validated model). Refuses to overwrite an existing app folder |
| 4. Application with the display text | `python3 ../../yujie/chip/build_banner.py ../../data/runs/${M}_talamo --npz ../../handoff/runs/$M/weights_int8.npz` | builds (25.59 % for the validated model) |

**`eval_modes.py` overwrites the tracked file `yujie/chip/chip_expectation.json`** with the model it was run on. Copy
the file away first and put it back afterwards, or the repository shows a change that is not yours.

**Checked on 2026-10-09 with no board attached (measured, on this Mac, host `snn` Python):**

- `eval_modes.py` on `handoff/runs/A_vsP_chip_shift3` (finished in under three minutes) gave for "CHIP, scored as it arrives"
  **0.577 / 0.591 / 0.378** (VocalSet songs / exercises / DAMP). The board gave exactly these numbers
  (`NEXT_SESSION.md`, section 1). Full table:

  | Rules | songs | exercises | DAMP |
  | --- | ---: | ---: | ---: |
  | As trained (`torchsnn`) | 0.593 | 0.619 | 0.375 |
  | Recurrent as on chip, no layer delay | 0.595 | 0.604 | 0.383 |
  | **Chip, scored as it arrives** | **0.577** | **0.591** | **0.378** |
  | Chip, labels shifted by the one-frame delay | 0.587 | 0.597 | 0.381 |
  | `snn_numpy.py` default | 0.581 | 0.618 | 0.384 |

- The software tests: 40 tests on the Mac (`test_windowing` 5, `test_transport_events` 3, `test_worker_failure` 3,
  `test_shutdown` 6, `test_slow_round_trips` 5, `test_live_view` 18) and 4 in the container (`test_board_lock`),
  44 in all, all passed with the board idle.
- The laptop-only loop with the live page on `data/live_chip/tones.wav` (command in `NEXT_SESSION.md`, section 7):
  the console showed the notes 60, 64, 67, 69, 67, 64, 62, 60, 64 and the program stopped by itself.

Steps 1, 3 and 4 were done for both models on 2026-10-08 (`NEXT_SESSION.md` at commit `50ea178`, section 4) and were
not repeated today; that handoff says they need no board, and building "needs no board" is also stated in the
docstring of `build_chip.py`.

## 3. The missing piece: a network in the live loop without the board

The loop has two modes (`live_chip.py --mode`): `chip` talks to `board_worker.py` in the container, `baseline` uses
the laptop's own pitch estimate and **no network at all**. `fake_worker.py` only answers with silence, for the failure
tests. So today a new network can be scored (step 2) but cannot be *played* through the tracker, the synth and the
page without a board.

**Proposal, not built (awaiting approval, as the working rules require):** `reference_worker.py`, a drop-in for
`board_worker.py` on the same JSON-lines protocol: it answers `hello` and `ready`, takes a request
`{id, features}` with `features` of shape 125 x 48, and returns `{type: result, id, spikes}` with `spikes` of shape
125 x 24, computed by `chipref.run(npz, x, **chipref.CHIP)`. `board_worker.py` already imports `chipref` and uses it for
`--verify-reference`, so the computation exists. It would run on the Mac, so no container is needed.

Proposed acceptance criteria:

1. For the test windows, its spikes are identical to the reference spikes `board_worker.py --verify-reference` compares
   against (0 differences).
2. The six test files and the container test still pass; a new test covers the worker.
3. Without the new option, `live_chip.py` and `live_view.py` behave exactly as before; the default stays.
4. It reports a configurable fake round-trip time and says in the page and the metrics that the chip is *simulated*,
   so nobody quotes its timing as a measurement.

With it, everything in section 4 that concerns the network can be tried live, with the microphone, with no board.

## 4. Work that needs no board, from the open points

From `NEXT_SESSION.md`; none of it has been started.

| Idea | Why no board | First check |
| --- | --- | --- |
| Sustained monophonic synth / steadier notes (`NEXT_SESSION.md` section 6, item 5; the feasibility report estimated 150 to 250 ms extra delay on note changes) | The tracker and synth run on the laptop | host-only note-sequence tests, then the known 2 s tones |
| Measure live note tracking (issue 2: median note lasted 0.30 to 0.34 s) | Needs recordings and a scorer, not the chip | recordings through the reference worker of section 3 |
| Is the microphone timestamp drift (issue 11) caused by the board? | `--mode baseline` with the microphone has no board | one 20-minute baseline run with the microphone; compare with the 20-minute chip run (about 0.03 s of drift per minute). **Untested idea** |
| Page readability (issue 13) | Browser and projector only | open `live_view.py` on tones in Safari on the projector |
| Next training session | Cluster | `TRAINING_FEEDBACK.md`, section 4 and 7 |

## 5. What needs the board again (do it once at the end)

Only when something changed that the reference cannot judge: a new model for the demo, or any change on the chip
side. In this order, with the board on USB, boot switch on UART, `evk-bridge` running, nothing else on the board:

1. `PITCH_DAY.md`, "At the venue": bring-up and the checks.
2. Whole test set on the chip, plain application (about 10 minutes): `board_check.py ... --crops 512`. It overwrites
   the tracked `board_check.json`; copy it away first.
3. Whole test set through the banner application on one boot (about 10 minutes): `check_banner.py ... --testset --debug-logs`.
4. One 3-minute microphone run with the demo command (`NEXT_SESSION.md`, section 4).

Accept a model only if items 2 and 3 of this list show every window identical to the reference. RAM boot only,
never flash.

## 6. What makes all this possible, and where it is backed up

| Needed | Where | If lost |
| --- | --- | --- |
| `chipref.py`, the tests, the live loop | this folder, in git | `git log` |
| Model weights and the test data | `handoff/runs/`, `handoff/testdata/` (in git, from Simon's branch) | Simon's branch `singalong-pitch-handoff` |
| Imported models and built applications | `data/runs/` (**git-ignored**, only on this Mac) | steps 1 to 4 above, or `archive/2026-10-09_wrapup_backup/pitch-handoff_local_state.tar.gz` |
| Python env `snn` (3.11, torch 2.14.1, numpy 2.4.6, sounddevice 0.5.6) | `~/miniforge3/envs/snn`; saved as `yujie/chip/snn.environment.yml` (exported 2026-10-09) | `conda env create -f yujie/chip/snn.environment.yml` (a download; not tried) |
| Talamo 4.7.1 | Docker image `innatera_sdk:2.3.0` (5.4 GB) | from Innatera's SDK package; `grp-gold-neurotum/` has the docs and the bridge wheel |
| The trainer | `jan/singalong/`, `simon/` | in git |

### Recreate the container

`singalong_sdk` mounts this folder at `/workspace` and `grp-gold-neurotum` read-only. If it is gone:

```bash
docker run -d --name singalong_sdk --group-add innatera \
  -v ~/TUM/NeuroTUM/agent-pitch-handoff:/workspace \
  -v ~/TUM/NeuroTUM/grp-gold-neurotum:/opt/grp-gold-neurotum:ro \
  -e PYTHONPATH=/opt/grp-gold-neurotum/inertial-measurement-unit/wristbandgesturerecognition:/workspace/jan/singalong \
  -e PYTHONWARNINGS=ignore -e MPLCONFIGDIR=/tmp/mpl -e EVK_BRIDGE=host.docker.internal:1700 \
  -w /workspace/jan/singalong innatera_sdk:2.3.0 sleep infinity
docker exec singalong_sdk pip install --user -q seaborn "torchsummary==1.5.1" "pyzenodo3==1.0.2" pretty_midi pytest mido
```

(From `50ea178:yujie/chip/NEXT_SESSION.md`, section 4, where it also was; unchanged. The container keeps running
after the pitch; stopping it with `docker stop singalong_sdk` frees memory and loses nothing on disk.)
