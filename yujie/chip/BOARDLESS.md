# Keep improving the sing-along without the board

Written 2026-10-09, after the pitch; section 3 updated the same evening when the simulated chip was built. Paths are relative to `~/TUM/NeuroTUM/agent-pitch-handoff` unless they start
with `/workspace` (inside the `singalong_sdk` container). Python on the Mac means
`python`.

## Short answer

**Yes, most of the work does not need the board.** What does not: the final proof on the chip, and anything about
timing. A new network can be tried live, with the tracker, synth and page, on a simulated chip: `--mode reference`
(section 3, built on 2026-10-09 and merged into `yujie/singalong-chip`, locally).

| Work | Board needed? | How |
| --- | --- | --- |
| Train a new model | no | Simon's trainer (`jan/singalong`, `simon/`) on the cluster; weights come back as `weights_int8.npz` |
| Import the weights into Talamo and check them | no | container, section 2 step 1 |
| Know what accuracy the chip will give | no | `eval_modes.py`, section 2 step 2 |
| Know whether it fits and builds for the chip | no | `build_chip.py`, `build_banner.py`, section 2 steps 3 and 4 |
| Note tracking, synth, the live page, the front-end | no | `--mode baseline`, tests, recordings |
| Run a network in the live loop | no | `--mode reference`, a simulated chip (section 3) |
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

## 3. A network in the live loop without the board: `--mode reference`

`live_chip.py` and `live_view.py` have three modes: `chip` (the board, through `board_worker.py` in the container),
`baseline` (the laptop's own pitch estimate, **no network**) and, since 2026-10-09, `reference`: a simulated chip.
The window goes to `reference_worker.py`, a local process on the same JSON-lines protocol as the board worker, which
answers with `chipref.run(npz, x, **chipref.CHIP)`. Everything else in the loop is the chip path: windows, stop
rules, tracker, synth, page. Options and limits: `LIVE_CHIP.md`, "Simulated chip".

```bash
python -u yujie/chip/live_view.py --source "mic:MacBook Pro Microphone" --mode reference --model <NAME> \
    --selector pitch --playout-delay 1.25 --sink synth
```

`<NAME>` is a folder under `handoff/runs/` with `weights_int8.npz`: put a new model there and it plays. It needs no
import and no build.

**Acceptance criteria and what was measured (2026-10-09, no board; commits `72344f4` to `f1bd49f`):**

| # | Criterion | Result |
| --- | --- | --- |
| 1 | Worker spikes equal the reference | 0 differences on 256 windows (216 real voiced DAMP windows and 40 random ones) for both models. A worker with the training rules fails this test |
| 2 | Same notes as the physical chip | The 45 note lines of the board log of the Happy Birthday recording match with `A_vsP_chip_shift3`, 0 differences; the older `A_vocalset_P` prints 42 lines that differ, so the comparison can fail |
| 3 | The chip path is untouched | All 44 earlier tests pass; `board_worker.py`, `fake_worker.py`, `ChipClient` and the chip-mode error texts are byte-identical; `--mode baseline` on `tones.wav` still prints the documented notes |
| 4 | It says it is simulated | Page tag "SIMULATED CHIP · no board" (read in the built-in browser; "Pulsar C1" appears nowhere); metrics `simulated: true`; `exact_reference` null in every window |
| 5 | Timing | Median round trip 0.394 s at the default, under 50 ms with `--sim-latency 0`; 0 slow round trips |
| 6 | Failures | A worker that dies gives "Board worker exited"; one slower than the window timeout gives "Board timeout"; neither leaves a process; end of input, a close request and SIGTERM stop the worker quietly |
| 7 | No Docker | The run succeeds with a `docker` on `PATH` that fails loudly and is never called; the same driver in chip mode does call it |
| 8 | Stability | Two 3-minute runs: 356 windows each, 0 late frames, 0 slow round trips, 0 errors, identical output (91 note lines) |

**Limits, said plainly:** the round trip is a configured wait (default 0.38 s, the board's median, not measured
here), so timing, late frames and the start-up handshake of a reference run mean nothing for the chip. The
microphone has not been tried in this mode. The match with the board rests on the reference having matched the chip on
every window compared (section 1) and on the one recording above.

**What this makes possible:** every item of section 4 that concerns the network can be tried live, with the microphone,
without the board. A chip run and a reference run differ only in the worker, which also turns the drift question
(issue 11) into a clean comparison.

## 4. Work that needs no board, from the open points

From `NEXT_SESSION.md`; none of it has been started.

| Idea | Why no board | First check |
| --- | --- | --- |
| Sustained monophonic synth / steadier notes (`NEXT_SESSION.md` section 6, item 5; the feasibility report estimated 150 to 250 ms extra delay on note changes) | The tracker and synth run on the laptop | host-only note-sequence tests, then the known 2 s tones |
| Measure live note tracking (issue 2: median note lasted 0.30 to 0.34 s) | Needs recordings and a scorer, not the chip | recordings through `--mode reference` (section 3) |
| Is the microphone timestamp drift (issue 11) caused by the board? | `--mode reference` with the microphone has no board and is otherwise the chip loop | one 20-minute reference run with the microphone and the same options as the 20-minute chip run (about 0.03 s of drift per minute). **Untested idea** |
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
