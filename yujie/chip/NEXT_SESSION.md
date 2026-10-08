# Start here: state of the sing-along chip work and what comes next

Written 2026-10-08, end of the session. Read this first in a new session; it is meant to be enough
to continue without the earlier conversation. Paths are relative to the workspace folder
`~/TUM/NeuroTUM` unless they start with `/workspace` (inside a container).

## 1. Where things stand

**Goal (Phase B):** run the team's pitch network on the Pulsar chip, with the audio front-end on a
laptop ("shape 2"), for the sing-along demo. The pitch is on Friday 9 October, 10:00.

| Step | State |
| --- | --- |
| Import the torch-trained weights into Talamo | **done**, check passes |
| Explain the difference between trainer and chip model | **done**, two rule differences, kept by decision, cost about 1 point |
| Build for the chip, confirm it fits | **done**, 104 neurons, fan-in 128 (the limit), RAM 23 % |
| Chip equals reference | **done**, all 1,451 test windows identical on the board |
| Time per window | **done**, 0.34 s per 2 s window, plus 6.6 s boot per call |
| **Live loop with the laptop front-end** | **not started. This is the next task.** |

Numbers and explanations: `yujie/chip/TRAINING_FEEDBACK.md` (kept current, pushed after every change).

**Model:** `handoff/runs/A_vocalset_P/weights_int8.npz`, pitch-only, 48 bands -> 80 recurrent -> 24
outputs (12 to 23 are pitch classes C to B; 0 to 11 are silent). Pitch accuracy on the board:
0.516 VocalSet songs, 0.589 exercises, 0.364 DAMP; chance is about 0.08.
The folder `grp-gold-neurotum/audio/B_damp_lr0.03` is an **older, wrong** model; do not use it.

## 2. Where everything is

| What | Where |
| --- | --- |
| Working copy for this work | `agent-pitch-handoff/`, branch `yujie/singalong-chip`, pushed to `nc-fortiss/MNH26SVR` |
| Chip-side code and results | `agent-pitch-handoff/yujie/chip/` |
| Teammate's handoff (model, test data, notes) | `agent-pitch-handoff/handoff/`, from branch `singalong-pitch-handoff` (do not commit there) |
| Team code (front-end, selector, trainer) | `agent-pitch-handoff/jan/singalong/`, `simon/` |
| Imported parameters and the built chip application | `agent-pitch-handoff/data/runs/A_vocalset_P_talamo/` (git-ignored: rebuild if missing) |
| The voice demo, frozen | `demo-voice-v1/` (outside any repository) |
| Voice demo source and notes | `grp-gold-neurotum/audio/scene-classification/`, branch `voice-baseline`, **local only**, tags `demo-voice-v1`, `demo-voice-v1.1` |
| Team vault | `MNH26SVR/` on `main`: never commit or push there |

Files in `yujie/chip/`:

| File | Purpose |
| --- | --- |
| `TRAINING_FEEDBACK.md` | findings for whoever trains the next model; the record of results |
| `chipref.py` | integer reference with the trainer's rules and the chip's rules; under `CHIP` it equals the board |
| `diagnose_import.py` | stage-by-stage comparison Talamo against reference |
| `eval_modes.py` | pitch accuracy of a model under both rule sets (no board) |
| `build_chip.py` | builds the host-fed chip application |
| `board_check.py` | board against reference (`--timing` for time per window) |
| `*.json` | results of the three scripts above |

One file outside `yujie/` was changed on this branch: `jan/singalong/Pulsar/IF_DNR/import_torch_weights.py`
(the import fix). Before the fix every import ran random weights.

## 3. How to get going again

**Containers.** Two SDK containers (image `innatera_sdk:2.3.0`, Talamo 4.7.1):

- `innatera_sdk_container`: mounts only `grp-gold-neurotum`. Used for the voice demo. Leave it alone.
- `singalong_sdk`: mounts `agent-pitch-handoff` at `/workspace` and `grp-gold-neurotum` read-only at
  `/opt/grp-gold-neurotum`. Used for everything here. If it is gone, recreate it:

```bash
docker run -d --name singalong_sdk --group-add innatera \
  -v ~/TUM/NeuroTUM/agent-pitch-handoff:/workspace \
  -v ~/TUM/NeuroTUM/grp-gold-neurotum:/opt/grp-gold-neurotum:ro \
  -e PYTHONPATH=/opt/grp-gold-neurotum/inertial-measurement-unit/wristbandgesturerecognition:/workspace/jan/singalong \
  -e PYTHONWARNINGS=ignore -e MPLCONFIGDIR=/tmp/mpl -e EVK_BRIDGE=host.docker.internal:1700 \
  -w /workspace/jan/singalong innatera_sdk:2.3.0 sleep infinity
docker exec singalong_sdk pip install --user -q seaborn "torchsummary==1.5.1" "pyzenodo3==1.0.2" pretty_midi pytest mido
```

**Board.** Pulsar EVK on USB, boot switch on UART, `evk-bridge` running on the Mac on port 1700
(`evk-bridge -i <BOARD_ID> --listen 0.0.0.0 --port 1700 --gdb-port 1701`). One client at a time.
Everything is RAM boot; nothing has ever been flashed, and that rule stands.

**Commands** (inside `singalong_sdk`, from `/workspace/jan/singalong`):

```bash
# 1. import the weights (no board); must end with IMPORT CHECK PASSED
python3 Pulsar/IF_DNR/import_torch_weights.py ../../handoff/runs/A_vocalset_P/weights_int8.npz \
    --out ../../data/runs/A_vocalset_P_talamo --check ../../handoff/testdata/damp250
# 2. expected accuracy on the chip (no board)
python3 ../../yujie/chip/eval_modes.py ../../handoff/runs/A_vocalset_P
# 3. build the chip application (no board; refuses to overwrite an existing app folder)
python3 ../../yujie/chip/build_chip.py ../../data/runs/A_vocalset_P_talamo \
    --npz ../../handoff/runs/A_vocalset_P/weights_int8.npz --check ../../handoff/testdata/damp250
# 4. board: time per window, then the whole test set (about 10 minutes)
python3 ../../yujie/chip/board_check.py ../../data/runs/A_vocalset_P_talamo --npz ../../handoff/runs/A_vocalset_P/weights_int8.npz --timing
python3 ../../yujie/chip/board_check.py ../../data/runs/A_vocalset_P_talamo --npz ../../handoff/runs/A_vocalset_P/weights_int8.npz --crops 512
```

For a new model from the trainer: put its run folder under `handoff/runs/`, then run 1, 2, 3, 4 with
its name. Step 2 alone already tells what the board will give.

**Voice demo, any time:** `cd ~/TUM/NeuroTUM/demo-voice-v1 && ./run.sh` (about 10 s to load). It replaces
whatever is on the board; the sing-along application has to be booted again afterwards.

## 4. The next task: the live loop

Design agreed in the session:

```
laptop microphone -> jan/singalong front-end (48 bands, 16 ms frames) -> sliding window of 125 frames (2 s)
   -> board (encoder + recurrent network) -> output spikes per frame -> discard the first 31 frames
   -> rate over the last 8 frames -> jan/singalong selector (--use melody) -> violin
```

Steps, in order:

1. **Keep the board booted between windows.** A normal `pipe(...)` call boots every time (6.6 s).
   The recipe in the repository skill `pulsar-evk-app-bringup` uses private Talamo functions
   (`_assemble_embedded_pipeline`, `get_board_connection`, `boot_application`,
   `_configure_embedded_application_board_connection`, `_execute_pipeline`). Untried here. Measure the
   per-call time; expect about 0.34 s.
2. **Sliding window.** Every 0.35 to 0.5 s send the latest 125 frames. The network starts each window
   from a reset state, exactly as in training, and the first 31 frames (0.5 s) are warm-up.
3. **Connect to the team's host app.** `jan/singalong/live.py` reads evidence from several sources;
   `singalong/evidence.py` has a "board JSON lines" source. Either emit that format or add a source
   that calls the board directly. The model is pitch-only: use `--use melody`.
4. **Fallback.** One switch to the team's baseline evidence (`--evidence baseline`), so there is a
   demo if the board path fails on stage.
5. **End-to-end check** with a recording first (`--source wav:...`), then the laptop microphone.

Things to decide or watch:

- **Lag.** About 0.34 s processing plus the update interval, plus 16 ms from the chip's layer delay.
- **Laptop microphone:** switch off macOS voice isolation and noise reduction; fix the input device.
- **Output is sparse:** about 0.56 spikes per frame over the twelve pitch outputs, rate over 8 frames,
  so ties are common. The selector's smoothing matters more than any single frame.
- **Carrying state between windows** instead of resetting was not tested.
- **A new model may arrive.** A search around A_vocalset_P was running on the cluster; check
  `handoff/README.md` on `origin/singalong-pitch-handoff` for a new run before the demo.
- **Fan-in is at the limit** (128). A wider hidden layer will not fit as is.

## 5. Known quirks of this setup

- **"waiting for sync" / empty result.** A Talamo boot can fail when the previous Talamo application
  is still running on the board; it shows as an `IndexError` in `sparse_spikes_raster_to_dense` or
  as zero results. RAM-booting any image with `c1util.py` first clears it. `board_check.py` does this
  automatically and retries.
- **One client per board.** A recorder, a notebook or a console holding the bridge blocks a boot.
- **Stopping consoles:** use SIGINT or SIGTERM, never SIGKILL. A process started in the background
  with no terminal ignores SIGINT; use SIGTERM.
- **`data/` is git-ignored** in the team repository: imported parameters, the built application and
  the saved board spikes live only on this Mac.
- **Packages in `singalong_sdk`** were installed with `pip --user`; they are lost if the container is removed.
- **Builds are slow** (1 to 2 minutes): the SDK image is x86 running under emulation on this Mac.

## 6. Rules the user set

- RAM boot only; never write flash.
- Never commit or push to `main` of either repository, and never to a teammate's branch.
  Work on `yujie/*` branches.
- After every update to `TRAINING_FEEDBACK.md`: commit, push `yujie/singalong-chip`, and say where the file is.
- The two trainer-versus-chip rule differences are **kept**; document, do not "fix" the model.
- The voice demo must stay runnable at any time (`demo-voice-v1/`).
- Recorded audio stays out of git; `audio_data/data_mic/` is read-only for training work.
- Say plainly whether a number is measured on the board, software, or simulated.

## 7. Loose ends outside Phase B

- `grp-gold-neurotum`, branch `voice-baseline` (voice demo source, notes, guide) is **not pushed** to
  the Innatera repository; only `live-mic-scene` is. A snapshot is on the team repository as
  `yujie/live-mic-voice`, without the last display changes.
- The Innatera repository's `main` gained a "live microphone" example on 8 October; not merged locally.
- Another session left `tests/record_playback.py` untracked in `grp-gold-neurotum/audio/scene-classification/`.
- Never done for the voice demo: a known-tone check of the sample rate, microphone sensitivity and clipping.
