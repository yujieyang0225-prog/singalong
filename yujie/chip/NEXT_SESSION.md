# Start here: state of the sing-along chip work and what comes next

Updated 2026-10-08 after the live integration session was rechecked. Read this first in a new session; it is meant to be enough
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
| Persistent one-boot board adapter | **passed**, repeated deterministic windows match zero-state reference |
| Prerecorded audio -> chip -> selector/violin | **passed**, 20/20 windows exact |
| Live MacBook microphone -> chip -> GarageBand MIDI | **passed**, 26/26 windows exact |
| Multi-minute physical-chip stability (Milestone E) | **failed: the connection stops after about 338 windows, 2 min 49 s at 500 ms. Reproduced.** |

**In one sentence:** the live loop works on the chip for up to about 2 minutes 49 seconds per boot, every
window exact; it is not yet usable for an open-ended performance, and start-up is not yet reliable.

The live loop exists. Start with `yujie/chip/LIVE_CHIP.md` for exact host commands and interface semantics.
Use `--worker-debug` on the currently validated path; startup synchronization remains a known risk.

What has and has not been shown:

- Every chip result in the live loop was obtained **with `--worker-debug`** (the SDK's communication
  logging switched on). Starts without it failed three times. Why logging helps is unknown.
- "Prerecorded audio" means `data/live_chip/tones.wav`, a generated test signal, not singing.
- The microphone run was 15 s of room sound. **Nobody has sung into it, and nobody has listened to
  the violin.** Whether the accompaniment is musically right is untested.
- The whole session was rechecked on 2026-10-08: the software tests pass (5 of 5), the one-boot
  board check passes again (6 of 6 exact, `persistent_recheck.json`), and the stability failure
  repeats at the same count.

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
| `persistent_board.py`, `check_persistent.py` | Talamo 4.7.1 one-boot adapter and deterministic hardware checks |
| `board_worker.py`, `live_chip.py`, `windowing.py` | SDK worker, Mac live loop, and overlap/rate adapter |
| `LIVE_CHIP.md` | start/fallback commands, interface contract, timing definitions, recovery |
| `test_windowing.py`, `make_test_audio.py` | software checks and deterministic audio generator |
| `*.json` | original results, live measurements, and preserved failed-startup evidence |

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

## 4. Live loop and next task

Implemented Shape 2:

```
MacBook microphone (CoreAudio 16 kHz) -> existing StreamingFrontEnd (48 bands)
 -> latest 125 frames every 500 ms -> persistent SDK worker -> physical IF encoder + recurrent SNN
 -> 24-channel spikes -> eight-frame within-window rates -> unique newest frames
 -> existing melody-only score follower/HMM -> existing violin renderer -> GarageBand MIDI
```

`live_chip.py` runs on the Mac with `python`.
Only `board_worker.py` runs inside `singalong_sdk`. The Python environment now has sounddevice,
mido, and python-rtmidi. ffmpeg handles file decoding. No installed SDK files were changed.

**Start the chip demo** (from `agent-pitch-handoff`; check device IDs with `python -m sounddevice`):

```bash
python -u yujie/chip/live_chip.py   --source mic:1 --mode chip --worker-debug --sink midi:GarageBand --playout-delay 1.25
```

**Fallback:** the same command with `--mode baseline`; it does not touch the board. `--sink print`
needs no synth. GarageBand must have a suitable violin instrument selected; successful MIDI dispatch
was measured, downstream synth/DAC latency and audible musical quality were not assessed.

Important measured distinctions:

- Standalone persistent test: boot 5.923 s; first call 0.783 s; subsequent 0.335–0.368 s.
- Prerecorded live integration: 20/20 windows exact; median SDK round trip 0.369 s; audio-frame-end
  to renderer completion 1.0046 s with intentional 1 s playout.
- Live microphone: 26/26 windows exact; median SDK round trip 0.360 s; audio-frame-end to MIDI
  renderer completion 1.2547 s with intentional 1.25 s playout. 814 unique frames delivered.
- No live-run skipped updates, duplicates, or outputs >50 ms late in those two checks.
- Startup frames 0..92 are deliberately omitted. Each subsequent frame is delivered once;
  a short tail before the next complete scheduled update is counted separately on finite recordings.

**Startup caveat:** early worker starts returned no complete sample, including after RAM clearing.
An active-first self-check is followed by silence to verify reset. Silence-first failed in one standalone
probe, but active-first worker startup also failed, so the cause is not established. The measured
worker path with `--worker-debug` passed three consecutive starts. Do not claim logging fixes the root
cause. The wrapper now surfaces SDK timeout/reset warnings, fails closed, and never reboots between
windows. Failed-startup JSON files are retained.

**Milestone E result.** The connection stops after about 338 windows: **measured twice, at the same count.**

| Run | Windows before it stopped | All exact until then | How it stopped |
| --- | ---: | --- | --- |
| 185 s stability run, prerecorded audio at 500 ms (`chip_stability.json`) | 337, plus 2 start-up calls | yes | `wait_events()` returned `None` inside the SDK |
| Back-to-back calls, no audio pacing (`check_call_limit.py`, `call_limit.json`) | 338 | yes | the same error |

Because the count repeats with and without audio pacing, the limit belongs to the connection, not to
timing or to the audio. It is consistent with a ceiling of about 1,024 transport events at three
events per window, which the previous session saw in its trace; that reading is not proven.
At one window every 500 ms it means **about 2 minutes 49 seconds of continuous chip operation per boot.**
After that the loop stops with an error. It does not fall back silently.

**Next steps, in this order:**

1. **Decide how to live with the 338-window limit.** Options, none implemented:
   - keep each performance under about 2 min 40 s and restart the worker between performances
     (a restart costs about 7 s: 5.9 s boot plus the first call);
   - a planned reconnect before the limit, with an explicit, logged switch to the baseline for the gap;
   - a longer update period (1 s gives about 5 min 38 s, with updates half as often);
   - find out whether the host side of the connection can be reopened without rebooting the board.
   The installed SDK must not be modified.
2. **Start-up reliability.** Run 10 deliberate start/stop cycles with `--worker-debug` and count failures.
3. Rehearse with a person singing
and confirm the actual GarageBand violin sound through headphones. Check macOS input mode/voice
isolation manually. A 15-second ambient microphone test is not a musical-quality or calibrated
sample-rate test. Test startup reliability across deliberate restart cycles before relying on the chip
on stage; keep the explicit baseline switch ready. Do not reduce the 500 ms period based on the
0.34 s batch-throughput measurement.

The model mismatch, model weights, reset semantics, and old complete test results are unchanged.
See `LIVE_CHIP.md` for protocol/reset source locations, timestamps, readout behavior, performance
metric definitions, multi-minute commands, and explicit RAM-only recovery. Every new checkpoint
requires verification; software reference predictions are not physical-board measurements.

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
