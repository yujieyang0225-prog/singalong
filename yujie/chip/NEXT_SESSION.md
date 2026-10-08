# Start here: state of the sing-along chip work and what comes next

Updated 2026-10-08, late evening, after direct note tracking, the Python synth, the live view, Simon's
chip-trained model and the display banner were checked on the board. Read this first in a new session; it
is meant to be enough to continue without the earlier conversation. Paths are relative to the workspace
folder `~/TUM/NeuroTUM` unless they start with `/workspace` (inside a container).

## 1. Where things stand

**Goal (changed on 2026-10-08):** sing a note, and a synth plays that note, in any song, with the pitch
network running on the Pulsar chip and the audio front-end on a laptop ("shape 2"). Accompaniment that
follows Amazing Grace is no longer the goal; it still runs with `--selector score`.
The pitch is on Friday 9 October, 10:00.

| Step | State |
| --- | --- |
| Import, rule differences, build, chip equals reference (`A_vocalset_P`) | **done** earlier, all 1,451 test windows identical on the board |
| 338-window limit of one board session | **fixed**: cause found, leftover events are collected; longest single boot 1,451 windows, all exact |
| Start-up handshake failure ("waiting for sync") | **not fixed**; automatic clear-and-retry, about one start in three needs it |
| One user of the board at a time | **done**: board lock |
| Hung or dead worker | **done**: detected in 3 s, run stops with an error, never hangs |
| Notes released and metrics kept however a run ends | **done** |
| Sound without GarageBand | **done**: built-in Python synth; the user has heard it |
| Live view in the browser | **done** |
| Direct note tracking (`--selector pitch`) | **works on known tones**, chip and baseline; accuracy on live singing **not measured** |
| Simon's chip-trained model `A_vsP_chip_shift3` | **verified on the board**: 1,451 of 1,451 identical; 0.577 / 0.591 / 0.378 |
| Fixed text on the board's display | **verified**: 1,451 of 1,451 on one boot; the user saw the text |

**In one sentence:** the demo runs end to end on the chip with Simon's model, a display banner, a live
view and its own sound; how well it tracks a person's singing has not been measured, and the note comes
about one second after the voice.

What has and has not been shown:

- **Heard and seen by the user:** sound from the Python synth, and the text on the board's display.
  The user's impressions of the live note tracking (steady? right notes?) were not recorded.
- **No accuracy figure for live singing.** Dataset frame accuracy on the board is 0.38 to 0.59.
  In three 3-minute live runs the median tracked note lasted 0.30 to 0.34 s; nobody scored the notes.
- **Live runs used the MacBook speakers**, so the synth fed back into the microphone.
- **Every long chip session that passed had the SDK's communication logging on** (`--worker-debug`).
  One 1,451-window run with logging off stalled between window 300 and 399; cause unknown.
- **GarageBand was never heard.** The project it reopens has one audio track and no instrument, so MIDI
  had nothing to play. The MIDI path still exists (`--sink midi:GarageBand`); it is not the demo path.

Numbers for the training team: `yujie/chip/TRAINING_FEEDBACK.md`. Benchmark tables for the presentation:
`yujie/chip/HARDWARE_BENCHMARK.md`. How to run the loop and what its options mean: `yujie/chip/LIVE_CHIP.md`.
How it got here, with the decisions and mistakes: `yujie/chip/PROCESS.md`.

**Models.** Both are pitch-only, 48 bands -> 80 recurrent -> 24 outputs (12 to 23 are pitch classes C to B).

| Model | Trained under | Pitch accuracy on the board (songs / exercises / DAMP) | Used when |
| --- | --- | --- | --- |
| `A_vsP_chip_shift3` | the chip's rules | 0.577 / 0.591 / 0.378 | `--model A_vsP_chip_shift3` (the presentation) |
| `A_vocalset_P` | the laptop trainer's rules | 0.516 / 0.589 / 0.364 | default when `--model` is left out |

Chance is about 0.08. The folder `grp-gold-neurotum/audio/B_damp_lr0.03` is an **older, wrong** model; do not use it.

## 2. The demo

From `agent-pitch-handoff` on the Mac. Board on USB, boot switch on UART, `evk-bridge` running (section 4).

```bash
python -u yujie/chip/live_view.py --source mic:1 --mode chip --worker-debug --model A_vsP_chip_shift3 --banner --selector pitch --playout-delay 1.25 --start-attempts 5 --sink synth
```

Then open http://127.0.0.1:8765. Start-up takes about 25 s, or about 35 s more for each start that needs
a retry. Stop with Ctrl-C.

- These settings passed a 25 s run on a recording with silent output (46 of 46 windows exact, 0 late frames,
  `results/live_2026-10-08/final_smoke.json`). With the microphone and the synth, the same settings ran
  without `--banner` and with a 1.0 s delay. This exact line has not been run with a person singing.
- **Laptop only, no board:** replace `--mode chip --worker-debug --model A_vsP_chip_shift3 --banner
  --playout-delay 1.25 --start-attempts 5` with `--mode baseline --playout-delay 0.1`. The note then comes
  about 0.1 s after the voice. The display banner, if loaded, keeps saying the chip is running; power-cycle
  the board before showing this mode.
- **Use headphones** if possible, and held vowels at a slow pace.
- **If the page shows an error:** it names the cause. Start again; nothing has to be cleaned up by hand.
  If it says the board is in use, another program holds it; stop that one first.

## 3. Where everything is

| What | Where |
| --- | --- |
| Working copy for this work | `agent-pitch-handoff/`, branch `yujie/singalong-chip`, remote `nc-fortiss/MNH26SVR` |
| Chip-side code and results | `agent-pitch-handoff/yujie/chip/` |
| Teammate's handoff (models, test data, notes) | `agent-pitch-handoff/handoff/`, from branch `singalong-pitch-handoff` (do not commit there) |
| Team code (front-end, selector, trainer) | `agent-pitch-handoff/jan/singalong/`, `simon/` |
| Imported parameters and built chip applications | `agent-pitch-handoff/data/runs/<model>_talamo/` (git-ignored: rebuild if missing) |
| Run logs and metrics of the day | `agent-pitch-handoff/data/live_chip/` (git-ignored); the important ones are copied to `yujie/chip/results/` |
| The voice demo, frozen | `demo-voice-v1/` (outside any repository) |
| Voice demo source and notes | `grp-gold-neurotum/audio/scene-classification/`, branch `voice-baseline`, **local only** |
| Team vault | `MNH26SVR/` on `main`: never commit or push there |

Files in `yujie/chip/`:

| File | Purpose |
| --- | --- |
| `live_view.py` | **the demo's entry point**: runs `live_chip.py` unchanged and serves the live page |
| `live_chip.py` | the loop: capture, windows, board worker, selector, renderer, shutdown, metrics |
| `pitch_tracker.py` | direct note tracking (`--selector pitch`) and its one-voice renderer |
| `synth_sink.py` | built-in synth (`--sink synth`); run it alone to hear a test melody |
| `board_worker.py`, `persistent_board.py`, `transport_events.py` | worker in the SDK container, one-boot adapter, collection of leftover events |
| `board_lock.py` | one user of the board at a time |
| `windowing.py` | sliding windows and rates |
| `build_chip.py`, `build_banner.py`, `banner/` | builds the chip application; the same with a text on the display; the display step's C source |
| `chipref.py`, `diagnose_import.py`, `eval_modes.py` | integer reference, import diagnosis, expected accuracy without a board |
| `board_check.py` | test set on the board through the SDK's own batch path (plain applications only) |
| `check_persistent.py`, `check_call_limit.py`, `check_banner.py` | one-boot checks; `check_banner.py --testset` runs the whole test set on one boot |
| `midi_check.py` | sends a test phrase to a MIDI port such as GarageBand |
| `fake_worker.py`, `test_*.py` | software-only tests |
| `results/` | result files of 2026-10-08 evening, by model and for the live runs |
| `*.json` | earlier results and preserved failure records |
| `LIVE_CHIP.md`, `HARDWARE_BENCHMARK.md`, `TRAINING_FEEDBACK.md`, `PROCESS.md` | how to run, benchmark tables, findings for training, the story |

One file outside `yujie/` was changed on this branch: `jan/singalong/Pulsar/IF_DNR/import_torch_weights.py`
(the import fix). `handoff/runs/A_vsP_chip_shift3/` was copied from `origin/singalong-pitch-handoff` at `4917b31`.

## 4. How to get going again

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

**Rebuild a model's chip applications** (inside `singalong_sdk`, from `/workspace/jan/singalong`; no board
needed for steps 1 to 4). `M` is the model's folder name under `handoff/runs/`:

```bash
M=A_vsP_chip_shift3
# 1. import the weights; must end with IMPORT CHECK PASSED
python3 Pulsar/IF_DNR/import_torch_weights.py ../../handoff/runs/$M/weights_int8.npz \
    --out ../../data/runs/${M}_talamo --check ../../handoff/testdata/damp250
# 2. expected accuracy on the chip. Writes yujie/chip/chip_expectation.json (tracked): move the result away and restore that file
python3 ../../yujie/chip/eval_modes.py ../../handoff/runs/$M
# 3. plain chip application (refuses to overwrite an existing app folder)
python3 ../../yujie/chip/build_chip.py ../../data/runs/${M}_talamo \
    --npz ../../handoff/runs/$M/weights_int8.npz --check ../../handoff/testdata/damp250
# 4. the same with the text on the display
python3 ../../yujie/chip/build_banner.py ../../data/runs/${M}_talamo --npz ../../handoff/runs/$M/weights_int8.npz
# 5. board: whole test set, plain application (about 10 minutes). Writes yujie/chip/board_check.json (tracked): same care as in step 2
python3 ../../yujie/chip/board_check.py ../../data/runs/${M}_talamo --npz ../../handoff/runs/$M/weights_int8.npz --crops 512
# 6. board: whole test set, banner application, on one boot (about 10 minutes)
python3 -u ../../yujie/chip/check_banner.py ../../data/runs/${M}_talamo --npz ../../handoff/runs/$M/weights_int8.npz --testset --debug-logs
```

Step 2 alone already tells what the board will give. For `A_vocalset_P` and `A_vsP_chip_shift3` steps 1 to 4
are done on this Mac.

**Software tests** (no board): from `yujie/chip` with the host Python, `test_windowing.py`,
`test_transport_events.py`, `test_worker_failure.py`, `test_shutdown.py`; in the container,
`docker exec -w /workspace/yujie/chip singalong_sdk python3 test_board_lock.py`. All passed on 2026-10-08.

**Voice demo, any time:** `cd ~/TUM/NeuroTUM/demo-voice-v1 && ./run.sh` (about 10 s to load). It replaces
whatever is on the board. It does not use the board lock, so stop everything else first.

## 5. Open problems, most important first

1. **Nobody has judged the live note tracking.** Rehearse with a person singing, headphones on, and write
   down what is heard. For a number: record one sung take with known notes and feed it to chip and baseline
   with `--source wav:`.
2. **The note is about 1 s late on the chip.** Half of it is waiting for the next 0.5 s window, the rest is
   the 0.38 s round trip, most of which is sending the 2 s window over the serial link. `--playout-delay 1.0`
   works but gave 32 to 40 late frames in two of three 3-minute runs; at 1.25 s a 4-minute microphone
   run had 2 and the recording runs had none. Going well below needs a different chip application
   (shorter windows, or frames one at a time), not a setting.
3. **Start-up handshake.** Fails in roughly one start in three (2 of 4 in one test series). The retry
   handles it; with `--start-attempts 5` a complete failure is unlikely. Start the demo before going on stage.
   The cause (the host's handshake reply is sent once and never repeated) is in the SDK and is not fixed.
4. **Sessions without `--worker-debug` are not trusted** beyond about 300 windows (one stall, cause unknown).
5. **The octave is chosen on the laptop**, from the 48 bands. The network only names the pitch class.
   Say so when presenting.
6. **`board_check.py --banner` does not work reliably**: after a failed start, its clear-and-reboot was
   rejected by the board's loader 2 times out of 2 ("Unexpected response"). Use `check_banner.py --testset`
   for the banner application. In the live loop's own sequence the boot after a clear worked 4 times out of 4.
7. **Not measured:** power, time on the chip alone, hidden-layer activity on the board.

## 6. Known quirks of this setup

- **"waiting for sync" / empty result.** The start-up handshake failed. A new start, after a RAM-only clear
  with `c1util.py`, usually works; the live loop does this by itself.
- **One client per board.** Everything in `yujie/chip/` that opens the board takes `/tmp/pulsar_board.lock`
  in the container and refuses to start if it is held. The voice demo and other tools do not.
- **A second client crashes the first.** On 2026-10-08 a check started during a live run made the chip
  application stop with "ping message corrupted". That is what the lock is for.
- **Stopping:** SIGINT or SIGTERM, never SIGKILL on anything that holds the board. A worker that does not
  react to end-of-input is sent SIGTERM by the loop; if it still runs, the loop says so and exits.
- **Scripts that write tracked result files in place:** `eval_modes.py` (`chip_expectation.json`),
  `board_check.py` (`board_check.json`), `check_banner.py` without `--testset` (`banner_check.json`).
  Running them for another model overwrites the record; copy the result away and restore the file.
- **`data/` is git-ignored**: imported parameters, built applications and run logs live only on this Mac.
- **Packages in `singalong_sdk`** were installed with `pip --user`; they are lost if the container is removed.
- **Builds are slow** (2 to 4 minutes): the SDK image is x86 running under emulation on this Mac.
- **The display text stays** until the board is reset or powered off, whatever runs afterwards.

## 7. Rules the user set

- RAM boot only; never write flash.
- Never commit or push to `main` of either repository, and never to a teammate's branch.
  Work on `yujie/*` branches.
- No commit, push, overwrite or discard without the user's explicit permission. (This is newer than the
  earlier rule to push after every update of `TRAINING_FEEDBACK.md`; ask before pushing.)
- Work step by step: propose, name measurable acceptance criteria, implement only after approval, report
  the actual results including failures, then wait.
- Do not switch the demo to a new model or application by default; new things are options.
- The two trainer-versus-chip rule differences are **kept** for `A_vocalset_P`; do not "fix" a model.
- Never modify the installed SDK.
- The voice demo must stay runnable at any time (`demo-voice-v1/`).
- Recorded audio stays out of git; `audio_data/data_mic/` is read-only for training work.
- Say plainly whether a number is measured on the board, software, or simulated, and do not say something
  was heard unless a person heard it.

## 8. Loose ends outside Phase B

- `grp-gold-neurotum`, branch `voice-baseline` (voice demo source, notes, guide) is **not pushed** to
  the Innatera repository; only `live-mic-scene` is. A snapshot is on the team repository as
  `yujie/live-mic-voice`, without the last display changes.
- The Innatera repository's `main` gained a "live microphone" example on 8 October; not merged locally.
- Another session left `tests/record_playback.py` untracked in `grp-gold-neurotum/audio/scene-classification/`.
- Never done for the voice demo: a known-tone check of the sample rate, microphone sensitivity and clipping.
- A board microphone with the feature extraction on the chip would remove the serial transfer and most of
  the delay. Pieces exist: the voice demo has microphone capture, 16 kHz resampling and frame-by-frame
  stepping of a network on the board, and `jan/singalong/kernels_c/bands/bands_core.c` is a portable C
  version of the 48-band front-end with notes for the port. They have not been put together or run on
  the board, and the model was trained on windows that start from a reset state. Not started.
