# Start here: handoff for the sing-along chip work

Snapshot of 2026-10-09, 01:30. The pitch is on Friday 9 October, 10:00.
Paths are relative to `~/TUM/NeuroTUM/agent-pitch-handoff` unless they start with `/workspace` (inside a
container). Details live in `yujie/chip/LIVE_CHIP.md` (options, timing), `HARDWARE_BENCHMARK.md` (figures
for the presentation), `TRAINING_FEEDBACK.md` (for the training team) and `PROCESS.md` (history, mistakes).

## 1. Architecture and model

**Goal:** a person sings or hums a note and a synth plays that note, in any song, with the pitch network
on the physical Pulsar C1.

```
MacBook microphone -> front-end on the laptop (48 semitone bands, one frame per 16 ms)
  -> latest 2 s window every 0.5 s -> worker in the singalong_sdk container -> Pulsar C1
     (IF encoder + recurrent SNN 48 -> 80 -> 24, RAM boot, fixed text on the display)
  -> spikes of outputs 12..23 -> rate over 8 frames -> pitch tracker on the laptop
     (pitch class from the chip, octave from the 48 bands) -> one-voice renderer
  -> built-in Python synth (sounddevice) and a live page at http://127.0.0.1:8765
```

**Validated model:** `A_vsP_chip_shift3`, trained by Simon under the chip's rules
(`handoff/runs/A_vsP_chip_shift3/weights_int8.npz`, md5 `6b47d452dddcd1cdddbe8d4f2a152035`).

| Measured on the board | Result |
| --- | --- |
| Test windows identical to the integer reference | 1,451 of 1,451, plain application and banner application |
| Pitch accuracy, songs / exercises / DAMP | 0.577 / 0.591 / 0.378 (chance about 0.08) |
| RAM | 23.23 % plain, 25.59 % with the display text |
| Round trip per 2 s window | about 0.38 s median |

The older `A_vocalset_P` (0.516 / 0.589 / 0.364) is still the default when `--model` is left out.
Baseline mode (`--mode baseline`) uses the laptop's own pitch estimate and no board.

## 2. Completed steps

| Step | Final status |
| --- | --- |
| 1. Board lock (`board_lock.py`) | passed, 6 of 6 criteria |
| 2. Worker timeout and failure detection | passed; a frozen worker gives an error in about 3 s and the program exits in about 24 s |
| 3. Notes released, metrics kept | 8 of 9 passed; whether the 10 s checkpoint costs late frames stayed inconclusive (`--checkpoint 0` turns it off) |
| 4. Simon's model on the board | passed, 8 of 8 |
| 4.5 Hardware benchmark | written: `HARDWARE_BENCHMARK.md` |
| 5. Text on the board's display (`--banner`) | passed; the whole test set on one boot, 1,451 of 1,451; the user saw the text |
| 6. Slow-round-trip limit (`--slow-limit`, default 3) | passed: 26 of 26 software tests; board recording check with the default, 46 of 46 exact, no late frames; three microphone runs with `--slow-limit 6`, longest streak 2, so the limit itself was never reached on the board |
| 7. Live page in the look of the slide deck, with a piano roll (`live_view.py`) | passed in software and on the board with recordings: 44 of 44 software tests; recording through the chip with the page open, 46 of 46 exact, no late frame; nobody has judged the page by eye on a projector, and it has not run with the microphone on the chip (issue 13) |

Also done: direct note tracking (`--selector pitch`), the built-in synth (`--sink synth`), the live page
(`live_view.py`), automatic clear-and-retry at start-up (`--start-attempts`), and the fix of the old
338-window limit. The user has heard the synth. Nobody has scored the live note tracking.

## 3. Branch, commits, uncommitted changes

- Branch `yujie/singalong-chip`, remote `nc-fortiss/MNH26SVR`. The remote is at `50ea178`
  (`1886d62` code and results, `50ea178` docs), the last state that was pushed.
- **Five local commits on top of it, not pushed.** Pushing needs the user's permission.

| Commit | What it holds |
| --- | --- |
| `ed75474` | the `--slow-limit` option with its software tests (`live_chip.py`, `fake_worker.py`, `test_slow_round_trips.py`) |
| `7a6791d` | documentation of it, the timestamp drift and the Bluetooth findings |
| `beff077` | the new live page and its tests (`live_view.py`, `test_live_view.py`) |
| `c3f026d` | documentation of the page (`LIVE_CHIP.md`, "What the page shows") |
| the newest | this file, brought up to date |

- **What `ed75474` changes:** `--slow-limit N` with default 3, two counters in the metrics
  (`slow_round_trips`, `longest_slow_streak`), and the error text, which now names the count
  ("3 consecutive round trips exceeded update period"). Without the option the stop rule is the old one.
- **What `beff077` changes:** only `live_view.py`, the page and what it is given to draw. The loop, the
  tracker, the synth and everything that talks to the board are untouched. `/state` kept every earlier key
  and gained `timeline`, `raw`, `decoder`, `model` and `latency.slow`.
- **Uncommitted on purpose:** `yujie/chip/results/live_2026-10-08/presentation_rehearsal.json`, the metrics
  of the rehearsal that stopped, and `yujie/chip/UI_REDESIGN_PLAN.md`, the plan the page was built from.
  The user keeps both local. The plan predates two later decisions (section 6), so the page differs from it.

## 4. The demo command and its safety settings

Board on USB, boot switch on UART, `evk-bridge` running, `singalong_sdk` container up. From `agent-pitch-handoff`:

```bash
python -u yujie/chip/live_view.py --source "mic:MacBook Pro Microphone" --mode chip --worker-debug --model A_vsP_chip_shift3 --banner --selector pitch --playout-delay 1.25 --start-attempts 5 --slow-limit 6 --sink synth
```

Open http://127.0.0.1:8765. Start-up takes about 25 s, plus about 35 s for each retry. Stop with Ctrl-C.
If Ctrl-C does nothing, `pkill -TERM -f yujie/chip/live_view.py` from another terminal runs the same clean
shutdown. With `--duration 180` a run stops by itself after 180 s of audio.

- `--worker-debug`: every long run that passed had it on. Keep it.
- `--start-attempts 5`: the start-up handshake fails in roughly one start in three; each failure is cleared
  (RAM only) and retried by itself.
- `--playout-delay 1.25`: the note comes about 1.25 s after the voice. 1.0 s works with more late frames.
- `--slow-limit 6`: the run stops after six round trips in a row longer than 0.5 s, not three (issue 1).
- Automatic: board lock, 3 s window timeout, notes released first on any stop, metrics every 10 s.
- **Microphone by name**: device numbers shift when a monitor or speaker is connected.
- **Sound output**: the Mac's default output at the moment of starting. Check it just before starting:
  macOS switches to a Bluetooth device by itself when it reconnects. Do not use a Bluetooth output
  (issue 11). Measured so far: the built-in speakers and one Bluetooth device, nothing else.
- **The page**: what each part shows is in `LIVE_CHIP.md`, "What the page shows". Only a running loop
  shows live values; "Not connected" means the program is not running. It is built for 1280 x 720 and
  scales with the window like a slide, so give it a wide window.
- **Laptop-only fallback**, no board, note after about 0.1 s:

```bash
python -u yujie/chip/live_view.py --source "mic:MacBook Pro Microphone" --mode baseline --selector pitch --playout-delay 0.1 --sink synth
```

The display text stays until the board is power-cycled, also in baseline mode.

## 5. Known issues

1. **Slow-round-trip stop rule.** The loop stops after `--slow-limit` round trips in a row longer than the
   0.5 s period: 3 by default, 6 in the demo command. With 3, the rehearsal stopped after 464 windows
   (0.54 s, 1.67 s, 0.62 s), and 10 s earlier it had missed a stop by 13 ms. Among the 3,174 chip windows
   up to then, 1.2 % were slow and this was the only streak of three. With 6, three microphone runs gave
   1,529 windows, 17 slow round trips and a longest streak of 2, so the higher limit was never needed on
   the board; that it rides through the rehearsal's pattern is shown in software only.
   Two other rules also stop a run and are unchanged: more than three update periods between two processed
   windows (a single stall of roughly 1.5 to 2 s; "Output gap exceeds valid post-burn-in window"), and the
   3 s window timeout. The rehearsal's 1.67 s stall was about 0.3 s short of the first of these (calculated).
   Making a longer single stall recoverable was discussed and deliberately not started.
2. **Live note tracking is not measured.** In three 3-minute live runs the median tracked note lasted
   0.30 to 0.34 s, so notes flip often. Dataset accuracy says nothing about humming into a MacBook.
3. **Sound output.** On a loudspeaker the synth is heard by the microphone and can keep itself going. Its
   loudness follows the voice, which damps this. A Bluetooth output drifts (issue 11). A louder speaker, a
   wired speaker and wired headphones are untested.
4. **About 1 s delay on the chip**, by design of the window scheme. Not a setting.
5. **Start-up handshake** (SDK-internal) is retried, not fixed.
6. **The octave is chosen on the laptop.** The network only names the pitch class.
7. **Runs without `--worker-debug`** are not trusted beyond about 300 windows (one stall, cause unknown).
8. **`board_check.py --banner`** fails in its own clear-and-reboot; use `check_banner.py --testset` for the
   banner application.
9. **MacBook input volume** was at 43 of 100 in the last run and the level was low.
10. Fixed on 2026-10-09: the comment at the top of `live_view.py` now shows the demo command.
11. **Microphone timestamp drift, unresolved and not investigated.** In microphone runs the frames reach
    the loop later and later than their timestamps, which uses up the margin of the playout delay: frames
    are late from a drift of about 0.43 s. Measured in minute 3: 0.24 s on the audible built-in speakers
    (rehearsal), 0.66 and 0.73 s with a Bluetooth output (nearly every window late from minute 2, and 73 %
    of all frames in the 410 s run), 0.08 s on the muted speakers with Bluetooth off (no late frame). The
    cause is unknown, and so is whether the real delay from voice to note grows. A fresh start begins at
    zero. Table and details: `LIVE_CHIP.md`, "Microphone timestamp drift".
12. **Ctrl-C did not reach one run** (macOS Terminal window, cause unknown). SIGTERM stopped it cleanly in
    9 s; see section 4.
13. **The new page is checked with recordings only.** Open points:
    - It has not run with the microphone on the chip, and nobody has judged it by eye on a projector.
    - In laptop-only 60 s runs it cost no late frame (0 of 3,747 with the page open and without). One 20 s
      run had 2 late frames while a headless Chrome was starting for a screenshot; the cause is not
      established. In a microphone run late frames come from issue 11 in any case.
    - The pitch range of the piano roll follows the notes shown, so the picture rescales when a higher or
      lower note appears.
    - At the end of a recording the status reads "No chip result for 2 s" for about a second before
      "Stopped". The old page did the same.
    - After a stop the footer keeps the last run's figures.

## 6. Next task, pending approvals, restrictions

**Nothing is approved to run right now.** When the session ended the user had set these restrictions, and
they hold until the user lifts them: no code or configuration changes, no installs or downloads, no builds
or tests, no board access, no retraining, no change to the banner application, no commit, no push.

Decisions waiting for the user:

1. **Push** of the five local commits (section 3). Not approved.
2. **Sound output for the pitch.** Bluetooth is out by the numbers of issue 11. Whichever output is chosen
   needs one 3-minute run with the demo command first. That run is also the first of the new page with
   the microphone on the chip, and the user's chance to say whether it is readable (issue 13).
3. **Sustained monophonic synth.** A read-only feasibility report was delivered. Findings: sustaining
   without retriggering, release on silence and a simple additive timbre already exist; the gap is note
   stability. Recommendation: expose the pitch tracker's existing confirmation time, minimum note length and
   margin as options, add one steadier "humming" setting and an optional timbre; keep loudness following the
   voice; no confirmation across several chip updates, no glide, no new controller class. Estimated extra
   delay on note changes: 150 to 250 ms. Proposed checks: host-only note-sequence tests, the generated tone
   file (known 2 s notes), one short board run, then live humming by the user. Awaiting approval.

Decided by the user on 2026-10-08 and 09, do not reopen without asking:

- No board test that pauses the worker (`kill -STOP`). It cannot tell limit 3 from limit 6 and adds a
  communication risk.
- The drift of issue 11 is recorded only. No investigation and no change to the audio scheduling before
  the pitch.
- The page keeps the microphone monitor as its own card (level, meter from -70 to 0 dB, "Voice detected" or
  "Silence"), to tell sound reaching the laptop apart from the network recognising a pitch.
- The timeline is a piano roll in the deck's light theme with rust-coloured notes. Its pitch range follows
  the notes; no snapping to octaves.
- `UI_REDESIGN_PLAN.md` stays local.

**The user's working rules:**

- Step by step: inspect, propose with measurable acceptance criteria, implement only after approval, report
  actual results including failures, then stop and wait.
- Show code and documentation diffs before committing. Never push without explicit permission.
- RAM boot only; never flash. Never modify the installed SDK. Never retrain or alter a model.
- Never commit to `main` or to a teammate's branch; work on `yujie/*`.
- New behaviour is an option; the validated path and the default stay as they are.
- One program on the board at a time. Never SIGKILL anything that holds the board.
- Say whether a number is measured on the board, in software, or estimated. Do not say something was heard
  unless a person heard it.
- Recorded audio stays out of git. The frozen voice demo (`~/TUM/NeuroTUM/demo-voice-v1/run.sh`) must stay runnable.

## 7. Commands and files

**Software tests** (no board), from `yujie/chip` with `python`:
`test_windowing.py`, `test_transport_events.py`, `test_worker_failure.py`, `test_shutdown.py`,
`test_slow_round_trips.py`, `test_live_view.py`; in the container:
`docker exec -w /workspace/yujie/chip singalong_sdk python3 test_board_lock.py`. Together 44 tests.

**The page without the board, the microphone or sound** (from `agent-pitch-handoff`, then open
http://127.0.0.1:8765):

```bash
python -u yujie/chip/live_view.py --source wav:data/live_chip/tones.wav --mode baseline --selector pitch --playout-delay 0.1 --sink print --duration 20
```

The file holds 2-second tones; the console and the piano roll show the notes 60, 64, 67, 69, 67, 64, 62, 60,
64. It is git-ignored; `yujie/chip/make_test_audio.py data/live_chip/tones.wav --seconds 190` makes it.

**Board checks:**

```bash
# is anything using the board?
docker exec singalong_sdk sh -c 'ps -eo pid,args | grep -E "[b]oard_worker|[c]heck_|[c]1util|[b]oard_check"'
# whole loop on a recording, silent, every window compared (expect 46 of 46 exact, no late frames);
# with live_view.py in place of live_chip.py the same check also serves the page
python -u yujie/chip/live_chip.py \
  --source "wav:Hackathon/08 Songs/sources/01_Happy_Birthday/audio/624721__asiilanna__happy-birthday.wav" \
  --mode chip --worker-debug --verify-reference --model A_vsP_chip_shift3 --banner --selector pitch \
  --playout-delay 1.25 --sink print --duration 25 --metrics data/live_chip/recheck.json
# whole test set through the banner application on one boot (about 10 minutes)
docker exec -w /workspace/jan/singalong singalong_sdk python3 -u /workspace/yujie/chip/check_banner.py \
  /workspace/data/runs/A_vsP_chip_shift3_talamo --npz /workspace/handoff/runs/A_vsP_chip_shift3/weights_int8.npz \
  --testset --debug-logs
```

**Setup if something is missing:**

- Bridge on the Mac: `evk-bridge -i <BOARD_ID> --listen 0.0.0.0 --port 1700 --gdb-port 1701`.
- The container `singalong_sdk` (image `innatera_sdk:2.3.0`, Talamo 4.7.1) mounts this folder at `/workspace`.
  `data/` is git-ignored, so imported models and built applications exist only on this Mac, under
  `data/runs/<model>_talamo/`. The commands to recreate the container and to rebuild a model (import,
  `build_chip.py`, `build_banner.py`) are in this file at commit `50ea178`, section 4:
  `git show 50ea178:yujie/chip/NEXT_SESSION.md`.
- `eval_modes.py`, `board_check.py` and `check_banner.py` (without `--testset`) overwrite tracked result
  files in `yujie/chip/`; copy results away and restore the file.

**Files in `yujie/chip/`:**

| File | Purpose |
| --- | --- |
| `live_view.py` | the demo's entry point; runs `live_chip.py` and serves the page |
| `live_chip.py` | the loop: capture, windows, board client, selector, renderer, shutdown, metrics |
| `pitch_tracker.py` | note tracking and the one-voice renderer |
| `synth_sink.py` | the built-in synth; run alone it plays a test melody |
| `board_worker.py`, `persistent_board.py`, `transport_events.py` | worker in the container, one-boot adapter, event clean-up |
| `board_lock.py` | one user of the board |
| `build_chip.py`, `build_banner.py`, `banner/` | chip applications; the display step's C source |
| `check_persistent.py`, `check_banner.py`, `board_check.py` | board checks |
| `fake_worker.py`, `test_*.py` | software tests |
| `results/` | result files by model and for the live runs of 2026-10-08 |

Other places: Simon's models and test data in `handoff/` (his branch is `singalong-pitch-handoff`; do not
commit there), team code in `jan/singalong/` and `simon/`, run logs in `data/live_chip/`.
