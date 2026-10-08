# Shape 2 live loop: how to run it and what it does

The Mac runs audio capture, the existing `StreamingFrontEnd`, window scheduling, a selector and the
sound. `singalong_sdk` runs a persistent physical-board worker. The board runs the IF encoder and the
recurrent network. No board microphone firmware is involved.

Updated 2026-10-08, 23:40. State and open problems: `NEXT_SESSION.md`.

## Start and fallback

From `<repo root>` on the Mac:

```bash
# The demo: physical chip, Simon's model, text on the board's display, the sung note on the built-in synth.
python -u yujie/chip/live_view.py --source "mic:MacBook Pro Microphone" --mode chip --worker-debug --model A_vsP_chip_shift3 --banner --selector pitch --playout-delay 1.25 --start-attempts 5 --slow-limit 6 --sink synth

# Laptop only. Does not start Docker or touch the board. The note comes about 0.1 s after the voice.
python -u yujie/chip/live_view.py --source "mic:MacBook Pro Microphone" --mode baseline --selector pitch --playout-delay 0.1 --sink synth
```

Then open http://127.0.0.1:8765. Stop with Ctrl-C or SIGTERM. If Ctrl-C does nothing, send SIGTERM from
another terminal; it runs the same clean shutdown. On 2026-10-08 Ctrl-C did not reach one run (cause
unknown) and SIGTERM to the host program, sent by process number, ended it in 9 s with every shutdown
step done. By name:

```bash
pkill -TERM -f yujie/chip/live_view.py
```

With `--duration 180` a run stops by itself after 180 s of audio.

- `live_view.py` runs `live_chip.py` unchanged and adds the page; `live_chip.py` takes the same options
  and prints one line per note change.
- The microphone is named, because device numbers shift whenever a monitor, headset or speaker is
  connected. `mic:N` still works; `python -m sounddevice`
  lists the numbers. `mic` alone takes the Mac's default input.
- The synth plays on the Mac's default output as it is when the demo starts. macOS makes a Bluetooth
  device the default output and input by itself when it reconnects, so check the output just before
  starting. The demo still records from the named MacBook microphone.
- **Do not use a Bluetooth output.** In both runs on one (2026-10-08) the timing drifted far faster than
  on the built-in speakers, and from the second minute nearly every window was late ("Known limits").
  Bluetooth also adds its own delay, which the page cannot see.
- A loudspeaker is heard by the microphone, and the tracker treats the synth's sound like a voice; keep a
  speaker away from the MacBook. Headphones avoid that. Wired headphones and a wired speaker have not
  been run with the demo.
- Board failures stop the run with an error. There is no silent software substitution; choose
  `--mode baseline` yourself. Never use SIGKILL on anything that holds the board.

Prerequisites: an imported model and its built application under `data/runs/<model>_talamo/`
(`NEXT_SESSION.md`, section 4), the `singalong_sdk` container running, and the bridge on port 1700 with
the board's boot switch on UART. Host packages: NumPy, sounddevice, mido with python-rtmidi, ffmpeg.
CoreAudio provides the requested 16 kHz input; unsupported input settings fail before capture.

## Options

| Option | Meaning | Default |
| --- | --- | --- |
| `--source mic[:N]`, `mic:<name>`, `wav:<file>` | microphone by number or by its name (or a unique part of it), or a recording played at real speed | required |
| `--mode chip`, `baseline` | network on the board, or the laptop's own pitch estimate | `chip` |
| `--model NAME` | chip: imported model `data/runs/NAME_talamo` | `A_vocalset_P` |
| `--banner` | chip: the application that shows a fixed text on the board's display | off |
| `--selector pitch` | play the sung note itself | |
| `--selector score`, `key`, `song` | the team's chord selectors; `score` follows Amazing Grace | `score` |
| `--sink synth[:N]` | built-in synth on output device N | |
| `--sink print`, `wav:<file>` | no sound; note log only, or an offline rendering to a file | `print` |
| `--sink midi:<port>`, `fluidsynth:<sf2>` | an existing MIDI port, or FluidSynth (not installed on this Mac) | |
| `--playout-delay S` | seconds from the end of an audio frame to its note. Chip: at least period + 0.4. Baseline: at least 0.05 | 1.0 |
| `--period S` | chip: seconds between windows, 0.5 to 1 | 0.5 |
| `--release S` | seconds without voice before the sound stops | pitch 0.4; chord selectors off |
| `--smooth N` | pitch: frames averaged before deciding | 12 |
| `--worker-debug` | chip: SDK communication logging. Every long run that passed had it on | off |
| `--verify-reference` | chip: compare every window with the integer reference and stop on a difference | off |
| `--window-timeout S` | chip: seconds to wait for one window before stopping with an error | 3 |
| `--slow-limit N` | chip: round trips in a row longer than the period before the run stops with an error. The demo uses 6 | 3 |
| `--start-attempts N` | chip: board starts tried when the start-up handshake fails | 3 |
| `--checkpoint S` | seconds between partial metrics files; 0 turns them off | 10 |
| `--duration S`, `--metrics <file>` | stop after S seconds; where the metrics go | `data/live_chip/latest.json` |
| `--view-port N`, `--open-browser` | `live_view.py` only: port of the page, open it in the browser | 8765, off |

`--sink midi:GarageBand` needs GarageBand running with a software-instrument track selected. GarageBand
ignores the program change the sink sends, so the sound is whatever instrument that track has. Sending
succeeds even when nothing sounds. `midi_check.py` sends a test phrase.

## What the page shows

Mode (chip or laptop), run state and errors; microphone level and whether the front-end counts the frame
as voice; the tracked note; the twelve pitch-class values coming from the chip or the baseline; the notes
sounding, on a keyboard from E2 to E6, and the last MIDI messages; chip round trip and number of windows;
delay from sound to note; late frames, skipped updates and connection errors. The page only observes.

## What protects a run

- **Board lock.** Everything in this folder that opens the board takes `/tmp/pulsar_board.lock` inside the
  container, without waiting. A second user is refused before anything is sent and is told who holds it.
  The lock goes away with its process. The voice demo does not use it.
- **Start-up retry.** If the board boots but the start-up handshake fails ("waiting for sync"), the loop
  clears the board (RAM boot of the same application) and starts a fresh worker, up to `--start-attempts`
  times, before any audio is processed. Each failure is printed and listed in the metrics.
  Other start-up errors stop at once. Nothing is retried or rebooted once audio is being processed.
- **Window timeout.** No answer within `--window-timeout` stops the run with "Board timeout".
- **Slow round trips.** `--slow-limit` round trips in a row longer than the period stop the run with
  "N consecutive round trips exceeded update period". The default is 3. The demo command uses 6, so that
  a short stall does not end the demo; until the limit is reached, notes come late and updates may be
  skipped to catch up. The metrics count them (`slow_round_trips`, `longest_slow_streak`).
  A single stall that leaves more than three update periods between two processed windows (about 2 s)
  still stops the run at any limit: "Output gap exceeds valid post-burn-in window".
- **Shutdown, in this order:** notes off (renderer, then the sink's own all-notes-off), metrics written,
  threads joined, sound closed, worker closed, metrics rewritten with a `shutdown` record. Every step runs
  even if an earlier one failed or a signal arrived. A worker that ignores end-of-input gets SIGTERM after
  5 s; if it still runs 10 s later the loop names its process and exits. It is never force-killed.
- **Checkpoint.** While running, `<metrics>.partial` is rewritten every `--checkpoint` seconds and removed
  after a clean finish.

Measured on the board with a worker frozen in mid-run: error on the page after 3.2 s, notes released
0.2 s after the failure was detected, program exited after 23.7 s, metrics kept
(`results/live_2026-10-08/step3_frozen.json`).

## Direct note tracking (`--selector pitch`)

`pitch_tracker.py`. Input per 16 ms frame: the twelve pitch-class values, the 48 bands and the front-end's
voiced flag. Output: one MIDI note or silence.

- **Pitch class:** the strongest of the twelve values, averaged over the last `--smooth` frames.
- **Switching:** another class must be 25 % stronger for 6 frames, and the current note must have lasted 9 frames.
- **Silence:** no voiced frame for `--release` seconds releases the note. The front-end's voiced flag adapts
  to the room's noise floor.
- **Octave:** not from the network, which has no octave. On the laptop: among the bands of that pitch
  class, the lowest one that is at least 60 % as strong as the strongest. It changes within a note only
  after 12 frames of agreement.
- **Sound:** one voice, velocity from confidence, expression (CC 11) from the singer's level as in the
  team's renderer.

Checked on a generated tone sequence (C E G A G E D C, 2 s each): all notes and octaves right in baseline
mode and on the board; a change is followed about 0.12 to 0.14 s later in baseline mode and about 0.2 s
later in chip mode, counted in audio time, with the playout delay on top. On a recording of singing, chip
(with `A_vocalset_P`) and baseline agreed on the note in 47 % of frames; that is agreement, not accuracy. Nothing was scored
against known notes.

## Reproduce checks

```bash
# Software only, from yujie/chip: windows and rates, leftover events, worker failure, shutdown, slow round trips.
for t in test_windowing.py test_transport_events.py test_worker_failure.py test_shutdown.py test_slow_round_trips.py; do
  python $t; done
docker exec -w /workspace/yujie/chip singalong_sdk python3 test_board_lock.py

# The synth alone: a known melody on the default output. Only a person listening confirms the sound.
python yujie/chip/synth_sink.py

# Physical board: one boot, six deterministic windows against the reference.
docker exec -w /workspace/jan/singalong singalong_sdk python3 -u /workspace/yujie/chip/check_persistent.py \
  /workspace/data/runs/A_vsP_chip_shift3_talamo \
  --npz /workspace/handoff/runs/A_vsP_chip_shift3/weights_int8.npz \
  --output /workspace/data/live_chip/persistent_recheck.json

# Physical board through the whole loop, on a recording, no sound, every window compared.
python -u yujie/chip/live_chip.py \
  --source "wav:Hackathon/08 Songs/sources/01_Happy_Birthday/audio/624721__asiilanna__happy-birthday.wav" \
  --mode chip --worker-debug --verify-reference --model A_vsP_chip_shift3 --banner --selector pitch \
  --playout-delay 1.25 --sink print --duration 25 --metrics data/live_chip/recheck.json
```

Expected for the last one: 46 of 46 windows exact, no late frames, round trip about 0.38 s.
Use separate metrics file names to keep earlier evidence. Microphone audio is never saved; metrics hold
levels and timing only.

## Known limits

- **About 1 s from voice to note on the chip**, by design of the window scheme (next section).
- **Start-up handshake fails in roughly one start in three.** Retried automatically; not fixed.
- **Runs without `--worker-debug` are not trusted beyond about 300 windows**: one such run stalled between
  window 300 and 399, cause unknown. With it on, 1,451 windows ran on one boot.
- **The display text stays** until the board is reset or powered off, also in baseline mode.
- **Live accuracy is unmeasured.**
- **Microphone timestamps drift.** Unresolved, cause unknown, not investigated (next paragraphs).
- **`--slow-limit 6` has not met a streak of three on the board.** With the old limit of 3 the rehearsal
  of 2026-10-08 stopped after 464 windows (0.54 s, 1.67 s, 0.62 s), and 10 s earlier it had missed a
  stop by 13 ms. With 6, three microphone runs gave 1,529 windows, 17 slow round trips and a longest
  streak of 2. That 6 rides through the rehearsal's pattern is shown in software only
  (`test_slow_round_trips.py`).

### Microphone timestamp drift

In microphone runs the frames reach the loop later and later than their own timestamps. With a recording
they do not (0.003 to 0.006 s throughout). The playout schedule is tied to those timestamps, so the
margin of the 1.25 s playout delay shrinks: a window's first frame is late by about
drift + round trip - 0.77 s (derived from the loop; for the rehearsal's largest lateness it gives 1.372 s
against 1.374 s measured). Frames are therefore counted late from a drift of about 0.43 s. Once a whole
window is late its frames go to the synth together, so notes can only change in half-second steps (read
from the loop, not measured).

Measured on 2026-10-08 with the demo command (the rehearsal without `--slow-limit`). Drift is how long
after the timestamp of its last frame a window was submitted, as the median of each minute, over windows
that were waiting for audio:

| Run | Sound output | Drift in minute 1 / 2 / 3 | Late frames |
| --- | --- | --- | --- |
| Rehearsal, 235 s | MacBook speakers, audible | 0.01 / 0.23 / 0.24 s, 0.42 s at the end | 210 of 14,470 |
| 23:10, 410 s | one Bluetooth device | 0.43 / 0.60 / 0.73 s, 1.31 s at the end | 18,494 of 25,501 (73 %) |
| 23:23, 180 s | the same Bluetooth device | 0.25 / 0.45 / 0.66 s | 2,483 of 11,126 (22 %) |
| 23:30, 180 s | MacBook speakers, muted, Bluetooth off | 0.02 / 0.06 / 0.08 s | 0 of 11,126 |

Not known: the cause; whether it is playing through the Bluetooth device or only having it connected (it
was also the Mac's default input in both runs); whether the real delay from voice to note grows, because
the metrics are tied to the drifting timestamps. The muted run is a single run. A fresh start begins at
zero. Until this is understood: no Bluetooth output, start the demo shortly before it is shown, and run
the output that will be used once for three minutes beforehand. The metrics of these four runs are on
this Mac only, not in git (`data/live_chip/demo_slowlimit6_*.json`,
`yujie/chip/results/live_2026-10-08/presentation_rehearsal.json`).

The earlier limit of about 338 windows per boot is fixed. Each window used to leave three event handles
open, and the SDK's native library can only wait on handle numbers below 1,024; `transport_events.py` now
collects the leftover events after every window. Before the fix, measured twice:

| Run | Windows before it stopped | All exact until then | How it stopped |
| --- | ---: | --- | --- |
| 185 s stability run, prerecorded audio at 500 ms (`chip_stability.json`) | 337, plus 2 start-up calls | yes | `wait_events()` returned `None` inside the SDK |
| Back-to-back calls, no audio pacing (`check_call_limit.py`, `call_limit.json`) | 338 | yes | the same error |

After the fix: 366 of 366 windows in the same 185 s run (`chip_stability_event_cleanup.json`), and 1,451 of
1,451 on one boot (`results/A_vsP_chip_shift3/banner_testset_check.json`).

## Timing and selector contract

- Each full inference is 125 frames of 48 integer-valued float32 features in 0..100.
  The existing frontend uses a 1024-sample Hann window and 256-sample hop at 16 kHz.
  A 125-frame sequence spans 2.048 seconds of source samples including the FFT support.
- A 500 ms update is 31.25 frames: exclusive ends are 125, 157, 188, 219, 250, etc.
  Scheduling uses the audio frame count, so rounding cannot accumulate drift.
- The first delivery is frames 93..124. Earlier frames warm the model/readout and are deliberately
  omitted. After that, each frame is delivered exactly once. The first 31 frames of any replayed
  window are never delivered. A gap too large for that valid region is a fatal error.
- Eight-frame moving rates are computed within each complete returned window, before cropping.
  This uses its overlap as readout context and avoids mixing spikes from different reset histories.
- The compiled app returns 126 frames including its flush frame. Only the first 125 are retained,
  exactly as in the validated `board_check.py`. The chip's one-frame layer delay is preserved.
- Frame `i` is timestamped at `(i*256 + 1024)/16000`, the end of its original audio support.
  No duplicate frame is delivered to the selector, which receives one step per 16 ms with `onset=True`,
  matching `live.py`. The integration always uses the melody head (channels 12..23); chord channels are ignored.
- Results are played out at those timestamps plus `--playout-delay`. This preserves selector and
  renderer pacing instead of sending bursts every half second. In chip mode the delay must cover the wait
  for the next window (up to 0.5 s) and the round trip (about 0.38 s): 1.0 s works with little margin,
  1.25 s is the safe value. In baseline mode each frame is delivered as it is computed.
- Capture continues independently of inference. Histories and output queues are bounded. Old
  submission deadlines may be skipped to catch up, and are counted; duplicate delivery is rejected.
  `--slow-limit` consecutive host round trips exceeding the period stop the run (default 3).

`inference_seconds` covers dataset preparation, transfer, physical inference, return, sample reset,
and output conversion inside the SDK worker. `host_roundtrip_seconds` additionally includes JSON,
Docker transport, and optional reference comparison. Neither is end-to-end latency.
`output_latency_seconds_summary` measures original audio-frame-end to renderer call completion.
For microphone input the epoch is derived from PortAudio ADC timestamps. It includes the deliberate
playout delay but excludes the selector's reaction time to a new note, the synth's output buffer and
attack, and the speaker; a frame center would add 32 ms. First-note startup also includes SDK loading,
boot, self-checks, and history collection. Host-to-board and board-to-host transfer are not separately
instrumented. `startup_failures` lists failed starts, and `shutdown` records how the run ended.

## SDK compatibility and startup

Private APIs are confined to `persistent_board.py` (pipeline name initialization, assembly, connection
attachment, execution) and `transport_events.py` (a wrapper around the SDK's native connection object).
The adapter requires Talamo 4.7.1 and the verified 48/80-recurrent/24 shape. The installed SDK is never modified.

`Pipeline.__call__` enters `_boot_embedded_application` each call. The adapter instead boots once,
holds `Soc.get_board_connection()` open, attaches it to the assembled pipeline, and materializes
every result while attached. Talamo's `board_comm.py` sends the internal-state reset command on
channel 1002 and awaits acknowledgment on 1003 after every sample. The generated reset kernel
resets the DNR/encoder state. `auto_reset=0` in the SNN kernel does not disable this host reset.
Empirical active/silent/repeat tests verify the zero-state behavior.

Start-up: after the boot, host and board do a handshake inside the SDK's native library. The host
answers the board's broadcast once and never repeats it; if that exchange fails, pings time out with
"waiting for sync" and no sample comes back on that boot. The worker checks a seeded active input against
`chipref.CHIP`, then silence, before declaring ready; these results never reach the selector. When the
handshake fails the loop clears the board and starts a fresh worker (see "What protects a run").
Whether `--worker-debug` changes the failure rate is not established; the failure also occurred with it on.

The banner application (`--banner`) is the same network with one pass-through step in front
(`banner/src/sa_banner/banner.c`) that copies each 48-value frame and draws a text once after boot.

The RAM-only clearing command, for use by hand:

```bash
docker exec singalong_sdk sh -c 'python3 /workspace/yujie/chip/board_lock.py "manual clear" -- python3 "$SPINE_DIR/tools/c1util.py" -S /workspace/data/runs/A_vsP_chip_shift3_talamo/app_hostfed_125/bin/c1/release/app.c1boot'
```

The frozen voice demo is independent and unchanged:
`cd ~/TUM/NeuroTUM/demo-voice-v1 && ./run.sh`.

## Validation record

`TRAINING_FEEDBACK.md` sections 6 and 7, `HARDWARE_BENCHMARK.md`, and the files under `results/`.
Do not infer live-chip success from the baseline or mock tests. The original 1,451-window results
and original model artifacts remain unchanged.
