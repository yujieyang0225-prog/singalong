# Shape 2 live integration

The Mac runs audio capture, the existing `StreamingFrontEnd`, window scheduling, the existing
score follower/HMM, and violin renderer. `singalong_sdk` runs a persistent physical-board worker.
The board runs the IF encoder and recurrent network. No board microphone firmware is involved.

## Start and fallback

From `<repo root>` on the Mac:

```bash
# Physical chip, MacBook microphone, existing GarageBand MIDI input.
# Check the device list first if devices have changed.
python -m sounddevice
python -u yujie/chip/live_chip.py \
  --source mic:1 --mode chip --worker-debug --sink midi:GarageBand --playout-delay 1.25

# Explicit laptop baseline fallback. Does not start Docker or touch the board.
python -u yujie/chip/live_chip.py \
  --source mic:1 --mode baseline --sink midi:GarageBand
```

Use headphones to avoid feeding the synth back into the microphone. Select a violin instrument
in GarageBand; the existing MIDI sink sends General MIDI program 40, but the receiving app controls
the actual instrument. MIDI delivery is measurable; acoustic quality requires a listening check.
`--sink print` needs no synth, `--sink wav:data/live_chip/demo.wav` renders the existing additive
violin to a file after stopping. This WAV sink is offline output, not audible live synthesis.
`--selector score` is the default Amazing Grace score follower; `--selector key` handles other songs.
The integration always uses the melody head (channels 12..23); chord channels are ignored.

Stop with Ctrl-C or SIGTERM. Board failures stop with an error; there is no silent software
substitution. Restart explicitly with `--mode baseline` if needed. Never use SIGKILL on the worker.

Prerequisites: the verified model and built app in `data/runs/A_vocalset_P_talamo`, running
`singalong_sdk`, and the bridge on port 1700 with the board's boot switch on UART. Host dependencies
are NumPy, sounddevice, mido/python-rtmidi, and ffmpeg (already available in the tested setup).
CoreAudio provides the requested 16 kHz stream; unsupported input settings fail before capture.
The microphone's native default is 48 kHz. No alternate feature implementation is used.

## Reproduce checks

```bash
# Software-only frontend, window, rate, HMM, and threaded mock-transport tests.
python yujie/chip/test_windowing.py

# Physical board: one boot, deterministic repeated calls, exact reference comparison.
docker exec singalong_sdk python3 -u /workspace/yujie/chip/check_persistent.py \
  /workspace/data/runs/A_vocalset_P_talamo \
  --npz /workspace/handoff/runs/A_vocalset_P/weights_int8.npz \
  --output /workspace/data/live_chip/persistent_recheck.json

# Reproducible tones (not a model accuracy test; audio is git-ignored).
python yujie/chip/make_test_audio.py \
  data/live_chip/tones.wav --seconds 190

# Prerecorded physical-chip pipeline, comparing every window to chipref.
python -u yujie/chip/live_chip.py \
  --source wav:data/live_chip/tones.wav --duration 12 --worker-debug --verify-reference \
  --sink wav:data/live_chip/tones_violin.wav --metrics data/live_chip/prerecorded.json

# Multi-minute run; timestamps advance at real audio speed.
python -u yujie/chip/live_chip.py \
  --source wav:data/live_chip/tones.wav --duration 185 --worker-debug --verify-reference \
  --sink print --metrics data/live_chip/stability.json
```

Use separate metrics filenames to preserve earlier evidence. Console logs show the active mode.
Microphone audio is not saved; metrics contain levels and timing, not raw audio.

## Known limit: about 338 windows per boot

The connection stops after about 338 windows: **measured twice, at the same count.**

| Run | Windows before it stopped | All exact until then | How it stopped |
| --- | ---: | --- | --- |
| 185 s stability run, prerecorded audio at 500 ms (`chip_stability.json`) | 337, plus 2 start-up calls | yes | `wait_events()` returned `None` inside the SDK |
| Back-to-back calls, no audio pacing (`check_call_limit.py`, `call_limit.json`) | 338 | yes | the same error |

Because the count repeats with and without audio pacing, the limit belongs to the connection, not to
timing or to the audio. It is consistent with a ceiling of about 1,024 transport events at three
events per window, which the previous session saw in its trace; that reading is not proven.
At one window every 500 ms it means **about 2 minutes 49 seconds of continuous chip operation per boot.**
After that the loop stops with an error. It does not fall back silently.
Reproduce with `check_call_limit.py` (its docstring has the command). Plan each chip performance to
stay under about 2 min 40 s, or restart the worker between pieces, until this is solved.

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
  No duplicate frame is delivered to the HMM. The existing HMM receives one step per 16 ms with
  `onset=True`, matching `live.py`. No new musical decoder or smoothing thresholds are introduced.
- Results are played out at those timestamps plus an explicit delay (1 s by default; use 1.25 s
  for microphone buffering headroom). This preserves selector and renderer pacing instead of
  sending bursts of MIDI events every half second. The initial omitted frames are not HMM steps;
  the score follower initializes on the first delivered evidence.
- Capture continues independently of inference. Histories and output queues are bounded. Old
  submission deadlines may be skipped to catch up, and are counted; duplicate delivery is rejected.
  Three consecutive host round trips exceeding the period stop the run.

`inference_seconds` covers dataset preparation, transfer, physical inference, return, sample reset,
and output conversion inside the SDK worker. `host_roundtrip_seconds` additionally includes JSON,
Docker transport, and optional reference comparison. Neither is end-to-end latency.
`output_latency_seconds_summary` measures original audio-frame-end to renderer call completion.
For microphone input the epoch is derived from PortAudio ADC timestamps. It includes the deliberate
playout delay but excludes unmeasured downstream synth/DAC/acoustic latency; a frame center would
add 32 ms. First-note startup also includes SDK loading, boot, self-checks, and history collection.
Host-to-board and board-to-host transfer are not separately instrumented.

## SDK compatibility and startup

Private APIs are confined to `persistent_board.py`: pipeline name initialization, assembly,
connection attachment, and execution. The adapter requires Talamo 4.7.1 and the verified
48/80-recurrent/24 shape. The installed SDK is never modified.

`Pipeline.__call__` enters `_boot_embedded_application` each call. The adapter instead boots once,
holds `Soc.get_board_connection()` open, attaches it to the assembled pipeline, and materializes
every result while attached. Talamo's `board_comm.py` sends the internal-state reset command on
channel 1002 and awaits acknowledgment on 1003 after every sample. The generated reset kernel
resets the DNR/encoder state. `auto_reset=0` in the SNN kernel does not disable this host reset.
Empirical active/silent/repeat tests verify the zero-state behavior.

On this setup, startup with an all-zero first input failed to return a complete sample, including
after a plain RAM clear; active-first standalone tests passed twice. The worker uses a seeded active
input checked against `chipref.CHIP`, followed by a silence/reset check, before declaring ready.
These startup results are never sent to the selector. Active-first worker startup also failed once,
so the input pattern is not an established cause or sufficient remedy. Two worker starts with
`--worker-debug` passed; use that measured path for now. Logging is not claimed as a root-cause fix.
The precise SDK/transport cause is unresolved.
No hidden retry or reboot occurs during inference. Empty, malformed, or mismatched results stop.

If startup fails, preserve its metrics and inspect board ownership. The previous session's explicit
RAM-only clearing command is available, but is not silently invoked by the live loop:

```bash
docker exec singalong_sdk sh -c 'python3 "$SPINE_DIR/tools/c1util.py" -S /workspace/data/runs/A_vocalset_P_talamo/app_hostfed_125/bin/c1/release/app.c1boot'
```

The frozen voice demo is independent and unchanged:
`cd ~/TUM/NeuroTUM/demo-voice-v1 && ./run.sh`.
Its image/checksum prerequisites were checked, without unnecessarily running the restore command.

## Validation record

See `TRAINING_FEEDBACK.md` section 6 and the sibling JSON files for current measured status.
Do not infer live-chip success from the baseline or mock tests. The original 1,451-window results
and original model artifacts remain unchanged.
