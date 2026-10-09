# Sing-along on the Pulsar C1 (`yujie/chip`)

A person sings or hums a note; a small recurrent spiking network on the Pulsar C1 finds the pitch class; the laptop
picks the octave and a synth plays the note. Pitched 2026-10-09. This folder holds the live loop, the board
tooling, the checks, the results and the documents.

## Read in this order

| # | Document | Read it for |
| --- | --- | --- |
| 1 | [`NEXT_SESSION.md`](NEXT_SESSION.md) | The handoff: state, known issues, decisions, the working rules, what waits for approval |
| 2 | [`BOARDLESS.md`](BOARDLESS.md) | How to keep improving **without the board**, including the simulated chip |
| 3 | [`LIVE_CHIP.md`](LIVE_CHIP.md) | Every option, what the page shows, measured limits |
| 4 | [`PITCH_DAY.md`](PITCH_DAY.md) | Bring-up checklist for the demo with the board |
| 5 | [`HARDWARE_BENCHMARK.md`](HARDWARE_BENCHMARK.md), [`TRAINING_FEEDBACK.md`](TRAINING_FEEDBACK.md), [`PROCESS.md`](PROCESS.md) | Figures for slides, notes for the training team, how it came about |

## Run it

From the repository root (`agent-pitch-handoff`), with `PY=python`.

| What | Command |
| --- | --- |
| On the board (needs the board, `evk-bridge`, the container) | `$PY -u yujie/chip/live_view.py --source "mic:MacBook Pro Microphone" --mode chip --worker-debug --model A_vsP_chip_shift3 --banner --selector pitch --playout-delay 1.25 --start-attempts 5 --slow-limit 6 --sink synth` |
| **Simulated chip, no board** | `$PY -u yujie/chip/live_view.py --source "mic:MacBook Pro Microphone" --mode reference --model A_vsP_chip_shift3 --selector pitch --playout-delay 1.25 --sink synth` |
| Laptop only, no network | `$PY -u yujie/chip/live_view.py --source "mic:MacBook Pro Microphone" --mode baseline --selector pitch --playout-delay 0.1 --sink synth` |
| Silent, on a recording | any mode with `--source wav:<file> --sink print --duration 20` in place of the microphone source and the synth sink |

The page is http://127.0.0.1:8765. Tests (no board): `$PY test_<name>.py` from this folder for each `test_*.py`, and
`docker exec -w /workspace/yujie/chip singalong_sdk python3 test_board_lock.py` for the one that runs in the container.

## What is in here

| Part | Files |
| --- | --- |
| Entry points | `live_view.py` (the demo: the loop plus the page), `live_chip.py` (the loop alone) |
| The loop | `windowing.py`, `pitch_tracker.py` (note tracking, one-voice renderer), `synth_sink.py` |
| Board | `board_worker.py` (in the container), `persistent_board.py`, `transport_events.py`, `board_lock.py` |
| Simulated chip | `reference_worker.py`, `chipref.py` (the integer reference both the worker and the checks use) |
| Chip applications | `build_chip.py`, `build_banner.py`, `banner/` |
| Board checks | `board_check.py`, `check_banner.py`, `check_persistent.py`, `check_call_limit.py`, `diagnose_import.py`, `midi_check.py` |
| Evaluation | `eval_modes.py` (accuracy under the chip's rules, no board; overwrites the tracked `chip_expectation.json`) |
| Tests | `test_*.py`, `fake_worker.py` (a misbehaving stand-in for the board worker) |
| Results | `results/` by model and by run, `*.json` at the top (older evidence; the documents cite them) |
| Environment | `snn.environment.yml` (the host Python environment) |

## Git

| Branch | What it is |
| --- | --- |
| `yujie/singalong-chip` | The sing-along work. The pitch version is the tag `pitch-2026-10-09` (`151e96b`), the remote tip; `abb31e2` holds the documentation written after it |
| `yujie/reference-worker` | The simulated chip (`--mode reference`), started from `abb31e2`; each commit is one step, listed by `git log --oneline abb31e2..` |
| `yujie/multi-agent`, `yujie/agent-model`, `yujie/live-mic-voice` | The approach of 7 October and the voice demo; not used by the sing-along |
| `main`, `singalong-pitch-handoff` | The team's and Simon's. Never commit to them |

How commits are written here:

- **One step per commit**, with its tests: `chip:` for code, tests and results, `docs:` for documents only.
- The subject says what changed for a reader of `git log`; the body says what it does, what stays unchanged, and how
  it was checked, **labelled as measured on the board, in software, or estimated**.
- Documents and code that must agree go in the same series; the documents are updated last.
- Work on a `yujie/<topic>` branch. Nothing is pushed without the user's permission.
- Recorded audio and `data/` stay out of git; evidence that a test needs is committed as a small fixture with its
  provenance (`results/reference_worker/`).
