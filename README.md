# Sing-along on a neuromorphic chip (Innatera Pulsar C1)

Team project from the **NeuroTUM Hackathon 2026** (Innatera × Sofilab challenge), pitched on 9 October 2026.

A person sings or hums a note. A small recurrent spiking neural network (SNN) running on the Innatera Pulsar C1
finds the pitch class, the laptop picks the octave, and a synthesizer plays the note back in real time.

## My part (Yujie Yang)

I took the team's trained network from the laptop to the physical chip and made it run live:

- Found and fixed why the trained weights no longer matched after import into Innatera's toolchain.
- Built the chip applications, including a C kernel that writes to the board's display.
- Wrote the live loop: microphone -> chip -> note tracking -> synthesizer, with a live web page and a piano roll.
- Traced a crash after 338 windows to event handles leaked by the SDK (3 per call, until the 1,024 limit of
  `select()`) and fixed it in our code: 1,451 of 1,451 windows then ran on one boot.
- After the pitch, built a simulated chip that reproduces the board's output, so the system can be developed without
  the hardware.

About 3,900 lines of Python and C and 63 tests, in [`yujie/chip/`](yujie/chip/README.md).

## How it works

```
microphone -> front-end on the laptop (48 semitone bands, one frame per 16 ms)
  -> latest 2 s window every 0.5 s -> board worker -> Pulsar C1
     (IF encoder + recurrent SNN 48 -> 80 -> 24, RAM boot, fixed text on the board's display)
  -> spikes of outputs 12..23 -> rate over 8 frames -> pitch tracker on the laptop
     (pitch class from the chip, octave from the 48 bands) -> one-voice renderer
  -> synthesizer and a live page with a piano roll (http://127.0.0.1:8765)
```

## Results, measured on the board

| What | Result |
| --- | --- |
| Test windows identical to the integer reference | 1,451 of 1,451 (plain application and with the display text) |
| Pitch-class accuracy: VocalSet songs / VocalSet exercises / DAMP | 0.577 / 0.591 / 0.378 (chance is about 0.08) |
| Chip RAM used | 23.23 % (25.59 % with the display text) |
| Round trip per 2 s window | about 0.38 s (median) |

Model: `A_vsP_chip_shift3` (`handoff/runs/`). Details and how each number was measured:
[`yujie/chip/NEXT_SESSION.md`](yujie/chip/NEXT_SESSION.md) and
[`yujie/chip/HARDWARE_BENCHMARK.md`](yujie/chip/HARDWARE_BENCHMARK.md).

## Who did what

| Folder | Author | What |
| --- | --- | --- |
| [`yujie/`](yujie/chip/README.md) | Yujie Yang | Live loop on the physical chip (`live_view.py`, `live_chip.py`), board worker and tooling, chip applications including a C kernel that writes to the board's display, the simulated chip (`--mode reference`), software tests, documentation |
| [`jan/`](jan/singalong/README.md) | Jan Vitzthum | Audio front-end (48 semitone bands, also as a C kernel), the SNN model and its training (PyTorch and Innatera's Talamo), the first laptop demo |
| [`simon/`](simon/), [`handoff/`](handoff/README.md) | Simon Appoltshauser | Dataset preparation, training runs and hyper-parameter search on the cluster, evaluation, the trained models |

The full commit history is kept: `git shortlog -sn` shows who committed what.

## Run it without the board

Tested on macOS with conda and `ffmpeg`.

```bash
conda env create -f yujie/chip/snn.environment.yml    # creates the environment "snn"
conda activate snn
mkdir -p data/live_chip
python yujie/chip/make_test_audio.py data/live_chip/tones.wav --seconds 190
```

The network on a **simulated chip** (an integer reference of the chip, no board), on the generated test tones;
open http://127.0.0.1:8765 while it runs:

```bash
python -u yujie/chip/live_view.py --source wav:data/live_chip/tones.wav --mode reference --model A_vsP_chip_shift3 --selector pitch --playout-delay 1.25 --sink print --duration 20
```

The software tests, from `yujie/chip`:

```bash
for t in windowing transport_events worker_failure shutdown slow_round_trips live_view reference_worker; do python test_$t.py; done
```

Two tests skip because their data is not in this repository (see below). `test_board_lock.py` runs in the SDK
container. Running on the physical chip needs the Pulsar C1 evaluation kit and Innatera's SDK, which are not public:
[`yujie/chip/PITCH_DAY.md`](yujie/chip/PITCH_DAY.md).

## Not included, and third-party material

- **Innatera's SDK (Talamo) and starter repository** are proprietary and not included. The code imports them where
  it talks to the chip. Innatera, Pulsar and Talamo are names of Innatera Nanosystems.
- **DAMP (Smule) recordings and features computed from them** are not included: their research licence does not allow
  redistribution. Scores measured on them are reported. One test skips without them.
- **Song recordings, scores and MIDI files used during the hackathon** are not included (third-party licences). One
  test skips without the Happy Birthday recording.
- **The team's notes and the organisers' material** are not included.
- **VocalSet** (J. Wilkins, P. Seetharaman, A. Wahl, B. Pardo, "VocalSet: A Singing Voice Dataset", ISMIR 2018;
  [doi:10.5281/zenodo.1193957](https://doi.org/10.5281/zenodo.1193957), CC BY 4.0): the models were trained on it,
  and `handoff/testdata/vocalset/` holds band features computed from it.
- The board-side C code in `yujie/chip/banner/` uses the SDK's API and follows its display example for the pin set-up.

## About the documents and the history

This is a public copy of the branch `yujie/singalong-chip` of the team's private repository. It keeps the commits of
the code, the models and the documents, and leaves out the material listed above. The documents in `yujie/chip/` are
the working notes written during the hackathon. Some refer to the team's local set-up and to branches that are not
in this repository. Commit IDs in them point to this repository, except `f7cd2fd`, `4917b31` and `fd3e4cb`, which
exist only in the team's repository. The tag `pitch-2026-10-09` marks the version shown at the pitch.

## Licence

No licence is granted: all rights reserved by the authors. The code is published to be read; copying or reusing it
needs the authors' permission.
