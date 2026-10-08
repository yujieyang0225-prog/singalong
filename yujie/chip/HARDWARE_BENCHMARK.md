# Pulsar C1 hardware benchmark: A_vsP_chip_shift3

Collected 2026-10-08 from the board checks of that evening. Every figure says where it was measured: **board**
(physical Pulsar C1 EVK, RAM boot), **software** (Talamo 4.7.1 model or the integer reference on a
PC), or **file** (read from the weight file or build output). Nothing here was estimated.

Model: `handoff/runs/A_vsP_chip_shift3/weights_int8.npz`, md5 `6b47d452dddcd1cdddbe8d4f2a152035`,
trained by Simon under the chip's rules. Imported with `IMPORT CHECK PASSED`.

## Network and build

| Item | Value | Source |
| --- | --- | --- |
| Architecture | 48 inputs → 80 recurrent hidden → 24 outputs (104 neurons) | file: build report |
| Synapses in use | 10,864 of 12,160 possible (3,758 input, 6,180 recurrent, 926 output) | file: weights |
| Largest fan-in | 128, which is the chip's limit | file: weights |
| Weights | 8-bit integers, thresholds 127 | file: weights |
| RAM | 91,360 bytes of 384 KB = 23.23 % | file: build log |
| Boot image | 60,664 bytes | file: build report |
| Clock | 100 MHz | file: build report |
| Boot | RAM boot, 5.9 to 6.0 s; nothing is flashed | board |

## Hardware–software consistency (board)

| Test set | Windows | Identical to the reference | Spikes, board / reference |
| --- | ---: | ---: | ---: |
| VocalSet songs | 427 | 427 | 26,267 / 26,267 |
| VocalSet exercises | 512 | 512 | 27,079 / 27,079 |
| DAMP | 512 | 512 | 37,997 / 37,997 |
| **Total** | **1,451** | **1,451** | **91,343 / 91,343** |

No missing or extra spike in any window; no retries during the run.

## Recognition accuracy

Pitch class per 16 ms frame, on sung frames of 2-second windows. Chance is about 8 %.

| Test set | Board | Software, chip rules | Software, as trained |
| --- | ---: | ---: | ---: |
| VocalSet songs | 57.7 % | 57.7 % | 59.3 % |
| VocalSet exercises | 59.1 % | 59.1 % | 61.9 % |
| DAMP | 37.8 % | 37.8 % | 37.5 % |

These are dataset figures. There is no accuracy figure for live singing.

## Host–chip round trip per 2-second window (board)

| Run | Windows | Median | 95th percentile | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Recording through the live loop | 46 | 0.391 s | 0.431 s | 0.492 s |
| Live microphone, take 1 | 356 | 0.383 s | 0.415 s | 0.485 s |
| Live microphone, take 2 | 356 | 0.388 s | 0.426 s | 0.654 s |

- Measured inside the worker only, without the link to the Mac process: median 0.37 s.
- Full test set: 531.8 s for 1,451 windows, 0.367 s per window including three boots.
- The round trip includes sending the window over a 1 Mbaud serial link, the chip's computation,
  the reply and the reset. These parts were not measured separately.

## Real-time performance (board)

| Item | Value |
| --- | --- |
| Throughput | 2 s of audio in about 0.38 s, roughly 5 times faster than real time |
| Update period | one window every 0.5 s |
| Sound-to-note delay | 1.003 s median, with a planned delay of 1.0 s |
| Late frames (more than 50 ms) | 0 of 1,439; 0 of 11,126; 40 of 11,126 (0.4 %) in the three runs |
| Skipped updates | 0 in all three runs |
| Windows exact in the live loop | 46 of 46 where checked; the two microphone runs were not checked |

The 40 late frames all follow the single 0.654 s round trip.

## Spike activity (board, full test set)

| Item | Value |
| --- | --- |
| Output spikes | 91,343 over 1,451 windows |
| Per window | 63 spikes |
| Per 16 ms frame | 0.50 spikes across the 12 pitch outputs |
| Share of pitch-output entries with a spike | 4.2 % |
| Chord outputs (0 to 11) | silent in every window |
| Silence in | zero spikes out (one-boot check) |
| Full-scale input | 221 spikes in one window (one-boot check) |

Hidden-layer activity on the board was not measured: the chip returns only output spikes.

## With the display banner (board)

A second build of the same network shows a fixed text on the EVK display ("SING-ALONG SNN / NeuroTUM /
Pitch network / running on / Pulsar C1"). One small pass-through step in front of the network draws it
once after boot; the data is handed on unchanged.

| Item | Plain application | With display banner |
| --- | --- | --- |
| RAM | 91,360 bytes, 23.23 % | 100,640 bytes, 25.59 % |
| Boot image | 60,664 bytes | 68,576 bytes |
| Test set, windows identical to the reference | 1,451 of 1,451 (three boots) | 1,451 of 1,451 (one boot) |
| Accuracy: songs / exercises / DAMP | 57.7 / 59.1 / 37.8 % | 57.7 / 59.1 / 37.8 % |
| Time per window | 0.367 s including three boots | 0.366 s median, 0.388 s 95th percentile, 0.554 s maximum |
| 300 random windows | not run | 300 of 300 identical, median 0.366 s |
| Recording through the live loop | 46 of 46 identical, 0 late frames | 46 of 46 identical, 0 late frames |

The banner run of the whole test set is also the longest session on a single boot: 1,451 windows,
547 s. It ran with the SDK's communication logging on, as every live run does. One earlier attempt with
logging off stalled between window 300 and 399 and saved nothing; the cause is unknown.
The text was seen on the display by the user. It stays there until the board is reset or powered off.

## Not measured

- Power or energy.
- Time on the chip alone, separate from the serial transfer.
- Start-up reliability as a rate. With this model, 2 of 5 starts failed in the handshake on
  2026-10-08 and all were recovered by the automatic retry; that is too few to quote.

## Where the figures come from

All under `yujie/chip/`:

| Figure | File |
| --- | --- |
| Build, RAM | `results/A_vsP_chip_shift3/build_chip_125.json`, `build_banner.json` |
| Consistency, accuracy, spikes | `results/A_vsP_chip_shift3/board_check.json` |
| Software accuracy | `results/A_vsP_chip_shift3/chip_expectation.json` |
| One-boot check | `results/A_vsP_chip_shift3/persistent_check.json` |
| Banner: test set on one boot, 300 windows | `results/A_vsP_chip_shift3/banner_testset_check.json`, `banner_check.json` |
| Round trip, late frames | `results/live_2026-10-08/step4_new_pitch.json`, `demo_new_model_take1.json`, `demo_new_model.json` |
| Banner in the live loop | `results/live_2026-10-08/step5_banner.json`, `step5_plain.json` |

The built applications and the imported parameters are under the git-ignored `data/` folder and exist
only on the Mac they were built on; `NEXT_SESSION.md` has the commands that rebuild them.
