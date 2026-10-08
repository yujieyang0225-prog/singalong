# How the sing-along network got onto the chip: the process, step by step

Written 2026-10-08. This is the story of Phase B in the order it happened: what was tried, what
was found, what was decided and why. For the current state and how to continue, read
`NEXT_SESSION.md`. For numbers meant for whoever trains the next model, read `TRAINING_FEEDBACK.md`.
For running the live loop, read `LIVE_CHIP.md`.

Every result below says where it comes from: **board** (measured on the Pulsar EVK), **software**
(Talamo's model or the integer reference on a PC), or **untested**.

## The goal

Run the team's pitch network on the Pulsar chip while the audio front-end, the song follower and the
violin stay on a laptop ("shape 2"):

```
laptop microphone -> 48 semitone bands, one frame per 16 ms -> 2 s window (125 frames)
   -> chip: IF encoder + recurrent spiking network (48 -> 80 -> 24) -> output spikes per frame
   -> laptop: rate over 8 frames -> score follower -> violin (MIDI)
```

Two other shapes were set aside: everything on the laptop with the board only as a microphone
(easier, less of a chip demo), and everything on the chip (not feasible in the time).

## Step 0. Finding the right model

- A folder of weights called `B_damp_lr0.03` was looked at first. It was a chord-and-pitch model with
  pitch accuracy 0.24, and its files did not agree with each other about which epoch they were from.
- A teammate then pushed the branch `singalong-pitch-handoff` with a proper handoff: the demo model
  `A_vocalset_P` (pitch-only), the test data, the exact trainer commit, and an honest note that the
  import into Talamo did not behave like the reference.
- **Decision:** use `A_vocalset_P`; treat the first folder as obsolete. Work on a separate branch,
  `yujie/singalong-chip`, so the teammate's branch stays as handed over.
- **Practical step:** a second SDK container, `singalong_sdk`, because the existing one could only see
  another repository.

## Step 1. Why Talamo and the reference disagreed

The handoff reported "0.912 agreement, 895 output spikes against 416". Reproduced exactly, then taken
apart stage by stage (`diagnose_import.py`). Four things were found:

1. **The import had never taken effect.** The script wrote the weights into the model's live tensors,
   but Talamo's pipeline call reloads a stored parameter set. Every earlier check, and anything built
   from the saved file, had been running a randomly initialised network.
2. **The encoder was scaled by 128.** Talamo's quantization step leaves the encoder multiplying its
   input by 2^7; the trained thresholds are in unscaled units.
3. **The recurrent weight matrix was not transposed.** It is square, so the shape test that transposes
   the other matrices never fired.
4. **The agreement measure was misleading.** About 95 % of entries are silent in both spike trains, so
   unrelated trains score above 0.9. Scored on where the spikes are, the original result was 0.02.

After fixing 1 to 3, a small difference remained. Reading Talamo's own layer code explained it:

| | Trainer | Talamo's model of the chip |
| --- | --- | --- |
| Recurrent input | added with the input, one floor at 0 | applied after the neuron fires, with its own floor |
| Output layer | sees hidden spikes of the same step | sees them one step (16 ms) later |

With those two rules, the integer reference (`chipref.py`) reproduced Talamo spike for spike.

**A mistake made here and corrected:** it was first reported that the trainer used a different encoder
reset rule from the chip. That came from a README and from the default of a laptop reference script,
not from the trainer's code. The trainer resets to zero, like the chip. The correction is recorded in
the feedback file.

## Step 2. Fixing the import, and deciding what to do about the differences

- **Fixed** `import_torch_weights.py`: store the values properly, reset the encoder scale, transpose
  the recurrent matrix, and replace the check with one that compares four things for identical spikes:
  the pipeline, the steps called directly, the saved file reloaded, and the chip-rule reference.
- **Measured the cost of the two rule differences** (software, same crops and scoring as the team's
  evaluation): pitch accuracy 0.530 / 0.599 / 0.364 as trained, 0.516 / 0.589 / 0.364 under chip rules.
  About one point, mostly from the 16 ms delay.
- **Decision (by the project owner):** keep the differences. Do not change the model or retrain; write
  them down as feedback for the next training session. That became `TRAINING_FEEDBACK.md`, which is
  pushed after every update.

## Step 3. Building for the chip and checking it fits

- Built a host-fed application (`build_chip.py`): the PC sends 125 frames of 48 bands, the chip returns
  the output spikes of every frame.
- **Fit:** 104 neurons, 10,917 synapses in use, largest fan-in 128, which is exactly the limit. The
  mapper accepted it. Application RAM 23 %.
- **First board run:** 99 output spikes, every one in the same place as the reference.

## Step 4. Timing, then the whole test set on the board

- **Timing (board):** 0.34 s per 2 s window once booted; 6.6 s per call for the boot. So a live loop
  needs the board booted once and kept open.
- **Full comparison (board, `board_check.py`):** all 1,451 test windows identical to the reference,
  100,847 spikes, none missing or extra. Pitch accuracy on the board 0.516 / 0.589 / 0.364, equal to
  the software prediction.
- A first attempt returned nothing: a Talamo boot can fail to synchronise when a Talamo application is
  already running. The script now clears the board with the plain boot utility and retries.

At this point the chip was proven to compute what the reference computes, for this model.

## Step 5. The live loop (done in a following session, then rechecked)

Planned as five milestones.

**A. Keep the board booted. Passed (board).** A small adapter (`persistent_board.py`) uses Talamo's
private functions to boot once and send window after window. Six calls, all exact, including silence
after sound, which shows the network is reset between windows. Boot 5.9 s, first call 0.78 s, then
0.335 to 0.368 s. Rerun later with the same result.

**B. Sliding windows. Passed (software).** Every 500 ms the latest 125 frames are sent; the first 93
frames at start-up are dropped, and after that each new frame is delivered to the song follower once,
with its original timestamp. A one-second playout buffer keeps the 16 ms rhythm.

**C. Prerecorded audio through the chip. Passed (board), with caveats.** 20 of 20 windows exact,
median round trip 0.37 s. But several starts failed with an empty result before one worked, and the
runs that worked had the SDK's communication logging switched on. The audio was a generated test tone.

**D. Laptop microphone through the chip to MIDI. Passed (board), with caveats.** 26 of 26 windows
exact, 814 frames delivered, about 1.26 s from sound to MIDI, of which 1.25 s is the deliberate
playout delay. It was 15 seconds of room sound; nobody sang.

**E. Multi-minute stability. Failed (board), and the failure is a fixed limit.** The run stopped after
337 windows. A direct test with back-to-back windows and no audio stopped after 338 with the same
error. The same count twice means the limit belongs to the connection: about 2 minutes 49 seconds of
chip operation per boot at one update every half second. Until the limit, every window was exact.

The software-only baseline path ran for 185 seconds without a problem and is one option away.

## Where it stands

| | |
| --- | --- |
| Weights on the chip compute the same as the reference | proven on the board, 1,451 windows |
| Live loop, microphone to MIDI through the chip | works for up to about 2 min 49 s per boot |
| Start-up | unreliable; only validated with debug logging on |
| A person singing, and listening to the violin | **untested** |

## What was learned along the way

1. **Check that an import is actually used.** Compare the pipeline's output with a direct call and
   with the reloaded file. The random-weights bug survived every earlier check.
2. **Score spike trains by spikes, not by equal entries.**
3. **Read the SDK's own code for the rules.** Two notes and one script default described the encoder
   wrongly; the code settled it.
4. **Prove the chip against a reference first, then build the application around it.** After step 4,
   every later problem could be assigned to the transport or the timing, never to the network.
5. **A count that repeats points at a resource limit.** The 338-window stop looked like a stability
   problem until it was reproduced without audio.
6. **Keep a frozen fallback.** The voice demo stayed runnable throughout, outside the repository.
7. **Say what a test used.** "Prerecorded audio passed" meant a test tone; "microphone passed" meant
   room noise. Both are true and neither is singing.

## Open, in order

1. Decide how to live with the 338-window limit: short performances with a restart in between, a
   planned reconnect with a visible switch to the baseline, a slower update, or finding a way to
   reopen the connection without a reboot.
2. Measure start-up reliability over repeated starts.
3. Rehearse with a person singing and someone listening.

## Files, by step

| Step | Files in `yujie/chip/` |
| --- | --- |
| 1 | `diagnose_import.py`, `chipref.py` |
| 2 | `eval_modes.py`, `chip_expectation.json`, `TRAINING_FEEDBACK.md`; the import fix is in `jan/singalong/Pulsar/IF_DNR/import_torch_weights.py` |
| 3 | `build_chip.py` |
| 4 | `board_check.py`, `board_timing.json`, `board_check.json` |
| 5 A | `persistent_board.py`, `check_persistent.py`, `persistent_check.json`, `persistent_recheck.json` |
| 5 B | `windowing.py`, `test_windowing.py` |
| 5 C, D | `board_worker.py`, `live_chip.py`, `make_test_audio.py`, `prerecorded_check.json`, `chip_microphone.json`, the `startup_*.json` failure records |
| 5 E | `chip_stability.json`, `check_call_limit.py`, `call_limit.json`, the `baseline_*.json` fallback checks |
| All | `NEXT_SESSION.md`, `LIVE_CHIP.md`, this file |
