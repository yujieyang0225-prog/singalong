# How the sing-along network got onto the chip: the process, step by step

Written 2026-10-08, continued the same evening (steps 6 to 11). This is the story of Phase B in the order it happened: what was tried, what
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

**E. Multi-minute stability. Failed (board), and the failure is a fixed limit (solved in step 6).** The run stopped after
337 windows. A direct test with back-to-back windows and no audio stopped after 338 with the same
error. The same count twice means the limit belongs to the connection: about 2 minutes 49 seconds of
chip operation per boot at one update every half second. Until the limit, every window was exact.

The software-only baseline path ran for 185 seconds without a problem and is one option away.

## Step 6. The limit and the start-up failure, read from the SDK's source

- **The 338-window limit.** Talamo creates a new connection manager for every call and never collects
  its last ping replies. Each one keeps an operating-system event handle open: three per window. The
  SDK's native library waits on these handles with `select()`, which only works for handle numbers below
  1,024. With about 11 handles open at rest, (1,024 − 11) / 3 ≈ 338. A parallel session added
  `transport_events.py`, which collects the leftover replies after each window (commit `1f96bf7`).
  **Board:** the 185 s run then completed, 366 of 366 windows exact, with the number of open handles
  constant at 11. Later the same evening 1,451 windows ran on one boot.
- **The start-up failure.** After a boot, host and board do a handshake in the native library. The
  board broadcasts until it is answered; the host answers once and never repeats it. If that exchange
  fails, nothing recovers on that boot, and the SDK reports "waiting for sync" after its pings time out.
  The bridge's log shows that after every boot the SDK reconnects and switches the serial speed to
  115,200 and back to 1,000,000 while the board is already sending. Whether that is what breaks the
  exchange is **not established**. It fails in roughly one start in three.

## Step 7. A demo that can be seen and heard

- Milestone D had "passed", but the project owner had seen and heard nothing. Taken apart, it had shown
  that MIDI messages were generated and that a GarageBand port accepted them. Nobody had checked that
  GarageBand did anything with them, and nobody had listened.
- GarageBand's last project has one audio track and no instrument, so MIDI had nothing to play. That is
  read from the project file; it is not proven that this project was open during the test.
- This session could start GarageBand and see its MIDI port, but could not see or click inside it.
- **Decision (by the project owner):** drop the GarageBand dependency. FluidSynth and a SoundFont were not
  installed, so a small synth on `sounddevice` came first (`synth_sink.py`): nothing to install, sound made
  in the audio callback thread. **The owner heard it.**
- `live_view.py` serves a page with level, note, mode, round trip, delay and errors. It wraps the loop
  from outside: the sink and the selector get the same calls as before.

## Step 8. The goal changes: the sung note, not an accompaniment

- The loop had been built around the team's score follower, which knows Amazing Grace and plays the
  chord written over the current position. The owner's goal is different: sing any note, hear that note.
- `pitch_tracker.py` replaces the selector for this (`--selector pitch`); everything up to the chip's
  output is unchanged. The network has twelve pitch-class outputs and no octave, so the octave is chosen
  on the laptop from the 48 bands.
- **Software and board:** a generated tone sequence is tracked note for note, octave included, in both
  modes. On a recording of singing, chip and baseline agree on the note in 47 % of frames. Nothing has
  been scored against known notes.
- **Why the note is a second late on the chip.** Not computation: a 2 s window is sent every 0.5 s, the
  round trip is 0.38 s (most of it the serial transfer of the window), and every frame is played at a
  fixed delay that covers both. One frame of the network is 64 ms of audio, which is where shorter figures
  come from; they would apply to a chip application that is fed frame by frame and keeps its state.

## Step 9. Making runs safe, one approved step at a time

From here the owner asked for a fixed routine: inspect, propose with measurable criteria, implement only
after approval, report the actual results including failures, stop.

| Step | Why | Result |
| --- | --- | --- |
| 1. Board lock | a check started during a live run crashed its chip application | second user refused before anything is sent; first session unharmed, 60 of 60 exact (board) |
| 2. Worker timeout and failure detection | the same incident left the loop hanging for 5 minutes: it closed the worker's output pipe while a thread was still reading it | frozen worker: error on the page after 3.0 s, program exits, nothing force-killed (board) |
| 3. Notes off and metrics | notes were released only after waiting for threads, metrics were written last and could be lost | notes off 0.2 s after a failure, metrics kept, partial file every 10 s (board and software) |

One criterion stayed open: whether the 10 s checkpoint costs late frames. Three 60 s runs each way gave
1, 0, 2 late frames before and 19, 0, 5 after, on a Mac under changing load. `--checkpoint 0` turns it off.

## Step 10. Simon's model trained under the chip's rules

- `A_vsP_chip_shift3` is the answer to the feedback of step 2: same network, trained with the chip's rules.
- **Software:** import check passed; expected accuracy 0.577 / 0.591 / 0.378.
- **Board:** all 1,451 test windows identical to the reference; accuracy 0.577 / 0.591 / 0.378, against
  0.516 / 0.589 / 0.364 for `A_vocalset_P`. Same RAM (23.23 %) and the same time per window.
- It is an option (`--model`), not the default. Three 3-minute live runs with the owner singing completed,
  two on this model and one on `A_vocalset_P`; what they sounded like was not recorded, and the takes
  were not sung alike.
- `HARDWARE_BENCHMARK.md` collects the figures for the presentation.

## Step 11. A text on the board's display

- Wanted: something on the board that shows it is working, without making the demo slower. Showing the
  detected note would redraw the display in every window; a fixed text is drawn once. The fixed text was chosen.
- The SDK's display step expects a number to look up and ends the pipeline, so it could not simply be
  added. `banner/` holds a small pass-through step instead: it copies each frame and draws the text with
  the first one. Its first version made the SDK buffer whole windows; declaring the same frame-by-frame
  input as the encoder fixed that.
- **Board:** 300 random windows identical, median 0.366 s; a recording through the live loop 46 of 46 with
  and without the banner. The owner saw the text.
- The whole test set took four tries. Through `board_check.py`, the first try matched two sets (939
  windows) before a start failed and the script's clear-and-reboot was rejected by the board's loader;
  the second try failed the same way at its first start. On one boot with the SDK's logging off, the run
  stalled between window 300 and 399. On one boot with logging on, as in the live loop: **1,451 of 1,451**.

## Mistakes in steps 6 to 11, and what they changed

1. A board check was started while a live run was using the board. It crashed the run. Hence the lock.
2. `eval_modes.py` and `board_check.py` write tracked result files in place; running them for the new model
   overwrote the old model's records. Both were restored from git; the quirk is now written down.
3. A failed run was first described from the last lines of its log ("failed at its first boot"). The whole
   log showed two sets had matched before the failure. The report was corrected.
4. A test script waited on the board without a time limit and saved results only at the end. It sat for
   34 minutes and kept nothing. It now has a limit per window and saves after every set.
5. Two test commands were malformed and ran nothing. Both were noticed from their output and rerun.
6. The on-chip front-end was described as firmware still to be written. A portable C version already
   exists in `jan/singalong/kernels_c/bands/`; it has not been built for the board.

## Where it stands

| | |
| --- | --- |
| Weights on the chip compute the same as the reference | proven on the board for both models, 1,451 windows each |
| Live loop, microphone to sound through the chip | works; longest single boot 1,451 windows; stops cleanly on any board fault |
| Start-up | handshake fails about one time in three; retried automatically, not fixed |
| The sung note on a synth, live view, text on the display | working; sound and display confirmed by the owner |
| How well it tracks a person singing | **not measured** |
| Delay from voice to note on the chip | about 1 s, by design of the window scheme |

## What was learned along the way

1. **Check that an import is actually used.** Compare the pipeline's output with a direct call and
   with the reloaded file. The random-weights bug survived every earlier check.
2. **Score spike trains by spikes, not by equal entries.**
3. **Read the SDK's own code for the rules.** Two notes and one script default described the encoder
   wrongly; the code settled it.
4. **Prove the chip against a reference first, then build the application around it.** After step 4,
   every later problem could be assigned to the transport or the timing, never to the network.
5. **A count that repeats points at a resource limit.** The 338-window stop looked like a stability
   problem until it was reproduced without audio. The count itself, three handles per window up to 1,024,
   then named the resource.
6. **Keep a frozen fallback.** The voice demo stayed runnable throughout, outside the repository.
7. **Say what a test used.** "Prerecorded audio passed" meant a test tone; "microphone passed" meant
   room noise. Both are true and neither is singing.
8. **"Passed" has to say what a person saw or heard.** A message that was sent is not a sound.
9. **Check the goal before polishing the route.** The loop followed one song very carefully; the owner
   wanted the sung note.
10. **One user of the board needs a lock, not care.**
11. **Every wait on the board needs a time limit, in test scripts too, and results should be saved as
    they come.**
12. **Add new things as options and leave the validated path alone.** Each change could be compared with
    the run before it, and undone.
13. **Test in the configuration the demo uses.** The long run passed with the SDK's logging on, as the
    demo has it, and stalled with it off.

## Open, in order

1. Have a person sing and write down what is heard; for a number, one recorded take with known notes.
2. The delay of about 1 s: accept it, or build a chip application that is fed in shorter pieces.
3. The start-up handshake, which is retried but not understood.
4. Why a long run without the SDK's logging stalled.

Details and the current commands: `NEXT_SESSION.md`.

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
| 6 | `transport_events.py`, `test_transport_events.py`, `persistent_event_cleanup_check.json`, `chip_stability_event_cleanup.json`, `chip_microphone_event_cleanup.json`, `startup_failure_event_cleanup.json` |
| 7 | `synth_sink.py`, `live_view.py`, `midi_check.py` |
| 8 | `pitch_tracker.py`, `results/live_2026-10-08/pitch_tracker_*.json` |
| 9 | `board_lock.py`, `test_board_lock.py`, `fake_worker.py`, `test_worker_failure.py`, `test_shutdown.py`, `results/live_2026-10-08/step1_*.json` to `step3_*.json` |
| 10 | `handoff/runs/A_vsP_chip_shift3/`, `results/A_vsP_chip_shift3/`, `HARDWARE_BENCHMARK.md`, `results/live_2026-10-08/demo_*.json`, `step4_*.json` |
| 11 | `build_banner.py`, `banner/`, `check_banner.py`, `banner_check.json`, `results/A_vsP_chip_shift3/banner_*.json`, `results/live_2026-10-08/step5_*.json`, `final_smoke.json` |
| All | `NEXT_SESSION.md`, `LIVE_CHIP.md`, this file |
