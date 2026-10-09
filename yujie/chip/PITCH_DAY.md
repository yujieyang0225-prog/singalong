# Pitch day: getting the demo up and keeping it up

Friday 9 October, pitch at 10:00. Written at 02:10 the night before, from what was measured that night.
Run everything from `~/TUM/NeuroTUM/agent-pitch-handoff`. Background: `NEXT_SESSION.md`, `LIVE_CHIP.md`.

## What is proven, and what is not

| | Status |
| --- | --- |
| A 3-minute run with the microphone on the chip | at least six runs completed |
| The longest single run so far | 20 minutes, to the end: 2,396 windows, every one identical to the reference, no stop, round trip steady at 0.40 s (speakers muted, nobody singing) |
| Anything longer than 20 minutes in one go | never run |
| Smooth timing | at least 3 minutes in every clean run; 11 minutes in the 20-minute run. After that the notes can start to arrive in half-second steps (issue 11; worked out from the loop, not judged by ear) |
| Sound output | Bluetooth: one clean run of three. Built-in speakers and wired earphones: inside the timing margin for 3 minutes in all three of their runs |

**A run stays up; what wears out is its timing.** In the 20-minute run the drift grew by about 0.03 s a
minute. Late frames began in minute 11, and from minute 14 nearly every window was late (7 % of all frames
by the end). With sound on, the 3-minute runs drifted faster than that. A fresh start takes under a minute
and brings the timing back to zero, so restart every 10 minutes at the latest, and sooner if "late frames"
in the footer starts to climb.

## Pack

- Laptop and charger. Keep it on power: on battery the Mac may sleep, and sleep drops the board connection.
- Pulsar board and its USB cable. Boot switch on UART.
- The sound output you will use, and wired earphones as a spare.
- The projector adapter.

## At the venue, in this order

1. **Power, board, projector.** Plug everything in first: connecting a projector or a speaker changes the
   Mac's sound devices.
2. **Keep the Mac awake** in a terminal that stays open:

   ```bash
   caffeinate -dimsu
   ```

3. **Bridge.** After a shutdown it is not running: nothing starts it by itself. Plug the board in first; if
   the board is replugged later, stop the bridge with Ctrl-C and start it again. Check, and start it in its
   own terminal window, which stays open, if the check prints nothing:

   ```bash
   pgrep -fl "evk-bridge -i"
   ```

   ```bash
   evk-bridge -i C1_027 --listen 0.0.0.0 --port 1700 --gdb-port 1701
   ```

4. **Docker and the container.** Docker Desktop does not start at login on this Mac, and the container does
   not come back by itself. Open Docker first and wait until the next command answers without an error
   (about a minute):

   ```bash
   open -a Docker
   ```

   ```bash
   docker ps --filter name=singalong_sdk --format '{{.Names}} {{.Status}}'
   ```

   If that prints nothing, or not "Up", start the container:

   ```bash
   docker start singalong_sdk
   ```

5. **Nothing else on the board** (prints nothing when the board is free):

   ```bash
   docker exec singalong_sdk sh -c 'ps -eo pid,args | grep -E "[b]oard_worker|[c]heck_|[c]1util|[b]oard_check"'
   ```

6. **Sound.** In System Settings, Sound: choose the output. The synth uses whatever is selected at the
   moment the demo starts, and macOS switches by itself when a projector or a Bluetooth device connects.
   Select "MacBook Pro Microphone" as input once to set its level; the demo always records from it.
7. **Close what you do not need** (other browsers and tabs, Teams, Zoom) and switch on Do Not Disturb. Slow
   moments on the Mac are what make round trips slow and the timing drift.
8. **Page check without the board** (20 s, silent). The piano roll must show a staircase of notes:

   ```bash
   python -u yujie/chip/live_view.py --source wav:data/live_chip/tones.wav --mode baseline --selector pitch --playout-delay 0.1 --sink print --duration 20
   ```

   ```bash
   open -a Safari http://127.0.0.1:8765
   ```

## The demo

Use the macOS Terminal app, as on the nights before: it already has permission to use the microphone.

Start it one to two minutes before you need it. Start-up takes 25 s, or about a minute when the first start
has to be retried. The last three starts each needed one retry; the four before them did not.

```bash
python -u yujie/chip/live_view.py --source "mic:MacBook Pro Microphone" --mode chip --worker-debug --model A_vsP_chip_shift3 --banner --selector pitch --playout-delay 1.25 --start-attempts 5 --slow-limit 6 --sink synth
```

- Open http://127.0.0.1:8765 in one browser only, in a wide or full-screen window.
- The page must read `CHIP · Pulsar C1` and "Running on the chip". On the Microphone card the level
  should move when you sing and read "Voice detected".
- Stop with Ctrl-C. If Ctrl-C does nothing, from another terminal:

  ```bash
  pkill -TERM -f yujie/chip/live_view.py
  ```

## Keeping it going

- **Restart between showings, and every 10 minutes at the latest.** Stop, start again, wait for "Running on
  the chip". The board itself held for 20 minutes; the restart is for the timing.
- **Watch the footer.** "late frames" climbing means the timing has drifted: restart. "connection errors"
  above the count it had after start-up means the board link had a problem.
- **If it stops by itself** the page shows the reason in a banner. Start the same command again. A run stops
  on six slow round trips in a row, on one stall of about 2 s, on no answer from the board for 3 s, or on a
  microphone glitch reported by macOS.
- **If the board will not start after its five attempts:** unplug and replug the board, restart the bridge
  (step 3), start the demo again.
- **If the board is lost for good:** the laptop-only command below needs no board. The page then reads
  `LAPTOP ONLY`, and you should say so.

  ```bash
  python -u yujie/chip/live_view.py --source "mic:MacBook Pro Microphone" --mode baseline --selector pitch --playout-delay 0.1 --sink synth
  ```

## Two settings that might stretch one run, both untested here

Both are existing options. Neither has been run in this setup, so try one for four minutes before relying on it.

- `--playout-delay 1.75` in place of `1.25`: the note comes half a second later, and the timing has about
  twice the room before frames are late.
- `--checkpoint 0`: no metrics file every 10 s while running. That file grows with the length of the run.
  Whether writing it costs late frames was never settled.

## Not to do

- No Bluetooth output unless you have just seen it run clean for three minutes.
- Never force-quit the demo or anything else that holds the board. Never flash the board.
- Do not let two programs use the board at once.
