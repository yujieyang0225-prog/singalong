#!/usr/bin/env python3
"""Stand-in for board_worker.py in software tests. No board, no SDK.

Answers two windows with silence, then misbehaves as told:
  hang      stops answering and ignores end of input (dies on SIGTERM)
  exit      exits
  stubborn  stops answering, ignores end of input and SIGTERM
  delays    answers every window, some of them late. The next argument says which, counting
            requests from 0: "5:0.55,6:1.6" delays requests 5 and 6, "4+:0.6" every request from 4 on
"""
import json
import os
import signal
import sys
import time

mode = sys.argv[1]
if mode == 'stubborn':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(json.dumps(dict(type='hello', pid=os.getpid())), flush=True)
print(json.dumps(dict(type='ready', mode='fake-worker')), flush=True)
silence = [[0] * 24 for _ in range(125)]
late, late_from = {}, None
if mode == 'delays':
    for item in sys.argv[2].split(','):
        which, seconds = item.split(':')
        if which.endswith('+'):
            late_from = (int(which[:-1]), float(seconds))
        else:
            late[int(which)] = float(seconds)
for count, line in enumerate(sys.stdin):
    if count == 2 and mode != 'delays':
        break
    request = json.loads(line)
    if mode == 'delays':
        time.sleep(late.get(count, late_from[1] if late_from and count >= late_from[0] else 0))
    print(json.dumps(dict(type='result', id=request['id'], spikes=silence, inference_seconds=0.01,
                          exact_reference=None)), flush=True)
if mode in ('exit', 'delays'):
    sys.exit(0)
while True:
    time.sleep(1)
