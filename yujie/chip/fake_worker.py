#!/usr/bin/env python3
"""Stand-in for board_worker.py in software tests. No board, no SDK.

Answers two windows with silence, then misbehaves as told:
  hang      stops answering and ignores end of input (dies on SIGTERM)
  exit      exits
  stubborn  stops answering, ignores end of input and SIGTERM
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
for count, line in enumerate(sys.stdin):
    if count == 2:
        break
    request = json.loads(line)
    print(json.dumps(dict(type='result', id=request['id'], spikes=silence, inference_seconds=0.01,
                          exact_reference=None)), flush=True)
if mode == 'exit':
    sys.exit(0)
while True:
    time.sleep(1)
