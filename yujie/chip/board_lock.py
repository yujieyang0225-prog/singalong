#!/usr/bin/env python3
"""One user of the Pulsar board at a time, for everything in this folder that opens it.

An operating-system lock on one file inside the SDK container. It is taken without waiting:
a second user fails at once and is told who holds the board. The system drops the lock when
the holder exits, however it exits, so there are no stale locks.

Not covered: the frozen voice demo (other container) and tools that do not use this module.

Run a command under the lock (exit code 75 if the board is busy):

    python3 board_lock.py "clear board" -- python3 "$SPINE_DIR/tools/c1util.py" -S app.c1boot
"""
from contextlib import contextmanager
import fcntl
import json
import os
import subprocess
import sys
import time

PATH = os.environ.get('PULSAR_BOARD_LOCK', '/tmp/pulsar_board.lock')
BUSY_EXIT = 75


class BoardBusy(RuntimeError):
    pass


def acquire(purpose):
    """Take the lock or raise BoardBusy. Keep the returned file open for as long as the board is used."""
    handle = open(PATH, 'a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        holder = handle.read().strip() or 'unknown holder'
        handle.close()
        raise BoardBusy(f'The board is in use: {holder}. Stop that run first; nothing was sent to the board.') from None
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(dict(pid=os.getpid(), purpose=purpose, command=' '.join(sys.argv)[:200],
                                 since=time.strftime('%Y-%m-%d %H:%M:%S'))))
    handle.flush()
    return handle


def release(handle):
    if handle is not None and not handle.closed:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


@contextmanager
def held(purpose):
    handle = acquire(purpose)
    try:
        yield
    finally:
        release(handle)


if __name__ == '__main__':
    if len(sys.argv) < 4 or sys.argv[2] != '--':
        sys.exit(__doc__)
    try:
        with held(sys.argv[1]):
            sys.exit(subprocess.run(sys.argv[3:]).returncode)
    except BoardBusy as busy:
        print(busy, file=sys.stderr)
        sys.exit(BUSY_EXIT)
