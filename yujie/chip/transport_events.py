"""Track and consume native asynchronous events across Talamo pipeline calls.

Talamo 4.7.1 creates a fresh MocktcpManager per call. The manager can return
after data/reset completion with ping events still pending. Native event IDs
are eventfd file descriptors; consuming completion releases their requests.
This proxy wraps only the existing connection, never reconnects or reboots.
"""
import time


class TrackedEvents:
    def __init__(self, native):
        self.native = native
        self.pending = set()
        self.drained_total = 0
        self.max_event_id = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _track(self, event):
        if event in self.pending:
            raise RuntimeError(f'Native event ID reused before completion: {event}')
        self.pending.add(event)
        self.max_event_id = max(self.max_event_id, event)
        return event

    def queue_read(self, *args, **kwargs):
        return self._track(self.native.queue_read(*args, **kwargs))

    def queue_write(self, *args, **kwargs):
        return self._track(self.native.queue_write(*args, **kwargs))

    def wait_events(self, *args, **kwargs):
        result = self.native.wait_events(*args, **kwargs)
        if result is None:
            raise RuntimeError(f'Native wait_events returned no completion; {len(self.pending)} events pending')
        self.pending.discard(result[0])
        return result

    def drain(self, timeout=1.0):
        """Consume leftover ping completions only after data + reset succeeded.

        Do not call the manager's ping handler: that can enqueue another ping.
        A timeout or unexpected channel stops the session, never hides failure.
        """
        deadline = time.perf_counter() + timeout
        drained = 0
        while self.pending:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise RuntimeError('Timed out draining completed-window ping events')
            _, channel, reason, payload = self.wait_events(sorted(self.pending), remaining)
            if channel not in (1000, 1001) or reason not in ('read', 'write'):
                raise RuntimeError(f'Unexpected leftover event: channel={channel}, reason={reason}, result={payload!r}')
            if channel == 1001 and payload != b'pong':
                raise RuntimeError(f'Unexpected ping response: {payload!r}')
            drained += 1
        self.drained_total += drained
        return drained
