"""Software-only regression test for async events abandoned by a per-call manager."""
import unittest
from transport_events import TrackedEvents


class FakeNative:
    def __init__(self):
        self.events = {}
    def queue_write(self, channel, *_):
        return self.add(channel, 'write', None)
    def queue_read(self, channel, *_):
        return self.add(channel, 'read', b'pong')
    def add(self, channel, reason, data):
        event = next(i for i in range(10, 20) if i not in self.events)
        self.events[event] = (event, channel, reason, data)
        return event
    def wait_events(self, ids, timeout_secs=None):
        return self.events.pop(ids[0])


class EventTests(unittest.TestCase):
    def test_repeated_calls_do_not_exhaust_event_slots(self):
        native = FakeNative()
        tracked = TrackedEvents(native)
        for _ in range(500):
            data = tracked.queue_write(1, b'data')
            tracked.queue_read(1001)
            tracked.queue_write(1000, b'ping')
            tracked.wait_events([data])
            # The SDK returned without waiting for the pings.
            self.assertEqual(tracked.drain(), 2)
            self.assertFalse(native.events)
        self.assertEqual(tracked.drained_total, 1000)
        self.assertLess(tracked.max_event_id, 20)

    def test_unconsumed_data_is_not_silently_discarded(self):
        tracked = TrackedEvents(FakeNative())
        tracked.queue_write(1, b'data')
        with self.assertRaises(RuntimeError):
            tracked.drain()

    def test_native_no_completion_fails_clearly(self):
        class NoCompletion(FakeNative):
            def wait_events(self, *_):
                return None
        tracked = TrackedEvents(NoCompletion())
        tracked.queue_read(1001)
        with self.assertRaisesRegex(RuntimeError, 'no completion'):
            tracked.drain()


if __name__ == '__main__':
    unittest.main()
