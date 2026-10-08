"""Software-only test of board_lock: no board, no SDK. Uses its own lock file."""
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
HOLD = "import board_lock, time; f = board_lock.acquire('test holder'); print('held', flush=True); time.sleep(60)"
TRY = "import board_lock; board_lock.release(board_lock.acquire('second')); print('free')"


class LockTests(unittest.TestCase):
    def setUp(self):
        self.env = dict(os.environ, PULSAR_BOARD_LOCK=os.path.join(tempfile.mkdtemp(), 'board.lock'))

    def run_py(self, code, **kw):
        return subprocess.Popen([sys.executable, '-c', code], cwd=HERE, env=self.env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)

    def holder(self):
        process = self.run_py(HOLD)
        self.assertEqual(process.stdout.readline().strip(), 'held')
        return process

    def attempt(self):
        started = time.perf_counter()
        process = self.run_py(TRY)
        out, err = process.communicate(timeout=20)
        return process.returncode, out.strip(), err, time.perf_counter() - started

    def test_second_user_is_refused_quickly_and_told_who_holds_it(self):
        holder = self.holder()
        try:
            code, out, err, seconds = self.attempt()
            self.assertNotEqual(code, 0)
            self.assertIn('BoardBusy', err)
            self.assertIn('test holder', err)
            self.assertIn(str(holder.pid), err)
            self.assertLess(seconds, 1.0)
        finally:
            holder.terminate()
            holder.wait()

    def test_free_after_normal_exit(self):
        holder = self.holder()
        holder.terminate()
        holder.wait()
        self.assertEqual(self.attempt()[:2], (0, 'free'))

    def test_free_after_abrupt_kill(self):
        holder = self.holder()
        holder.send_signal(signal.SIGKILL)          # a test helper process, never a board worker
        holder.wait()
        self.assertEqual(self.attempt()[:2], (0, 'free'))

    def test_command_wrapper_reports_busy_with_exit_75(self):
        holder = self.holder()
        try:
            done = subprocess.run([sys.executable, 'board_lock.py', 'wrapped', '--', 'true'], cwd=HERE, env=self.env,
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 75)
        finally:
            holder.terminate()
            holder.wait()
        done = subprocess.run([sys.executable, 'board_lock.py', 'wrapped', '--', 'true'], cwd=HERE, env=self.env)
        self.assertEqual(done.returncode, 0)


if __name__ == '__main__':
    unittest.main()
