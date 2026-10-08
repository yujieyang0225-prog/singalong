#!/usr/bin/env python3
"""Shape 2: Mac audio/frontend/selector/violin + persistent physical Pulsar inference.

Run with the host snn Python. Only board_worker runs inside singalong_sdk.
Use --mode baseline for the team's explicit software fallback.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import queue
import resource
import signal
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'jan/singalong'))
import live as host
from singalong import frontend as fe, evidence as ev, renderer as rd
from windowing import newest_rates, window_end


class ChipClient:
    def __init__(self, container, verify, debug=False):
        command = ['docker', 'exec', '-i', container, 'python3', '-u', '/workspace/yujie/chip/board_worker.py']
        if verify:
            command.append('--verify-reference')
        if debug:
            command.append('--comm-debug')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.messages = queue.Queue()
        def read():
            try:
                for line in self.process.stdout:
                    self.messages.put(json.loads(line))
            except Exception as error:
                self.messages.put(dict(type='error', error=str(error)))
            finally:
                self.messages.put(dict(type='error', error='Board worker exited'))
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            self.ready = self.receive(120)
            if self.ready.get('type') != 'ready':
                raise RuntimeError(f'Unexpected startup response: {self.ready}')
        except BaseException:
            self.close()
            raise

    def receive(self, timeout):
        try:
            result = self.messages.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError('Board timeout; no software substitution was made') from None
        if result.get('type') == 'error':
            raise RuntimeError(result['error'])
        return result

    def infer(self, x, index):
        t = time.perf_counter()
        self.process.stdin.write(json.dumps(dict(id=index, features=x.tolist())) + '\n')
        self.process.stdin.flush()
        response = self.receive(10)
        if response.get('type') != 'result' or response.get('id') != index:
            raise RuntimeError('Board response sequence mismatch')
        response['host_roundtrip_seconds'] = time.perf_counter() - t
        return response

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()  # EOF lets worker exit its board context
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # Do not SIGKILL the board holder. A blocked SDK process needs explicit recovery.
            print('ERROR: worker did not exit after EOF; inspect singalong_sdk before reuse', file=sys.stderr)
        self.process.stdout.close()


def summary(values):
    if not values:
        return None
    return dict(min=float(np.min(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(np.max(values)))


class LiveRun:
    def __init__(self, args):
        self.a = args
        self.stop = threading.Event()
        self.cv = threading.Condition()
        self.history = deque(maxlen=512)
        self.total = 0
        self.finished = False
        self.outputs = queue.Queue(maxsize=256)
        self.errors = queue.Queue()
        self.metrics = dict(mode=args.mode, source=args.source, period_seconds=args.period,
                            worker_debug=getattr(args, 'worker_debug', False),
                            playout_delay_seconds=args.playout_delay, windows=[], frontend_seconds=[],
                            output_latency_seconds=[], output_lateness_seconds=[], capture_status=[],
                            skipped_updates=0, startup_frames_omitted=0, duplicate_frames=0,
                            captured_frames=0, delivered_frames=0, late_outputs=0, queue_peak=0,
                            connection_failures=0, recoveries=0, feature_nonzero_frames=0,
                            pcm_peak=0.0, pcm_clipped_samples=0)
        self.epoch = None

    def fail(self, error):
        self.errors.put(error)
        self.stop.set()
        with self.cv:
            self.cv.notify_all()

    def add_pcm(self, front, pcm):
        self.metrics['pcm_peak'] = max(self.metrics['pcm_peak'], float(np.max(np.abs(pcm))))
        self.metrics['pcm_clipped_samples'] += int(np.count_nonzero(np.abs(pcm) >= 1))
        t = time.perf_counter()
        frames = list(front.feed(pcm))
        self.metrics['frontend_seconds'].append(time.perf_counter() - t)
        with self.cv:
            for bands, db, voiced in frames:
                self.history.append((self.total, bands, db, voiced))
                self.total += 1
                self.metrics['feature_nonzero_frames'] += int(bool(np.any(bands)))
            self.metrics['captured_frames'] = self.total
            self.cv.notify_all()

    def capture(self):
        try:
            front = fe.StreamingFrontEnd()
            kind, _, arg = self.a.source.partition(':')
            if kind == 'wav':
                y = fe.decode_audio(arg)  # existing ffmpeg mono -> exactly 16 kHz
                if self.a.duration:
                    y = y[:int(self.a.duration * fe.SR)]
                self.epoch = time.perf_counter()
                for start in range(0, len(y), fe.HOP):
                    chunk = y[start:start + fe.HOP]
                    if self.stop.wait(max(0, self.epoch + (start + len(chunk)) / fe.SR - time.perf_counter())):
                        break
                    self.add_pcm(front, chunk)
            elif kind == 'mic':
                import sounddevice as sd
                dev = int(arg) if arg.isdigit() else (arg or None)
                sd.check_input_settings(device=dev, channels=1, dtype='float32', samplerate=fe.SR)
                chunks = queue.Queue(maxsize=128)
                def callback(indata, frames, times, status):
                    if status:
                        self.metrics['capture_status'].append(str(status))
                        self.fail(RuntimeError(f'Audio capture discontinuity: {status}'))
                        return
                    # PortAudio ADC timestamps map input sample time to perf_counter.
                    if self.epoch is None:
                        self.epoch = time.perf_counter() + times.inputBufferAdcTime - times.currentTime
                    try:
                        chunks.put_nowait(indata[:, 0].copy())
                    except queue.Full:
                        self.fail(RuntimeError('Microphone queue overflow'))
                with sd.InputStream(samplerate=fe.SR, channels=1, dtype='float32', blocksize=fe.HOP,
                                    device=dev, callback=callback) as stream:
                    self.metrics['audio_device'] = str(sd.query_devices(dev, 'input'))
                    self.metrics['input_latency_seconds'] = stream.latency
                    self.metrics['sample_rate'] = stream.samplerate
                    samples = 0
                    while not self.stop.is_set():
                        try:
                            pcm = chunks.get(timeout=1)
                        except queue.Empty:
                            raise RuntimeError('No microphone samples within one second') from None
                        self.add_pcm(front, pcm)
                        samples += len(pcm)
                        if self.a.duration and samples >= self.a.duration * fe.SR:
                            break
            else:
                raise ValueError('Use --source wav:<path> or mic[:device]')
        except BaseException as error:
            self.fail(error)
        finally:
            with self.cv:
                self.finished = True
                self.cv.notify_all()

    def put(self, item):
        while not self.stop.is_set():
            try:
                self.outputs.put(item, timeout=.1)
                self.metrics['queue_peak'] = max(self.metrics['queue_peak'], self.outputs.qsize())
                return
            except queue.Full:
                pass

    def infer(self, client):
        try:
            update, last, slow = 0, None, 0
            baseline = ev.BaselineEvidence() if self.a.mode == 'baseline' else None
            while not self.stop.is_set():
                end = window_end(update, self.a.period) if client else update + 1
                with self.cv:
                    self.cv.wait_for(lambda: self.total >= end or self.finished or self.stop.is_set())
                    if self.stop.is_set() or self.total < end:
                        break
                    if client:
                        latest = max(update, int(math.floor((self.total - 125) / (self.a.period * fe.FPS))))
                        self.metrics['skipped_updates'] += latest - update
                        update = latest
                        end = window_end(update, self.a.period)
                        rows = [r for r in self.history if end - 125 <= r[0] < end]
                        if len(rows) != 125:
                            raise RuntimeError('Capture backlog exceeded bounded history')
                    else:
                        rows = [r for r in self.history if r[0] == update]
                        if len(rows) != 1:
                            raise RuntimeError('Baseline capture backlog')
                if client:
                    submitted = time.perf_counter()
                    try:
                        response = client.infer(np.stack([r[1] for r in rows]), update)
                    except BaseException:
                        self.metrics['connection_failures'] += 1
                        raise
                    indices, rates = newest_rates(response['spikes'], end, last, self.a.period)
                    if last is None:
                        self.metrics['startup_frames_omitted'] = int(indices[0])
                    timing = {k: response[k] for k in ('inference_seconds', 'host_roundtrip_seconds', 'exact_reference')}
                    timing.update(update=update, end_frame_exclusive=end, submitted_seconds=submitted - self.epoch)
                    self.metrics['windows'].append(timing)
                    for key in ('worker_cpu_seconds', 'worker_peak_rss_bytes'):
                        if key in response:
                            self.metrics[key] = response[key]
                    slow = slow + 1 if response['host_roundtrip_seconds'] > self.a.period else 0
                    if slow >= 3:
                        raise RuntimeError('Three consecutive round trips exceeded update period')
                    for index, rate in zip(indices, rates):
                        row = rows[int(index) - (end - 125)]
                        self.put((int(index), rate[12:24], row[2], row[3]))
                    last = int(indices[-1])
                else:
                    index, bands, db, voiced = rows[0]
                    melody, _ = baseline(bands, voiced)
                    self.put((index, melody, db, voiced))
                update += 1
        except BaseException as error:
            self.fail(error)
        finally:
            self.put(None)

    def run(self):
        client = None
        capture = inference = None
        sink = render = None
        started, cpu_start = time.perf_counter(), time.process_time()
        try:
            if self.a.mode == 'chip':
                print('Starting physical board worker (SDK import, one RAM boot, startup handshake)...', flush=True)
                try:
                    client = ChipClient(self.a.container, self.a.verify_reference, getattr(self.a, 'worker_debug', False))
                except BaseException:
                    self.metrics['connection_failures'] += 1
                    raise
                self.metrics.update(client.ready)
            print(f'ACTIVE INFERENCE MODE: {self.a.mode.upper()} | source={self.a.source} | melody only', flush=True)
            sink = host.make_sink(self.a.sink)
            hmm = host.make_selector(self.a.selector, self.a.bpm)
            render = rd.Renderer(sink, fps=fe.FPS)
            capture = threading.Thread(target=self.capture, daemon=True)
            inference = threading.Thread(target=self.infer, args=(client,), daemon=True)
            capture.start()
            inference.start()
            previous, label = None, None
            while not self.stop.is_set():
                try:
                    item = self.outputs.get(timeout=.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                index, melody, db, voiced = item
                if previous is not None and index != previous + 1:
                    self.metrics['duplicate_frames'] += int(index <= previous)
                    raise RuntimeError('Missing or repeated selector frame')
                previous = index
                # Frame i covers [i*HOP, i*HOP+WIN); preserve the chip's one-frame delay.
                audio_end = (index * fe.HOP + fe.WIN) / fe.SR
                deadline = self.epoch + audio_end + self.a.playout_delay
                if self.stop.wait(max(0, deadline - time.perf_counter())):
                    break
                lateness = max(0, time.perf_counter() - deadline)
                if lateness > .05:
                    self.metrics['late_outputs'] += 1
                self.metrics['output_lateness_seconds'].append(lateness)
                if hasattr(sink, 'set_time'):
                    sink.set_time(audio_end + self.a.playout_delay)
                state, post = hmm.step(melody, None, onset=True)
                render.step(hmm.pcs(state), root_pc=int(hmm.em.T[state].argmax()),
                            confidence=float(post[state]), level_db=db)
                self.metrics['output_latency_seconds'].append(time.perf_counter() - (self.epoch + audio_end))
                self.metrics['delivered_frames'] += 1
                if hmm.names[state] != label:
                    label = hmm.names[state]
                    print(f'{audio_end:8.3f}s audio | {self.a.mode} | {label} | notes={render.notes}', flush=True)
            if not self.errors.empty():
                raise self.errors.get()
            self.metrics['completed'] = True
        except BaseException as error:
            self.metrics.update(completed=False, error=f'{type(error).__name__}: {error}')
            raise
        finally:
            self.stop.set()
            with self.cv:
                self.cv.notify_all()
            for thread in (capture, inference):
                if thread:
                    thread.join(timeout=12)
            if render:
                render.all_off()
            if isinstance(sink, rd.PrintSink):
                self.metrics['note_ons'] = sum(e[0] == 'on' for e in sink.events)
            if sink and hasattr(sink, 'close'):
                self.metrics['rendered_file'] = str(sink.close())
            if isinstance(sink, rd.MidiSink):
                sink.out.close()
            if client:
                client.close()
            self.metrics['wall_seconds'] = time.perf_counter() - started
            self.metrics['host_cpu_seconds'] = time.process_time() - cpu_start
            # ru_maxrss is bytes on macOS, KiB on Linux.
            self.metrics['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            if self.epoch is not None:
                self.metrics['trailing_frames_without_full_update'] = self.total - (self.metrics['windows'][-1]['end_frame_exclusive'] if self.metrics['windows'] else self.metrics['delivered_frames'])
            for name in ('frontend_seconds', 'output_latency_seconds', 'output_lateness_seconds'):
                self.metrics[name + '_summary'] = summary(self.metrics.pop(name))
            self.a.metrics.parent.mkdir(parents=True, exist_ok=True)
            self.a.metrics.write_text(json.dumps(self.metrics, indent=2) + '\n')
            print(f'Metrics: {self.a.metrics}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True)
    ap.add_argument('--mode', choices=['chip', 'baseline'], default='chip')
    ap.add_argument('--container', default='singalong_sdk')
    ap.add_argument('--period', type=float, default=.5)
    ap.add_argument('--playout-delay', type=float, default=1.0, help='Seconds after frame-end audio timestamp')
    ap.add_argument('--duration', type=float)
    ap.add_argument('--sink', default='print')
    ap.add_argument('--selector', choices=['key', 'song', 'score'], default='score')
    ap.add_argument('--bpm', type=float, default=80)
    ap.add_argument('--verify-reference', action='store_true')
    ap.add_argument('--worker-debug', action='store_true')
    ap.add_argument('--metrics', type=Path, default=ROOT / 'data/live_chip/latest.json')
    a = ap.parse_args()
    if not .5 <= a.period <= 1 or a.playout_delay < a.period + .4:
        ap.error('Use period 0.5..1 s and playout-delay >= period + 0.4 s')
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    LiveRun(a).run()


if __name__ == '__main__':
    main()
