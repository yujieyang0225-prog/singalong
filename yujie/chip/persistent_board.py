"""Repository-local Talamo 4.7.1 compatibility adapter; physical C1, RAM boot only.

Private APIs are confined here. No rebuilds, retries, or boots during infer().
The SDK's per-sample protocol resets the network, not the application.
"""
from contextlib import ExitStack
from pathlib import Path
import threading
import time
import warnings

import numpy as np
import torch
import talamo

import build_chip
from singalong import data


class PersistentBoard:
    def __init__(self, run, npz):
        if talamo.__version__ != "4.7.1":
            raise RuntimeError("PersistentBoard requires requalification for this Talamo version")
        self.app = Path(run).resolve() / "app_hostfed_125"
        if not (self.app / "bin/c1/release/app.c1boot").is_file():
            raise FileNotFoundError(self.app)
        self.pipe, _, self.weights, cfg = build_chip.load(run, npz)
        if (cfg['n_in'], cfg['hidden'], cfg['out'], cfg['recurrent']) != (48, 80, 24, True):
            raise ValueError("Only the verified 48/80 recurrent/24 model is supported")
        self.soc = talamo.device.c1.Soc(simulate=False, clock_frequency=build_chip.CLOCK_MHZ)
        self.pipe.to(self.soc)
        self.lock = threading.Lock()
        self.stack = None
        self.calls = 0
        self.failed = False

    @staticmethod
    def dataset(x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (125, 48) or not np.isfinite(x).all():
            raise ValueError("Expected finite [125,48] features")
        if (x < 0).any() or (x > 100).any() or not np.array_equal(x, np.round(x)):
            raise ValueError("Expected existing frontend's integer-valued features in [0,100]")
        return data.dense_dataset(torch.tensor(x[None]), torch.zeros(1, 125, 14))

    def __enter__(self):
        if self.stack is not None:
            raise RuntimeError("Session already open")
        example = self.dataset(np.zeros((125, 48), np.float32))
        self.pipe._initialize_pipeline_step_names()
        self.assembled = self.pipe._assemble_embedded_pipeline(example)
        stack = ExitStack()
        try:
            t = time.perf_counter()
            self.connection = stack.enter_context(self.soc.get_board_connection())
            self.connection.boot_application(str(self.app))
            self.boot_seconds = time.perf_counter() - t
            stack.enter_context(self.assembled._configure_embedded_application_board_connection(self.connection))
        except BaseException:
            stack.close()
            raise
        self.stack = stack
        return self

    def infer(self, x):
        with self.lock:
            if self.stack is None or self.failed:
                raise RuntimeError("Board session closed or failed; explicitly restart the application")
            t = time.perf_counter()
            try:
                # Materialize while the same connection context is still active.
                with warnings.catch_warnings():
                    # The container globally suppresses warnings. Surface protocol
                    # timeouts/reset failures instead of hiding their cause.
                    warnings.filterwarnings('error', category=UserWarning,
                                            module=r'talamo\.pipeline\.embedded\.utils\.board_comm')
                    result = self.assembled._execute_pipeline(self.dataset(x), verbose=0)
                if len(result) != 1:
                    raise RuntimeError('Board returned no complete sample; startup/transport failed. '
                                       'Stop and inspect the connection; no automatic reboot is performed.')
                out = np.asarray(result.data_as_numpy())
                # The compiled HostOutput sync_count is 126: 125 input frames
                # plus one pipeline flush frame. Match board_check.py's [:125].
                if out.shape != (1, 126, 24):
                    raise RuntimeError(f"Incomplete board output: {out.shape}, expected (1,126,24)")
                if not np.isin(out, [0, 1]).all():
                    raise RuntimeError("Non-binary spike output")
            except BaseException:
                self.failed = True
                raise
            self.calls += 1
            return out[0, :125].astype(np.int8), time.perf_counter() - t

    def __exit__(self, *exc):
        if self.stack is not None:
            self.stack.__exit__(*exc)
            self.stack = None
