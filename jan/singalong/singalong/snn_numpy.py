"""Integer reference implementation of the C1 network, frame by frame, from weights_int8.npz.

Third implementation next to Talamo's PyTorch neuron and the SoC simulator (which disagreed in 26 % of
spike slots, see 2026-10-07 Chord SNN training). Use it to (a) run the exported network on a laptop without
the SDK (live demo fallback, `live.py --evidence snn:<npz>`), (b) pin down where Talamo and the simulator
differ by comparing all three on the same input (Pulsar/IF_DNR/compare_torch_talamo.py).

All arithmetic is integer (numpy int64), one step per 16 ms frame, state kept across frames.
Options mirror the open questions in the vault (Pulsar SNN fabric (DNR) limits):
    enc_reset = "subtract" | "zero"      how the IF encoder resets after a spike
    layer_delay = 0 | 1                  does layer l+1 see layer l's spikes in the same step (0) or the next (1)
"""
from __future__ import annotations

import numpy as np


class C1Net:
    def __init__(self, npz_path, enc_reset="subtract", layer_delay=0):
        d = np.load(npz_path, allow_pickle=False)
        self.enc_thr = d["enc_threshold"].astype(np.int64)
        n_in = d["W_in"].shape[1]
        # C1 IF encoder (talamo/encoders/c1/if_encoder.py): weighted = floor(x * 2**w) * sign + bias; older exports
        # without these keys behave as w = 0, sign = 1, bias = 0 (which under-drives the encoder by 2**w)
        self.enc_w = d["enc_weight"].astype(np.int64) if "enc_weight" in d.files else np.zeros(n_in, np.int64)
        self.enc_sign = d["enc_weight_sign"].astype(np.int64) if "enc_weight_sign" in d.files else np.ones(n_in, np.int64)
        self.enc_bias = d["enc_bias"].astype(np.int64) if "enc_bias" in d.files else np.zeros(n_in, np.int64)
        self.W_in, self.b_h, self.thr_h = d["W_in"].astype(np.int64), d["b_h"].astype(np.int64), int(d["thr_h"].ravel()[0])
        self.W_rec = d["W_rec"].astype(np.int64) if "W_rec" in d.files else None
        self.W_out, self.b_o, self.thr_o = d["W_out"].astype(np.int64), d["b_o"].astype(np.int64), int(d["thr_o"].ravel()[0])
        self.enc_reset, self.layer_delay = enc_reset, layer_delay
        self.n_in, self.n_h, self.n_out = self.W_in.shape[1], self.W_in.shape[0], self.W_out.shape[0]
        self.reset()

    def reset(self):
        self.acc = np.zeros(self.n_in, np.int64)
        self.U_h = np.zeros(self.n_h, np.int64)
        self.U_o = np.zeros(self.n_out, np.int64)
        self.s_h_prev = np.zeros(self.n_h, np.int64)
        self.s_h_delayed = np.zeros(self.n_h, np.int64)

    def encode(self, x):
        w = np.floor(np.asarray(x, np.float64) * 2.0 ** self.enc_w).astype(np.int64) * self.enc_sign + self.enc_bias
        self.acc += w
        if self.enc_reset == "subtract":                 # T1-style variant, kept for comparison
            s = (self.acc >= self.enc_thr).astype(np.int64)
            self.acc -= s * self.enc_thr
        else:                                             # "zero" = C1 (Talamo): spike if acc > thr, then acc = 0
            s = (self.acc > self.enc_thr).astype(np.int64)
            self.acc *= (1 - s)
        return s

    @staticmethod
    def _neuron(U, I, thr):
        # C1 digital neuron (talamo/snn/neuron/c1/digital_neuron.py): U += W·s + bias, clamp to [0, 2^16 - 1]
        # (v_floor "zero"), spike if U > threshold (strict), reset to 0
        U = np.clip(U + I, 0, 2 ** 16 - 1)
        s = (U > thr).astype(np.int64)
        return U * (1 - s), s

    def step(self, x, return_all=False):
        """One frame: x [48] ints 0..100 -> output spikes [n_out] (and hidden, input spikes)."""
        s_in = self.encode(x)
        I_h = self.W_in @ s_in + self.b_h
        if self.W_rec is not None:
            I_h = I_h + self.W_rec @ self.s_h_prev
        self.U_h, s_h = self._neuron(self.U_h, I_h, self.thr_h)
        self.s_h_prev = s_h
        feed = self.s_h_delayed if self.layer_delay else s_h
        self.U_o, s_o = self._neuron(self.U_o, self.W_out @ feed + self.b_o, self.thr_o)
        self.s_h_delayed = s_h
        return (s_o, s_h, s_in) if return_all else s_o

    def run(self, X, reset=True):
        """Whole recording [T,48] -> output spikes [T,n_out]."""
        if reset:
            self.reset()
        return np.stack([self.step(x) for x in X])


class RateReadout:
    """Causal moving average over the last `win` frames (what the RISC-V computes from the spike counters)."""

    def __init__(self, n, win=8):
        self.buf = np.zeros((win, n))
        self.i = 0

    def __call__(self, spikes):
        self.buf[self.i % len(self.buf)] = spikes
        self.i += 1
        return self.buf[:min(self.i, len(self.buf))].mean(0)
