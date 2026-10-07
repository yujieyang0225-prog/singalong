"""The C1 network in plain PyTorch: trainable on any GPU without the Innatera SDK, exportable to the chip.

Equations (Pulsar SNN fabric (DNR) limits, Talamo SDK notes in the vault):
    IF encoder (per input line c):  A_c += x_c ;  spike if A_c >= theta_c ;  A_c -= theta_c      (level code, D03 A)
    digital neuron:                 U += W s_in + W_rec s_prev + b ;  U = max(U, 0) (v_floor 'zero')
                                    spike if U >= thr ;  U = 0 after a spike (reset to zero, no leak except bias)
    one SNN step per 16 ms frame; the recurrent input is the layer's own spikes from the previous step.

Chip units from the start: weights/bias in int8 range, thresholds integers (wristband example: 127 per layer),
encoder thresholds integers. Phase 1 trains in float; phase 2 (`qat=True`) rounds with a straight-through
estimator, so the exported integers behave like the trained net.

What is *not* modelled (check with Pulsar/IF_DNR/compare_torch_talamo.py before trusting numbers):
  - the 1-step delay per layer Talamo shows in its output ([T+1]); here layer l+1 sees layer l within the step
  - refractory period (0 here), encoder weight shifts (2^w, 1 here), int24 encoder accumulator wrap

Export: SingalongSNN.export_npz(path) -> the format singalong.snn_numpy reads and
Pulsar/IF_DNR/import_torch_weights.py loads into a Talamo pipeline.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_NEURONS, FAN_IN, FAN_IN_PAIRED = 384, 128, 256


def fits_c1(n_in, hidden, out, recurrent=True):
    """Does the network fit the DNR? Returns (ok, message)."""
    fan_h = n_in + (hidden if recurrent else 0)
    neurons = hidden + out
    notes = []
    if neurons > N_NEURONS:
        return False, f"{neurons} neurons > {N_NEURONS}"
    if fan_h > FAN_IN_PAIRED or out > FAN_IN or hidden > FAN_IN_PAIRED:
        return False, f"fan-in {fan_h} (hidden) / {hidden} (out) > {FAN_IN_PAIRED}"
    if fan_h > FAN_IN or hidden > FAN_IN:
        notes.append(f"fan-in {max(fan_h, hidden)} > {FAN_IN}: needs plane pairing (halves usable neurons to 192)")
        if neurons > N_NEURONS // 2:
            return False, f"{neurons} neurons > 192 with plane pairing"
    if out > 64:
        notes.append(f"{out} outputs > 64 readable spike counters")
    return True, "; ".join(notes) or f"{neurons} neurons, fan-in {fan_h}"


class SpikeFn(torch.autograd.Function):
    """Heaviside forward, triangular surrogate on the normalised distance to threshold."""
    @staticmethod
    def forward(ctx, v, width):
        ctx.save_for_backward(v)
        ctx.width = width
        return (v >= 0).to(v.dtype)

    @staticmethod
    def backward(ctx, g):
        v, = ctx.saved_tensors
        return g * torch.clamp(1 - v.abs() / ctx.width, min=0) / ctx.width, None


def spike(v, width=1.0):
    return SpikeFn.apply(v, width)


def ste_round(x):
    return x + (torch.round(x) - x).detach()


class IFEncoder(nn.Module):
    def __init__(self, n, threshold=120.0, lo=4.0, hi=20000.0):
        super().__init__()
        self.theta = nn.Parameter(torch.full((n,), float(threshold)))
        self.lo, self.hi = lo, hi
        self.qat = False

    def thresholds(self):
        th = self.theta.clamp(self.lo, self.hi)
        return ste_round(th) if self.qat else th

    def forward(self, x):
        """x [B,T,n] non-negative integers -> spikes [B,T,n]."""
        th = self.thresholds()
        acc = torch.zeros_like(x[:, 0])
        out = []
        for t in range(x.shape[1]):
            acc = acc + x[:, t]
            s = spike((acc - th) / th, 1.0)
            acc = acc - s * th
            out.append(s)
        return torch.stack(out, 1)


class C1Layer(nn.Module):
    def __init__(self, n_in, n, recurrent=False, threshold=127.0, w_scale=None, train_threshold=False):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n, n_in))
        self.W_rec = nn.Parameter(torch.empty(n, n)) if recurrent else None
        self.b = nn.Parameter(torch.zeros(n))
        self.thr = nn.Parameter(torch.full((1,), float(threshold)), requires_grad=train_threshold)
        self.reset_parameters(w_scale)
        self.qat = False

    def reset_parameters(self, w_scale=None):
        # input spike rates are ~0.3/frame: std of the summed input per step ≈ thr / 1.5 so neurons do fire
        s = w_scale or float(self.thr.item()) / (1.5 * np.sqrt(0.3 * max(self.W.shape[1], 1)))
        nn.init.normal_(self.W, 0, s)
        if self.W_rec is not None:
            nn.init.normal_(self.W_rec, 0, s / 2)
            with torch.no_grad():
                self.W_rec.fill_diagonal_(0)

    def q(self, p, lo=-128, hi=127):
        p = p.clamp(lo, hi)
        return ste_round(p) if self.qat else p

    def params_int(self):
        W = self.q(self.W)
        Wr = self.q(self.W_rec) if self.W_rec is not None else None
        return W, Wr, self.q(self.b), self.q(self.thr, 1, 65535)

    def forward(self, s_in, state=None, return_state=False):
        """s_in [B,T,n_in] spikes -> spikes [B,T,n]. state = (U, s_prev)."""
        W, Wr, b, thr = self.params_int()
        B, T, _ = s_in.shape
        U = torch.zeros(B, W.shape[0], device=s_in.device, dtype=s_in.dtype) if state is None else state[0]
        s_prev = torch.zeros_like(U) if state is None else state[1]
        I_in = s_in @ W.t() + b                        # [B,T,n], all steps at once
        out = []
        for t in range(T):
            U = U + I_in[:, t]
            if Wr is not None:
                U = U + s_prev @ Wr.t()
            U = U.clamp(min=0)                          # v_floor 'zero'
            s = spike((U - thr) / thr, 1.0)
            U = U * (1 - s)                             # reset to zero
            s_prev = s
            out.append(s)
        S = torch.stack(out, 1)
        return (S, (U, s_prev)) if return_state else S


class SingalongSNN(nn.Module):
    """IFEncoder(48) -> C1Layer(hidden, recurrent) -> C1Layer(out).  out = 12 chord + 12 melody by default."""

    def __init__(self, n_in=48, hidden=64, out=24, recurrent=True, enc_threshold=120.0, threshold=127.0):
        super().__init__()
        ok, msg = fits_c1(n_in, hidden, out, recurrent)
        if not ok:
            raise ValueError(f"does not fit the C1: {msg}")
        self.fit_note = msg
        self.enc = IFEncoder(n_in, enc_threshold)
        self.h = C1Layer(n_in, hidden, recurrent, threshold)
        self.o = C1Layer(hidden, out, False, threshold)
        self.cfg = dict(n_in=n_in, hidden=hidden, out=out, recurrent=recurrent, enc_threshold=enc_threshold,
                        threshold=threshold)

    def set_qat(self, on=True):
        for m in (self.enc, self.h, self.o):
            m.qat = on

    def forward(self, x, return_hidden=False):
        s_in = self.enc(x)
        s_h = self.h(s_in)
        s_o = self.o(s_h)
        return (s_o, s_h, s_in) if return_hidden else s_o

    @torch.no_grad()
    def export_arrays(self):
        """Rounded integer parameters as numpy, the chip's view of this network."""
        W, Wr, b, thr = [p.round().cpu().numpy() if p is not None else None for p in self.h.params_int()]
        Wo, _, bo, thro = [p.round().cpu().numpy() if p is not None else None for p in self.o.params_int()]
        out = dict(enc_threshold=self.enc.thresholds().round().cpu().numpy(), W_in=W, b_h=b, thr_h=thr,
                   W_out=Wo, b_o=bo, thr_o=thro, cfg=np.array(str(self.cfg)))
        if Wr is not None:
            out["W_rec"] = Wr
        return out

    def export_npz(self, path):
        np.savez(path, **self.export_arrays())


def spike_rate(spikes, win=8):
    """Causal moving average over the last `win` frames: [B,T,C] -> [B,T,C] in 0..1 (the RISC-V readout)."""
    pad = F.pad(spikes.transpose(1, 2), (win - 1, 0))
    return F.avg_pool1d(pad, win, stride=1).transpose(1, 2)


class DualLoss(nn.Module):
    """Chord head: BCE on rates (12 multi-hot). Melody head: CE over scaled rates (13th class 'no note' = all quiet).
    Plus a small activity term that keeps hidden neurons alive and sparse. Labels Y [B,T,14] (see data.py)."""

    def __init__(self, win=8, alpha=6.0, w_chord=1.0, w_pitch=1.0, w_act=0.05, target_rate=0.15):
        super().__init__()
        self.win, self.alpha = win, alpha
        self.w_chord, self.w_pitch, self.w_act, self.target_rate = w_chord, w_pitch, w_act, target_rate

    def forward(self, s_out, Y, s_hidden=None):
        from .data import MASK_CH, MEL_CH
        rate = spike_rate(s_out, self.win)
        m = Y[..., MASK_CH] > 0
        parts = {}
        n_chord = min(12, s_out.shape[-1])
        mc = m & (Y[..., :12].sum(-1) > 0)                     # all-zero chord row = chord unknown (pseudo-labelled real audio)
        if self.w_chord and n_chord and mc.any():
            r = rate[..., :12][mc] * (1 - 2e-3) + 1e-3         # affine, not clamp: silent neurons keep a gradient
            parts["chord"] = self.w_chord * F.binary_cross_entropy(r, Y[..., :12][mc])
        if self.w_pitch and s_out.shape[-1] >= 24:
            logits = self.alpha * rate[..., 12:24][m]
            # 13th logit for 'no note': high when all 12 melody outputs are quiet
            quiet = self.alpha * (0.5 - rate[..., 12:24][m].max(-1, keepdim=True).values)
            parts["pitch"] = self.w_pitch * F.cross_entropy(torch.cat([logits, quiet], -1), Y[..., MEL_CH][m].long())
        if self.w_act and s_hidden is not None:
            hr = s_hidden.mean((0, 1))                                # per-neuron firing rate
            parts["act"] = self.w_act * ((hr - self.target_rate) ** 2).mean() * 10
        loss = sum(parts.values())
        return loss, {k: float(v.detach()) for k, v in parts.items()}
