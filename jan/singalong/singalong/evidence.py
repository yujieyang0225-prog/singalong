"""Evidence sources for the selector: per frame, (melody_evidence[12], chord_evidence[12] or None).

  BaselineEvidence  harmonic-sum pitch-class salience straight from the 48 bands (D01 option D, no SNN).
                    The demo works with this today; it is also the "is the SNN worth it?" comparison.
  NumpySnnEvidence  the exported int8 network run by singalong.snn_numpy on the laptop (no SDK needed).
  TorchSnnEvidence  a train_torch.py checkpoint (float or QAT) run in PyTorch.
  BoardEvidence     spike counts streamed from the Pulsar EVK (JSON lines over the serial/USB link).
                    Line format: {"t": frame, "rates": [..24 values 0..1..]} or {"counts": [...], "win": 8}.
                    The board side is Pulsar/IF_DNR/deploy_live.py (untested here, see README).
"""
from __future__ import annotations

import json

import numpy as np

from . import frontend as fe

# harmonic weights for pitch-class salience: fundamental, octave, fifth (3rd harmonic), 2 octaves, major third (5th)
HARMONICS = [(0, 1.0), (12, 0.6), (19, 0.4), (24, 0.3), (28, 0.2)]


class BaselineEvidence:
    """bands [48] (0..100) -> melody salience [12]. Sub-harmonic summation: a note at MIDI m puts energy at
    m, m+12, m+19, m+24, m+28; sum those bands with falling weights, then fold to pitch class."""

    def __init__(self, smooth_frames=3):
        self.hist = []
        self.k = smooth_frames

    def __call__(self, bands, voiced=True):
        x = np.asarray(bands, float) / 100.0
        sal = np.zeros(fe.NB)
        for k in range(fe.NB):
            s, wsum = 0.0, 0.0
            for h, w in HARMONICS:
                if k + h < fe.NB:
                    s += w * x[k + h]; wsum += w
            sal[k] = s / wsum
        # a band is a *fundamental* only if the band an octave below is weaker (suppress harmonics of lower notes)
        for k in range(12, fe.NB):
            if x[k - 12] > 0.6 * x[k]:
                sal[k] *= 0.5
        chroma = np.zeros(12)
        for k in range(fe.NB):
            chroma[(fe.LO + k) % 12] = max(chroma[(fe.LO + k) % 12], sal[k])
        if not voiced or chroma.max() < 0.05:
            chroma[:] = 0
        else:
            chroma = np.clip(chroma - 0.5 * np.median(chroma), 0, None)   # sharpen: remove the floor
            chroma = chroma ** 2
            chroma /= max(chroma.max(), 1e-9)
        self.hist.append(chroma)
        self.hist = self.hist[-self.k:]
        return np.mean(self.hist, 0), None


class NumpySnnEvidence:
    def __init__(self, npz_path, win=8, **kw):
        from .snn_numpy import C1Net, RateReadout
        self.net = C1Net(npz_path, **kw)
        self.rate = RateReadout(self.net.n_out, win)

    def __call__(self, bands, voiced=True):
        r = self.rate(self.net.step(np.round(bands).astype(np.int64)))
        return _split(r)


class TorchSnnEvidence:
    def __init__(self, checkpoint, win=8, device="cpu"):
        import torch
        from .torchsnn import SingalongSNN
        ck = torch.load(checkpoint, map_location=device)
        self.net = SingalongSNN(**ck["cfg"]).to(device)
        self.net.load_state_dict(ck["model"])
        self.net.set_qat(True)
        self.net.eval()
        self.torch, self.device, self.win = torch, device, win
        self.buf = []
        self.state = None

    def __call__(self, bands, voiced=True):
        # run the whole history each call would be O(T); keep states manually instead
        x = self.torch.tensor(np.asarray(bands, np.float32))[None, None]
        with self.torch.no_grad():
            if self.state is None:
                self.acc = self.torch.zeros(1, x.shape[-1])
                self.state = (None, None)
            th = self.net.enc.thresholds()
            self.acc = self.acc + x[:, 0]
            s_in = (self.acc >= th).float()
            self.acc = self.acc - s_in * th
            s_h, st_h = self.net.h(s_in[:, None], self.state[0], return_state=True)
            s_o, st_o = self.net.o(s_h, self.state[1], return_state=True)
            self.state = (st_h, st_o)
        self.buf.append(s_o[0, 0].numpy())
        self.buf = self.buf[-self.win:]
        return _split(np.mean(self.buf, 0))


def _split(rate):
    rate = np.asarray(rate, float)
    if len(rate) >= 24:
        return rate[12:24], rate[:12]          # melody head, chord head
    return None, rate[:12]                     # chord-only network


class BoardEvidence:
    """Parse one JSON line from the board into (melody, chord) evidence; returns None if the line is not ours."""

    def __init__(self, win=8):
        self.win = win

    def parse(self, line):
        line = line.strip()
        if not line.startswith("{"):
            return None
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            return None
        if "rates" in d:
            return _split(np.asarray(d["rates"], float))
        if "counts" in d:
            return _split(np.asarray(d["counts"], float) / d.get("win", self.win))
        return None
