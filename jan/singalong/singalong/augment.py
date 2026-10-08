"""Augmentation on precomputed recordings (D11), always X and labels together, before cropping.

Works on the dicts from data.load_recordings():
    X_log [T,48] log2 band energy (not normalised), voiced [T], Y [T,12] chord pcs, MEL [T] melody class, M [T]

  band_shift(rec, k)   transpose by k semitones: roll the band rows, fill the rows that come in with noise
                       from the recording's own silent frames; roll chord pcs and melody class by k
  tempo(rec, f)        f > 1 faster: resample the time axis of everything (nearest frame)
  time_warp(rec, …)    smooth random tempo changes (rubato)
  noise(rec, db)       additive noise in the log domain + random gain (labels unchanged)
  band_dropout(rec, p) zero random bands in the normalised input (labels unchanged)
  random_augment(rec, rng, cfg) applies a random combination; returns a NEW dict with X renormalised.
"""
from __future__ import annotations

import numpy as np

from . import frontend as fe


def _finish(rec, X_log, voiced, Y, MEL, M):
    out = dict(rec)
    out.update(X_log=X_log, voiced=voiced, Y=Y, MEL=MEL, M=M, X=fe.normalise(X_log, voiced))
    return out


def band_shift(rec, k, rng=None):
    if k == 0:
        return rec
    rng = rng or np.random.default_rng()
    X = rec["X_log"]
    silent = X[~rec["voiced"]] if (~rec["voiced"]).sum() >= 10 else X
    mu, sd = silent.mean(0), silent.std(0) + 1e-3
    Xs = np.roll(X, k, axis=1)
    rows = range(0, k) if k > 0 else range(fe.NB + k, fe.NB)          # bands with no source after the shift
    for b in rows:
        Xs[:, b] = rng.normal(mu[b], sd[b], len(X))
    Y = np.roll(rec["Y"], k, axis=1)
    mel = rec["MEL"].copy()
    on = mel < 12
    mel[on] = (mel[on] + k) % 12
    return _finish(rec, Xs.astype(np.float32), rec["voiced"], Y, mel, rec["M"])


def _resample_idx(T, factor):
    n = max(int(round(T / factor)), 8)
    return np.clip(np.round(np.arange(n) * factor).astype(int), 0, T - 1)


def tempo(rec, factor):
    idx = _resample_idx(len(rec["X_log"]), factor)
    return _finish(rec, rec["X_log"][idx], rec["voiced"][idx], rec["Y"][idx], rec["MEL"][idx], rec["M"][idx])


def time_warp(rec, rng, max_dev=0.25, n_knots=6):
    """Piecewise-linear speed curve between knots in [1-max_dev, 1+max_dev]."""
    T = len(rec["X_log"])
    knots = rng.uniform(1 - max_dev, 1 + max_dev, n_knots)
    speed = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, n_knots), knots)
    pos = np.cumsum(speed)                                            # source position per output frame
    pos = pos[pos < T]
    idx = np.clip(np.round(pos).astype(int), 0, T - 1)
    return _finish(rec, rec["X_log"][idx], rec["voiced"][idx], rec["Y"][idx], rec["MEL"][idx], rec["M"][idx])


def noise(rec, rng, snr_db=None, gain_oct=None):
    """Noise added in the log2 domain, scaled so that silent frames rise towards the singing level at low SNR."""
    X = rec["X_log"]
    snr_db = rng.uniform(15, 40) if snr_db is None else snr_db
    peak = np.percentile(X[rec["voiced"]].max(1), 90) if rec["voiced"].any() else X.max()
    floor = peak - snr_db / 10 * np.log2(10)                          # dB -> octaves (log2 of power)
    nz = floor + rng.normal(0, 0.5, X.shape)
    Xn = np.logaddexp2(X, nz).astype(np.float32)                      # power addition in log2 domain
    Xn = Xn + (rng.uniform(-2, 2) if gain_oct is None else gain_oct)  # overall gain (removed by normalisation)
    return _finish(rec, Xn, rec["voiced"], rec["Y"], rec["MEL"], rec["M"])


def band_dropout(rec, rng, p=0.05):
    out = dict(rec)
    X = rec["X"].copy()
    drop = rng.random(X.shape[1]) < p
    X[:, drop] = 0
    out["X"] = X
    return out


DEFAULT = dict(shift=6, tempo=(0.75, 1.35), warp=0.2, noise=True, dropout=0.03)


def random_augment(rec, rng, cfg=DEFAULT):
    """Random transposition in [-shift, shift] (kept inside the band range for the sung notes), tempo, warp, noise."""
    out = rec if rec["X_log"].dtype == np.float32 else dict(rec, X_log=rec["X_log"].astype(np.float32))  # compact recs
    if cfg.get("shift"):
        k = int(rng.integers(-cfg["shift"], cfg["shift"] + 1))
        out = band_shift(out, k, rng)
    if cfg.get("tempo"):
        out = tempo(out, float(rng.uniform(*cfg["tempo"])))
    if cfg.get("warp"):
        out = time_warp(out, rng, cfg["warp"])
    if cfg.get("noise"):
        out = noise(out, rng)
    if cfg.get("dropout"):
        out = band_dropout(out, rng, cfg["dropout"])
    return out
