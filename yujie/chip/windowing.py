"""Pure sliding-window/rate adapter, independent of Talamo and audio devices."""
import math
import numpy as np

FRAMES, BURN, RATE_WIN, FPS = 125, 31, 8, 62.5


def window_end(update, period=0.5):
    """Exclusive frame end. Fractional hop count gives 31/32-frame steps at 500 ms."""
    return FRAMES + math.ceil(update * period * FPS - 1e-9)


def newest_rates(spikes, end, last_frame, period=0.5):
    """Return absolute indices and within-window causal rates, with no overlap replay.

    Startup delivers just the newest update's worth of frames. Older frames warm
    the SNN and readout. Later windows continue exactly after last_frame.
    """
    spikes = np.asarray(spikes)
    if spikes.shape != (FRAMES, 24) or not np.isin(spikes, [0, 1]).all():
        raise ValueError('Expected binary [125,24] spike raster')
    start = end - FRAMES
    first = end - math.ceil(period * FPS) if last_frame is None else last_frame + 1
    if first < start + BURN:
        raise RuntimeError('Output gap exceeds valid post-burn-in window; stop and restart')
    if first >= end:
        raise RuntimeError('Duplicate or out-of-order window')
    prefix = np.concatenate([np.zeros((1, 24)), np.cumsum(spikes, axis=0)], axis=0)
    indices = np.arange(first, end)
    local = indices - start
    rates = (prefix[local + 1] - prefix[np.maximum(0, local + 1 - RATE_WIN)]) / RATE_WIN
    return indices, rates
