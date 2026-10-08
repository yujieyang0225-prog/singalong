"""Integer reference of the sing-along network with selectable semantics, batched over crops.

Two rule sets matter:

  TRAINED  what jan/singalong/singalong/torchsnn.py computes, so what the weights were optimised for
  CHIP     what Talamo 4.7.1's software model of the C1 computes; with these rules this file reproduces
           Talamo's output spike for spike (yujie/chip/diagnose_import.py)

They share: IF encoder with reset to zero and a strict '>' threshold, neurons with a floor at 0, a strict '>'
threshold and reset to zero. They differ in two places:

  recurrent   TRAINED: the hidden layer adds W_rec @ (its spikes of the previous step) together with the
              input and the bias, then floors once.
              CHIP: the hidden layer integrates input + bias, floors, fires and resets; then W_rec @ (the
              spikes it just fired) is added to the membrane and floored again. Same weights, but an
              inhibitory recurrent kick is cut off at 0 before the next input arrives.
  layer_delay TRAINED: the output layer sees the hidden spikes of the same step.
              CHIP: it sees them one step later, so every output is one frame (16 ms) late.

W_rec is indexed [to, from], as torchsnn exports it.
"""
import numpy as np

TRAINED = dict(recurrent="pre", layer_delay=0)
CHIP = dict(recurrent="post", layer_delay=1)
MEMBRANE_MAX = 65535


def run(npz, X, recurrent="pre", layer_delay=0, return_hidden=False, encoder_reset="zero"):
    """X [N, T, n_in] or [T, n_in], integers 0..100 -> output spikes of the same leading shape, [.., T, n_out]."""
    w = np.load(npz, allow_pickle=False) if not hasattr(npz, "files") else npz
    single = X.ndim == 2
    X = np.floor(X[None] if single else X).astype(np.int64)
    n, steps, _ = X.shape
    theta = w["enc_threshold"].astype(np.int64)
    w_in, w_out = w["W_in"].astype(np.int64).T, w["W_out"].astype(np.int64).T          # [from, to]
    w_rec = w["W_rec"].astype(np.int64).T if "W_rec" in w.files else None
    b_h, b_o = w["b_h"].astype(np.int64), w["b_o"].astype(np.int64)
    thr_h, thr_o = int(w["thr_h"].ravel()[0]), int(w["thr_o"].ravel()[0])
    acc = np.zeros((n, len(theta)), np.int64)
    u_h, u_o = np.zeros((n, len(b_h)), np.int64), np.zeros((n, len(b_o)), np.int64)
    previous = np.zeros_like(u_h)
    delayed = np.zeros_like(u_h)
    out, hidden = np.zeros((n, steps, len(b_o)), np.int8), np.zeros((n, steps, len(b_h)), np.int8)
    for t in range(steps):
        acc += X[:, t]
        if encoder_reset == "zero":                      # trainer, Talamo and chip
            s_in = (acc > theta).astype(np.int64)
            acc *= 1 - s_in
        else:                                            # "subtract": the default of singalong/snn_numpy.py, used by nothing else
            s_in = (acc >= theta).astype(np.int64)
            acc -= s_in * theta
        current = s_in @ w_in + b_h
        if w_rec is not None and recurrent == "pre":
            current = current + previous @ w_rec
        u_h = np.clip(u_h + current, 0, MEMBRANE_MAX)
        s_h = (u_h > thr_h).astype(np.int64)
        u_h *= 1 - s_h
        if w_rec is not None and recurrent == "post":
            u_h = np.clip(u_h + s_h @ w_rec, 0, MEMBRANE_MAX)
        previous = s_h
        u_o = np.clip(u_o + (delayed if layer_delay else s_h) @ w_out + b_o, 0, MEMBRANE_MAX)
        s_o = (u_o > thr_o).astype(np.int64)
        u_o *= 1 - s_o
        delayed = s_h
        out[:, t], hidden[:, t] = s_o, s_h
    out, hidden = (out[0], hidden[0]) if single else (out, hidden)
    return (out, hidden) if return_hidden else out
