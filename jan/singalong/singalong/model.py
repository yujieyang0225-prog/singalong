"""Talamo pipeline for the sing-along SNN, built like Innatera's wristband example (the chip path).

Pipeline([c1.IFEncoder(48), Snn(GRSnn2LayersRNN 48 -> 64 recurrent -> 12)])
  - GRSnn2LayersRNN and the quantizers come from Innatera's wristband package
    (gesture_recognition_imu, mounted from the starter repo; not copied, proprietary).
  - The Snn returns spikes [T+1, B, 12]: time first, plus 1 frame of layer delay (as on the chip).
    output_activation() turns that into [B, T, 12] for the loss.
  - Readout = causal spike rate over the last RATE_WIN frames (D08), done in the loss / evaluation,
    i.e. on the RISC-V later, not in the SNN.
"""
import numpy as np
import torch
import torch.nn.functional as F

import talamo

RATE_WIN = 8          # frames (128 ms)


def build_pipeline(hidden=64, recurrent=True, enc_threshold=120.0, out_neurons=12):
    """out_neurons: 12 = chord head only (runs 0-2), 24 = 12 chord + 12 melody pitch-class outputs (D07 dual head)."""
    from gesture_recognition_imu.models import GRSnn2LayersRNN, GRSnn2Layers

    net_cls = GRSnn2LayersRNN if recurrent else GRSnn2Layers
    net = net_cls(in_neurons=48, hidden_neurons=hidden, out_neurons=out_neurons).set_timestep(1e-6)
    snn_step = talamo.snn.containers.Snn(net)
    enc = talamo.encoders.c1.IFEncoder(48, threshold_initializer=talamo.initializers.Constant(enc_threshold))
    return talamo.pipeline.elements.Pipeline([enc, snn_step]), snn_step


def output_activation(out):
    """Snn spikes [T+delay, B, 12] -> [B, T, 12] (drop the layer-delay step at the start)."""
    return out.permute(1, 0, 2)[:, 1:]


def spike_rate(spikes, win=RATE_WIN):
    """Causal moving average of spikes over the last `win` frames: [B, T, C] -> [B, T, C] in 0..1."""
    pad = F.pad(spikes.transpose(1, 2), (win - 1, 0))
    return F.avg_pool1d(pad, win, stride=1).transpose(1, 2)


class RateBCE(torch.nn.MSELoss):
    """BCE between the chroma spike rate and the chord pitch classes, on scored frames only.

    Subclasses MSELoss only so that Talamo's trainer treats the task as 'regression' (multi-label);
    the maths is BCE. Labels [B, T, 13 or 14]: 12 targets + 1 mask channel (+ melody class, unused here).

    Fixed 2026-10-07: the old `rate.clamp(1e-3, 1 - 1e-3)` has zero gradient for a silent output neuron
    (rate 0 is outside the clamp range), so neurons that never fired could never learn to fire. That is
    consistent with runs 0-2 'hardly firing'. The affine squeeze below keeps the gradient.
    """

    def forward(self, spikes, y):
        rate = spike_rate(spikes) * (1 - 2e-3) + 1e-3
        m = y[..., 12] > 0
        return F.binary_cross_entropy(rate[m], y[..., :12][m])


class DualRateLoss(torch.nn.MSELoss):
    """Chord head (outputs 0-11): BCE as above. Melody head (outputs 12-23): cross-entropy over the 12 melody
    rates scaled by alpha, plus a 13th 'no note' logit that is high when all melody outputs are quiet.
    Labels [B, T, 14] from singalong.data (channel 12 = mask, 13 = melody class 0..12). Same maths as
    singalong.torchsnn.DualLoss, so Talamo- and torch-trained nets are comparable."""

    def __init__(self, alpha=6.0, w_chord=1.0, w_pitch=1.0):
        super().__init__()
        self.alpha, self.w_chord, self.w_pitch = alpha, w_chord, w_pitch

    def forward(self, spikes, y):
        rate = spike_rate(spikes)
        m = y[..., 12] > 0
        loss = 0.0
        if self.w_chord:
            r = rate[..., :12][m] * (1 - 2e-3) + 1e-3
            loss = loss + self.w_chord * F.binary_cross_entropy(r, y[..., :12][m])
        if self.w_pitch and spikes.shape[-1] >= 24 and y.shape[-1] >= 14:
            mel = rate[..., 12:24][m]
            logits = torch.cat([self.alpha * mel, self.alpha * (0.5 - mel.max(-1, keepdim=True).values)], -1)
            loss = loss + self.w_pitch * F.cross_entropy(logits, y[..., 13][m].long())
        return loss


def fit(pipe, snn_step, X, Y, epochs, lr=1e-3, lr_enc=1e-2, batch_size=32, quantizers=None, loss_function=None):
    """pipe.fit exactly as in the wristband example, with per-frame labels.

    Note on learning rates: Talamo's parameters are in chip units (int8 weights, thresholds ~127), so lr 1e-3
    moves a weight of size 30 by 0.001 per step. The torch trainer (train_torch.py) needed lr ≈ 0.3 to learn
    at all; try --lr 0.1..0.5 here too (see 2026-10-07 Chord SNN training, 'what we found')."""
    q = lambda pat: pipe.query_torch_params(pat, talamo.utils.ParameterType.PARAMETER, references=False)  # noqa
    opt = torch.optim.AdamW([{"params": q("*encoder*threshold*"), "lr": lr_enc},
                             {"params": q("*snn_step.*synapses.weight"), "lr": lr},
                             {"params": q("*snn_step.*bias*"), "lr": lr}])
    from .data import dense_dataset
    rec = talamo.utils.recording.Recorder(model_performance_metrics=talamo.utils.recording.ModelMetrics(None, {}))
    pipe.fit(dataset=dense_dataset(X, Y), epochs=epochs, dataloader_args={"batch_size": batch_size, "shuffle": True},
             output_step=snn_step, output_activation=output_activation, optimizer=opt,
             loss_function=loss_function or RateBCE(), recorder=rec, quantizers=quantizers, verbose=0)


def make_quantizers(pipe, n_layers=2):
    """int8 weights + thresholds + bias per layer, encoder thresholds: as in the wristband notebook."""
    from gesture_recognition_imu.quantization import IfEncoderQuantizer, LayerScaleRoundAndClamp
    qs = [IfEncoderQuantizer(all_params=pipe.query_torch_params("*encoder*", references=True))]
    for i in range(n_layers):
        params = sum((pipe.query_torch_params(f"*snn_step.*{i}*{p}*", references=True)
                      for p in ("synapses.weight", "recurrent_synapses.weight", "threshold", "bias")), [])
        # de-duplicate ('synapses.weight' also matches 'recurrent_synapses.weight')
        seen, uniq = set(), []
        for p in params:
            if p.name not in seen:
                seen.add(p.name); uniq.append(p)
        qs.append(LayerScaleRoundAndClamp(all_params=uniq))
    return qs


@torch.no_grad()
def predict(pipe, X, batch=64, n_label_ch=14):
    """Run the pipeline (software, same path as the chip) -> output spike rates [N, T, n_out]."""
    from .data import dense_dataset
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        s = pipe(dense_dataset(xb, torch.zeros(len(xb), xb.shape[1], n_label_ch))).data_as_numpy()   # [B, T+1, n_out]
        out.append(spike_rate(torch.tensor(s[:, 1:], dtype=torch.float32)))
    return torch.cat(out)


def scores(rate, Y, thr=0.5):
    """Chroma F1 and exact-chord match of the chord head on scored frames; plus melody accuracy if present."""
    from .metrics import pitch_scores, scores as _scores
    out = _scores(rate[..., :12], Y, thr)
    if rate.shape[-1] >= 24:
        out.update(pitch_scores(rate[..., 12:24], Y))
    return out


def param_ranges(pipe):
    """min / max / all-integer of the parameters that go to the chip (check int8 ranges after quantization)."""
    out = {}
    for r in pipe.query_torch_params("*", references=True):
        if any(k in r.name for k in ("weight", "bias_current", "firing_threshold", "threshold")) and \
                "unit_synaptic" not in r.name:
            t = r.get_value().detach().float()
            if t.numel():
                out[r.name] = dict(min=round(float(t.min()), 3), max=round(float(t.max()), 3),
                                   integer=bool(torch.all(t == t.round())))
    return out
