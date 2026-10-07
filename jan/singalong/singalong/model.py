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


def build_pipeline(hidden=64, recurrent=True, enc_threshold=120.0):
    from gesture_recognition_imu.models import GRSnn2LayersRNN, GRSnn2Layers

    net_cls = GRSnn2LayersRNN if recurrent else GRSnn2Layers
    net = net_cls(in_neurons=48, hidden_neurons=hidden, out_neurons=12).set_timestep(1e-6)
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
    the maths is BCE. Labels [B, T, 13]: 12 targets + 1 mask channel.
    """

    def forward(self, spikes, y):
        rate = spike_rate(spikes).clamp(1e-3, 1 - 1e-3)
        m = y[..., 12] > 0
        return F.binary_cross_entropy(rate[m], y[..., :12][m])


def fit(pipe, snn_step, X, Y, epochs, lr=1e-3, lr_enc=1e-2, batch_size=32, quantizers=None):
    """pipe.fit exactly as in the wristband example, with per-frame labels."""
    q = lambda pat: pipe.query_torch_params(pat, talamo.utils.ParameterType.PARAMETER, references=False)  # noqa
    opt = torch.optim.AdamW([{"params": q("*encoder*threshold*"), "lr": lr_enc},
                             {"params": q("*snn_step.*synapses.weight"), "lr": lr},
                             {"params": q("*snn_step.*bias*"), "lr": lr}])
    from .data import dense_dataset
    rec = talamo.utils.recording.Recorder(model_performance_metrics=talamo.utils.recording.ModelMetrics(None, {}))
    pipe.fit(dataset=dense_dataset(X, Y), epochs=epochs, dataloader_args={"batch_size": batch_size, "shuffle": True},
             output_step=snn_step, output_activation=output_activation, optimizer=opt, loss_function=RateBCE(),
             recorder=rec, quantizers=quantizers, verbose=0)


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
def predict(pipe, X, batch=64):
    """Run the pipeline (software, same path as the chip) -> chroma spike rate [N, T, 12]."""
    from .data import dense_dataset
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        s = pipe(dense_dataset(xb, torch.zeros(len(xb), xb.shape[1], 13))).data_as_numpy()   # [B, T+1, 12]
        out.append(spike_rate(torch.tensor(s[:, 1:], dtype=torch.float32)))
    return torch.cat(out)


def scores(rate, Y, thr=0.5):
    """Chroma F1 and exact-chord match on scored frames. rate [N,T,12], Y [N,T,13]."""
    m = Y[..., 12] > 0
    p, y = (rate[m] >= thr).float(), Y[..., :12][m]
    tp, fp, fn = (p * y).sum(), (p * (1 - y)).sum(), ((1 - p) * y).sum()
    return dict(f1=round(float(2 * tp / (2 * tp + fp + fn + 1e-9)), 3),
                exact=round(float((p == y).all(1).float().mean()), 3))


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
