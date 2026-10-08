#!/usr/bin/env python3
"""Why do Talamo and the NumPy reference disagree on torch-trained weights? Stage-by-stage comparison.

Run in the SDK container (Talamo 4.7.1) from jan/singalong:

    python3 ../../yujie/chip/diagnose_import.py ../../handoff/runs/A_vocalset_P/weights_int8.npz ../../handoff/testdata/damp250

It writes the npz values into the Talamo pipeline exactly as import_torch_weights.py does, then compares
  1. the IF encoder alone against both reset rules of the reference,
  2. the whole network called step by step (which uses the values just written) against pipe(...)
     (which is what import_torch_weights.py --check, sim_check.py and deploy_live.py use),
  3. the output spikes against the reference for both orientations of the recurrent matrix.
Agreement is scored on spikes (F1 of spike positions). Counting equal entries is misleading here: about
95 % of the entries are silent in both, so two unrelated spike trains already "agree" in over 0.9.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jan" / "singalong"))
import talamo  # noqa: E402
from singalong import data, model  # noqa: E402
from singalong.snn_numpy import C1Net  # noqa: E402


def f1(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return round(float(2 * (a & b).sum() / max(1, a.sum() + b.sum())), 3)


def main(npz, dataset, first=600, frames=600):
    w = np.load(npz, allow_pickle=False)
    X = data.load_recordings("val", data=dataset)[0]["X"][first:first + frames]
    pipe, _ = model.build_pipeline(int(w["W_in"].shape[0]), recurrent="W_rec" in w.files, out_neurons=int(w["W_out"].shape[0]),
                                   enc_threshold=float(np.median(w["enc_threshold"])))
    pipe.quantize(quantizers=model.make_quantizers(pipe))
    refs = {r.name: r for r in pipe.query_torch_params("*", references=True)}

    def put(name, value):
        current = refs[name].get_value().detach()
        value = torch.tensor(np.asarray(value), dtype=current.dtype)
        if value.shape != current.shape:
            value = value.T if value.T.shape == current.shape else torch.full_like(current, float(value.reshape(-1)[0]))
        with torch.no_grad():
            refs[name].get_value().copy_(value)

    def load(recurrent_transposed, encoder_exponent):
        put("c1_if_encoder_step.threshold", w["enc_threshold"])
        if encoder_exponent is not None:      # the encoder multiplies its input by 2 ** weight; quantize() leaves it at 7
            put("c1_if_encoder_step.weight", np.full(len(w["enc_threshold"]), encoder_exponent, np.float32))
        put("snn_step.layers.0.synapses.weight", w["W_in"])
        put("snn_step.layers.0.recurrent_synapses.weight", w["W_rec"].T if recurrent_transposed else w["W_rec"])
        put("snn_step.layers.0.neurons.bias_current", w["b_h"]); put("snn_step.layers.0.neurons.firing_threshold", w["thr_h"])
        put("snn_step.layers.1.synapses.weight", w["W_out"])
        put("snn_step.layers.1.neurons.bias_current", w["b_o"]); put("snn_step.layers.1.neurons.firing_threshold", w["thr_o"])

    sequence, encoder = list(pipe.filter_steps(talamo.pipeline.elements.PipelineStep))[:2]
    ds = data.dense_dataset(torch.tensor(X[None]), torch.zeros(1, len(X), 14))
    print(f"input {X.shape}, encoder exponent after quantize(): {float(refs['c1_if_encoder_step.weight'].get_value().float().mean()):g}")

    load(False, 0)
    spikes = np.asarray(encoder(ds).data_as_numpy())[0]
    print(f"1. encoder (exponent 0): Talamo {int(spikes.sum())} spikes")
    for rule in ("zero", "subtract"):
        net = C1Net(npz, enc_reset=rule); net.reset()
        ref = np.stack([net.encode(x) for x in X])
        print(f"     reference enc_reset={rule:8s}: {int(ref.sum())} spikes, F1 {f1(spikes[:len(ref)], ref)}")

    print("2. and 3. network output")
    for transposed in (False, True):
        load(transposed, 0)
        direct = np.asarray(sequence(ds).data_as_numpy())[0]
        through_pipe = np.asarray(pipe(ds).data_as_numpy())[0]
        print(f"   W_rec transposed={transposed}: steps called directly {int(direct.sum())} spikes, pipe() {int(through_pipe.sum())} spikes, "
              f"same: {np.array_equal(direct, through_pipe)}")
        for rule in ("zero", "subtract"):
            ref = C1Net(npz, layer_delay=1, enc_reset=rule).run(X)
            print(f"       reference layer_delay=1 enc_reset={rule:8s}: {int(ref.sum())} spikes, F1 {max(f1(direct[s:s + len(ref)], ref) for s in (0, 1))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
