#!/usr/bin/env python3
"""Port check of a trained run on Innatera's C1 simulator (no board needed).

1. Load the run, quantized (int8) exactly as for the chip.
2. Software:  the Talamo pipeline in PyTorch          -> spikes
3. Simulator: the Snn moved to Soc(simulate=True)      -> spikes   (in-process SNN architecture simulator,
   see skill talamo-socsim-and-power). Same input: one validation singer streamed as ONE long sample,
   so the network keeps its state across frames, as it must when running live.
4. Compare spike-for-spike, and score both against the MIDI chords.
5. Simulated dynamic energy per frame from a Recorder (SNN path). Only for COMPARING designs, never quote it
   as the chip's power (skill: talamo-socsim-and-power).
6. --build: also generate + build the embedded application (does the mapper accept the network?).

Run:  jan/singalong/run.sh python3 Pulsar/IF_DNR/sim_check.py data/runs/<run> [--secs 10] [--build]
Writes data/runs/<run>/sim_check.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import talamo  # noqa: E402
from singalong import data, model  # noqa: E402

CLOCK_MHZ = 100   # C1 Soc default; always state it (skill: talamo, Soc traps)


def load_quantized(run):
    m = json.load(open(run / "metrics.json"))["args"]
    pipe, snn_step = model.build_pipeline(m["hidden"], recurrent=not m.get("feedforward", False),
                                          enc_threshold=m.get("enc_threshold", 120.0))
    pipe.quantize(quantizers=model.make_quantizers(pipe))          # put the steps in quantized state
    qfile = run / "trainable_params_quantized.json"
    if qfile.exists():
        pipe.trainable_params.load(str(qfile))
    else:                                                           # float checkpoint only: quantize it now
        pipe, snn_step = model.build_pipeline(m["hidden"], recurrent=not m.get("feedforward", False),
                                              enc_threshold=m.get("enc_threshold", 120.0))
        pipe.trainable_params.load(str(run / "trainable_params.json"))
        pipe.quantize(quantizers=model.make_quantizers(pipe))
    return pipe, snn_step


def run_pipe(pipe, X, recorder=None):
    ds = data.dense_dataset(X, torch.zeros(len(X), X.shape[1], 13))
    out = pipe(ds, recorder=recorder, verbose=0).data_as_numpy()    # [N, T+delay, 12]
    return torch.tensor(out[:, -X.shape[1]:], dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--rec", type=int, default=0, help="validation singer index")
    ap.add_argument("--start", type=float, default=9.0)
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--build", action="store_true", help="also generate + build the embedded app")
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else data.REPO / a.run
    res = dict(run=run.name, clock_mhz=CLOCK_MHZ, secs=a.secs)

    rec = data.load_recordings("val")[a.rec]
    t0, n = int(a.start * data.FPS), int(a.secs * data.FPS)
    X = torch.tensor(rec["X"][None, t0:t0 + n])
    Y = torch.tensor(np.concatenate([rec["Y"][t0:t0 + n], rec["M"][t0:t0 + n, None]], 1)[None]).float()
    res["singer"] = rec["id"]

    # --- software (PyTorch), quantized
    pipe, snn_step = load_quantized(run)
    t = time.time()
    sw = run_pipe(pipe, X)
    res["software_s"] = round(time.time() - t, 1)

    # --- simulator: move the Snn wrapper (not the torch module) to a simulated C1
    soc = talamo.device.c1.Soc(simulate=True, clock_frequency=CLOCK_MHZ)
    snn_step.to(soc)
    rec_pw = talamo.utils.recording.Recorder(
        model_performance_metrics=talamo.utils.recording.ModelMetrics(None, {}), record_data=["power_consumption"])
    t = time.time()
    sim = run_pipe(pipe, X, recorder=rec_pw)
    res["simulator_s"] = round(time.time() - t, 1)

    # --- compare
    res["spikes_software"] = int(sw.sum())
    res["spikes_simulator"] = int(sim.sum())
    res["spike_match"] = round(float((sw == sim).float().mean()), 4)
    first_diff = torch.nonzero((sw != sim).any(-1)[0])
    res["first_differing_frame"] = int(first_diff[0]) if len(first_diff) else None
    # time offset between the two spike trains (e.g. recurrent spikes one step later on the chip)
    shifts = {}
    for d in range(-3, 4):
        a_, b_ = (sw[:, :sw.shape[1] - d], sim[:, d:]) if d >= 0 else (sw[:, -d:], sim[:, :sim.shape[1] + d])
        shifts[d] = round(float((a_ == b_).float().mean()), 4)
    res["spike_match_by_shift"] = shifts
    np.savez_compressed(run / "sim_check_spikes.npz", software=sw.numpy(), simulator=sim.numpy(), X=X.numpy())
    res["score_software"] = model.scores(model.spike_rate(sw), Y)
    res["score_simulator"] = model.scores(model.spike_rate(sim), Y)

    # --- energy (relative only)
    log = rec_pw.get_data_log()
    if "power_consumption" not in log:
        res["energy"] = f"no power_consumption in recorder log, keys: {sorted(log)}"
    else:
        res["energy_raw"] = log["power_consumption"]
        dev = log["power_consumption"].get("devices", {})
        if not dev:
            raise RuntimeError("power_consumption has no devices; do not quote a number")
        n_frames = X.shape[1]
        # dynamic energy only (leakage and wall time are harness artefacts, skill talamo-socsim-and-power)
        res["dynamic_energy_nj_per_frame_by_device"] = {
            k: round(v["total_dynamic_energy_nj"] / n_frames, 2) for k, v in dev.items()}
    print(json.dumps({k: v for k, v in res.items() if k != "energy_raw"}, indent=1))

    # --- optional: embedded application (mapper + build), no board needed
    if a.build:
        app = (run / "app").resolve()
        cwd = os.getcwd()
        os.makedirs(run / "build_tmp", exist_ok=True)
        os.chdir(run / "build_tmp")                       # builds litter the CWD (skill: talamo)
        try:
            pipe.to(talamo.device.c1.Soc(simulate=True, clock_frequency=CLOCK_MHZ))
            pipe.generate_embedded_application(app_path=str(app), dataset=data.dense_dataset(X, Y))
            pipe.build_embedded_application(app_path=str(app))
            res["build"] = "ok: " + ", ".join(str(p.relative_to(run)) for p in app.rglob("*.c1boot"))
        except Exception as e:                               # report the real error, do not mask it
            res["build"] = f"FAILED: {type(e).__name__}: {str(e)[:500]}"
        finally:
            os.chdir(cwd)
        print("build:", res["build"])

    json.dump(res, open(run / "sim_check.json", "w"), indent=1, default=str)
    print(f"-> {run / 'sim_check.json'}")


if __name__ == "__main__":
    main()
