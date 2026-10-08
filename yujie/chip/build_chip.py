#!/usr/bin/env python3
"""Build the imported network as a host-fed chip application and report whether it fits.

The PC sends one window of band features [frames, 48]; the chip runs the IF encoder and the recurrent
network on its digital neurons and returns the output spikes of every frame. No microphone, no
front-end on the chip. Building needs no board. RAM boot only when it is run later; nothing is flashed.

    python3 ../../yujie/chip/build_chip.py ../../data/runs/A_vocalset_P_talamo \
        --npz ../../handoff/runs/A_vocalset_P/weights_int8.npz --check ../../handoff/testdata/damp250

Run in the SDK container from jan/singalong. The run folder is the output of import_torch_weights.py.
"""
import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "jan" / "singalong"), str(Path(__file__).parent)]
import talamo  # noqa: E402
import chipref  # noqa: E402
from singalong import data, model  # noqa: E402

CLOCK_MHZ = 100


def load(run, npz):
    """The pipeline with the imported parameters, verified against the chip-rule reference on random input."""
    w = np.load(npz, allow_pickle=False)
    cfg = ast.literal_eval(str(w["cfg"]))
    pipe, snn_step = model.build_pipeline(cfg["hidden"], recurrent=cfg["recurrent"], out_neurons=cfg["out"],
                                          enc_threshold=float(np.median(w["enc_threshold"])))
    pipe.trainable_params.load(str(Path(run) / "trainable_params_quantized.json"))
    return pipe, snn_step, w, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--npz", type=Path, required=True, help="the exported weights the run was imported from")
    ap.add_argument("--check", type=Path, required=True, help="dataset folder; one window of it is the example input")
    ap.add_argument("--frames", type=int, default=125, help="frames per window sent to the chip (125 = 2 s, the training crop)")
    ap.add_argument("--app", type=Path, default=None)
    a = ap.parse_args()
    run = a.run.resolve()
    app = (a.app or run / f"app_hostfed_{a.frames}").resolve()
    pipe, _, w, cfg = load(run, a.npz)

    rec = max(data.load_recordings("val", data=a.check), key=lambda r: len(r["X"]))
    X = rec["X"][600:600 + a.frames] if len(rec["X"]) >= 600 + a.frames else rec["X"][:a.frames]
    ds = data.dense_dataset(torch.tensor(X[None]), torch.zeros(1, len(X), 14))
    software = pipe(ds).data_as_numpy()[0][:len(X)]
    reference = chipref.run(w, X, **chipref.CHIP)
    assert np.array_equal(software, reference), "the loaded parameters do not reproduce the chip-rule reference; run the import first"
    result = dict(run=run.name, frames=a.frames, clock_mhz=CLOCK_MHZ, network=[cfg["n_in"], cfg["hidden"], cfg["out"]],
                  recurrent=cfg["recurrent"], software_equals_reference=True, example_output_spikes=int(software.sum()))

    if app.exists():
        sys.exit(f"{app} exists; choose another --app or remove it")
    scratch = run / "build_tmp"
    scratch.mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(scratch)                                      # builds leave files in the current directory
    try:
        pipe.to(talamo.device.c1.Soc(simulate=False, clock_frequency=CLOCK_MHZ))
        pipe.generate_embedded_application(app_path=str(app), dataset=ds)
        result["generated"] = True
        pipe.build_embedded_application(app_path=str(app))
        result["built"] = True
    except Exception as error:                                # report the real error
        result["error"] = f"{type(error).__name__}: {str(error)[:1500]}"
    finally:
        os.chdir(cwd)

    config = (app / "appconfig_generated.h").read_text() if (app / "appconfig_generated.h").exists() else ""
    result["generated_config"] = {k: int(v) for k, v in re.findall(r"#define (PIPELINE_CONFIG_\w+) \(?(-?\d+)\)?", config)}
    source = (app / "pipeline.c").read_text() if (app / "pipeline.c").exists() else ""
    result["kernels"] = sorted(set(re.findall(r'#include "kernel_(\w+)\.h"', source)))
    result["snn_kernel_params"] = dict(re.findall(r"\.(fire_count|auto_reset|num_frames|num_output_bitmaps) = (\w+)", source))
    log = (app / "app_build_log.log").read_text(errors="replace") if (app / "app_build_log.log").exists() else ""
    memory = re.search(r"ram:\s+(\d+) B\s+(\d+) KB\s+([\d.]+)%", log)
    if memory:
        result["ram_bytes"], result["ram_region_kb"], result["ram_percent"] = int(memory[1]), int(memory[2]), float(memory[3])
    boot = app / "bin/c1/release/app.c1boot"
    result["boot_image_bytes"] = boot.stat().st_size if boot.exists() else None
    (run / f"build_chip_{a.frames}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
