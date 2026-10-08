#!/usr/bin/env python3
"""Build the host-fed chip application with a fixed text on the EVK display. Separate app folder.

Same network, parameters and host protocol as build_chip.py's app_hostfed_125. One extra
pass-through kernel in front of the encoder draws the text once after boot. No board needed
to build; RAM boot only when run. The validated app_hostfed_125 is not touched.

    python3 ../../yujie/chip/build_banner.py ../../data/runs/A_vocalset_P_talamo \\
        --npz ../../handoff/runs/A_vocalset_P/weights_int8.npz

Run in the SDK container from jan/singalong.
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import talamo
from talamo.pipeline.elements import Pipeline, PreprocessingStep
from talamo.pipeline.embedded import DenseDataConstraint, EmbeddedKernelInfo, EmbeddedKernelParam
from talamo.pipeline.embedded.data_constraint import ANY_SHAPE

import build_chip
import chipref
from singalong import data

TEXT = "SING-ALONG SNN\nNeuroTUM\n\nPitch network\nrunning on\nPulsar C1"
APP = "app_hostfed_125_banner"


class Banner(PreprocessingStep):
    """Hands every frame on unchanged; its chip kernel draws TEXT once."""

    def get_step_name(self):
        return "sa_banner"

    def __call__(self, dataset, **kwargs):
        return dataset

    def embedded_input_data_constraint(self):
        return DenseDataConstraint(shape=(ANY_SHAPE, 48), stream_chunk_shape=(ANY_SHAPE, 48), dtype=np.int32,
                                   name="Banner input")       # frame by frame, like the encoder

    def generate_embedded_info(self, trainable_params, input_type, output_constraint):
        text = np.frombuffer((TEXT + "\x00").encode("ascii"), dtype=np.uint8)
        return [EmbeddedKernelInfo(params=[EmbeddedKernelParam("count", 48, np.uint32),
                                           EmbeddedKernelParam("text", text, np.uint8)],
                                   input_type=input_type, output_type=input_type, kernel_type="sa_banner",
                                   app_defconfig=["CONFIG_DISPLAY_EAOLEDS1026=y\n"])]


def load(run, npz):
    """build_chip.load's pipeline with the Banner step in front; same parameters."""
    import ast
    from gesture_recognition_imu.models import GRSnn2LayersRNN
    w = np.load(npz, allow_pickle=False)
    cfg = ast.literal_eval(str(w["cfg"]))
    assert (cfg["n_in"], cfg["hidden"], cfg["out"], cfg["recurrent"]) == (48, 80, 24, True)
    net = GRSnn2LayersRNN(in_neurons=48, hidden_neurons=cfg["hidden"], out_neurons=cfg["out"]).set_timestep(1e-6)
    snn_step = talamo.snn.containers.Snn(net)                 # as singalong.model.build_pipeline
    enc = talamo.encoders.c1.IFEncoder(48, threshold_initializer=talamo.initializers.Constant(
        float(np.median(w["enc_threshold"]))))
    pipe = Pipeline([Banner(), enc, snn_step], custom_kernel_dirs=[Path(__file__).parent / "banner"])
    pipe.trainable_params.load(str(Path(run) / "trainable_params_quantized.json"))
    return pipe, snn_step, w, cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--npz", type=Path, required=True)
    a = ap.parse_args()
    run = a.run.resolve()
    app = run / APP
    if app.exists():
        raise SystemExit(f"{app} exists; remove it to rebuild")
    pipe, _, w, cfg = load(run, a.npz)
    X = np.random.default_rng(42).integers(0, 101, (125, 48)).astype(np.float32)
    ds = data.dense_dataset(torch.tensor(X[None]), torch.zeros(1, 125, 14))
    software = pipe(ds).data_as_numpy()[0][:125]
    assert np.array_equal(software, chipref.run(w, X, **chipref.CHIP)), "banner pipeline differs from the reference"
    result = dict(app=APP, software_equals_reference=True, text=TEXT)
    scratch = run / "build_tmp"
    scratch.mkdir(exist_ok=True)
    os.chdir(scratch)
    pipe.to(talamo.device.c1.Soc(simulate=False, clock_frequency=build_chip.CLOCK_MHZ))
    pipe.generate_embedded_application(app_path=str(app), dataset=ds)
    pipe.build_embedded_application(app_path=str(app))
    source = (app / "pipeline.c").read_text()
    result["kernels"] = sorted(set(re.findall(r'#include "kernel_(\w+)\.h"', source)))
    log = (app / "app_build_log.log").read_text(errors="replace")
    memory = re.search(r"ram:\s+(\d+) B\s+(\d+) KB\s+([\d.]+)%", log)
    if memory:
        result["ram_bytes"], result["ram_percent"] = int(memory[1]), float(memory[3])
    boot = app / "bin/c1/release/app.c1boot"
    result["boot_image_bytes"] = boot.stat().st_size if boot.exists() else None
    (run / "build_banner.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
