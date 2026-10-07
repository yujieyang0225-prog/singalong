#!/usr/bin/env python3
"""Board side of the live demo: microphone -> bands kernel -> IF encoder -> SNN -> spike counts to the laptop.

    jan/singalong/run.sh python3 Pulsar/IF_DNR/deploy_live.py data/runs/<run> --mode hil      # SNN on chip, rest on PC
    jan/singalong/run.sh python3 Pulsar/IF_DNR/deploy_live.py data/runs/<run> --mode standalone --build

Three stages, each a fallback for the next (do them in this order at the venue):
  --mode pc          everything on the PC, board only streams audio (MicrophoneSensor) -> proves mic + front-end
  --mode hil         Snn on the chip, front-end + encoder in Python on the PC (as sim_check.py does on the simulator)
                     -> proves the mapped network; the laptop reads output spikes and runs live.py's selector
  --mode standalone  MicrophoneInput -> custom kernel 'bands' (kernels_c/bands) -> IFEncoder -> Snn -> HostOutput
                     -> the real demo: only spike counts leave the board

What this script does today: builds the Pipeline for the chosen mode, moves the Snn to the Soc, generates and
(--build) builds the embedded application, and for --mode hil runs a loop that streams frames from the laptop
mic to the board and prints JSON lines {"rates": [...]} that live.py --source board:<port> consumes.

UNTESTED in this session (no SDK, no board). Written from the vault's Talamo SDK notes and sim_check.py.
Things to check first (vault: Talamo SDK, Pulsar SNN fabric limits):
  - MicrophoneInput(chunk_size=256, config=PDMMicConfig(sample_rate=16000, alc=off?)) : default is 8 kHz + ALC
  - the custom kernel registration: pipe.register_custom_kernels(kernels_c/) with kernel_type "bands"
  - state across frames: enable_io=False (standalone) generates no reset kernel -> state persists (wanted);
    for hil mode call snn_set_inference_events / pipeline_reset=false equivalent, or send one long sample
  - HostOutput of the 24 spike counts per frame (uint8) and the UART rate (~1.5 kB/s at 62.5 fps, fine)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import talamo  # noqa: E402
from singalong import frontend as fe  # noqa: E402
from singalong import model  # noqa: E402
from Pulsar.IF_DNR.sim_check import load_quantized  # noqa: E402

CLOCK_MHZ = 100


class BandsKernel(talamo.pipeline.elements.PipelineStep):
    """Custom kernel step: 1024 int16 samples (hop 256) -> 48 uint8 bands. C in kernels_c/bands/."""

    def __init__(self):
        super().__init__()

    def forward(self, x):                                  # software reference for simulation / evaluate
        pcm = x.detach().cpu().numpy().astype(np.float32) / 32768.0
        out = []
        for frame in pcm:
            spec = np.abs(np.fft.rfft(frame * fe.HANN, fe.NFFT)) ** 2
            X = np.log2(spec @ fe.FB.T + 1e-10)
            out.append(fe.normalise(X[None])[0])
        return torch.tensor(np.stack(out))

    def generate_embedded_info(self):
        return talamo.pipeline.embedded.EmbeddedKernelInfo(kernel_type="bands", params={})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--mode", choices=["pc", "hil", "standalone"], default="hil")
    ap.add_argument("--simulate", action="store_true", help="use Soc(simulate=True) instead of the board")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--app", type=Path, default=None)
    a = ap.parse_args()
    pipe, snn_step = load_quantized(a.run)
    soc = talamo.device.c1.Soc(simulate=a.simulate, clock_frequency=CLOCK_MHZ * 1_000_000)

    if a.mode == "standalone":
        mic = talamo.pipeline.elements.MicrophoneInput(duration=0, chunk_size=fe.HOP,
                                                       config=talamo.pipeline.elements.PDMMicConfig(sample_rate=fe.SR))
        window = talamo.pipeline.elements.SlidingWindow(window_size=fe.WIN, step=fe.HOP)
        enc = [s for s in pipe.steps if "Encoder" in type(s).__name__][0]
        live = talamo.pipeline.elements.Pipeline([mic, window, BandsKernel(), enc, snn_step,
                                                  talamo.pipeline.elements.HostOutput()])
        live.register_custom_kernels(str(Path(__file__).resolve().parents[2] / "kernels_c"))
        live.embedded_config.enable_io = True                       # spike counts go to the PC
        live.to(soc)
        app = a.app or a.run / "app_standalone"
        live.generate_embedded_application(str(app), None)
        if a.build:
            live.build_embedded_application(str(app))
        print(f"embedded app in {app}; flash it, then run live.py --source board:<port>")
        return

    snn_step.to(soc)                                                  # HIL: only the Snn on the chip
    if a.mode == "pc":
        print("mode pc: nothing deployed; stream the board mic with talamo MicrophoneSensor and run live.py --source wav/mic")
        return
    # HIL loop: laptop mic -> Python front-end -> IF encoder (software) -> Snn (chip) -> JSON lines on stdout
    import sounddevice as sd
    front = fe.StreamingFrontEnd()
    enc = [s for s in pipe.steps if "Encoder" in type(s).__name__][0]
    win = 8
    buf = []
    print(json.dumps({"info": "hil loop started"}), flush=True)
    with sd.InputStream(samplerate=fe.SR, channels=1, dtype="float32", blocksize=fe.HOP) as stream:
        while True:
            pcm, _ = stream.read(fe.HOP)
            for bands, db, v in front.feed(pcm[:, 0]):
                x = torch.tensor(bands, dtype=torch.float32)[None, None]            # [1, 1 frame, 48]
                s_in = enc(x)
                s_out = snn_step(s_in).detach().cpu().numpy().reshape(-1)[-snn_step.out_features:] \
                    if hasattr(snn_step, "out_features") else snn_step(s_in).detach().cpu().numpy().reshape(-1)
                buf.append(s_out)
                buf = buf[-win:]
                print(json.dumps({"t": time.time(), "rates": np.mean(buf, 0).round(3).tolist(), "db": round(db, 1)}),
                      flush=True)


if __name__ == "__main__":
    main()
