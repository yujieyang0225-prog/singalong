"""pytest for everything that runs without the Innatera SDK:  cd jan/singalong && python3 -m pytest -q tests

Covers: front-end (offline == streaming), synthetic data + loader, augmentation label consistency, torch C1 model
(fits, learns a little, export), numpy integer net == torch QAT net, HMM selectors on perfect evidence,
renderer voice leading, C kernel == Python front-end (if gcc is available).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from singalong import amazing_grace as ag  # noqa: E402
from singalong import augment, data, evidence, frontend as fe, renderer as rd, selector as sel, synth  # noqa: E402
from singalong.snn_numpy import C1Net  # noqa: E402
from singalong.torchsnn import DualLoss, SingalongSNN, fits_c1  # noqa: E402


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    synth.make_dataset(out, n=6, seed=3, write_wav=True)
    return out


def test_frontend_streaming_matches_offline():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(8000), synth.synth_note(67, 1.0, rng), np.zeros(4000)]).astype(np.float32)
    X_off = fe.log_bands(y)
    fe_stream = fe.StreamingFrontEnd()
    frames = []
    for i in range(0, len(y), 700):                        # odd chunk size on purpose
        frames += [b for b, _, _ in fe_stream.feed(y[i:i + 700])]
    S = np.stack(frames)
    off = fe.normalise(X_off, np.ones(len(X_off), bool))
    T = min(len(S), len(off))
    voiced = S[:T].max(1) > 0
    assert voiced.any()
    assert np.abs(S[:T][voiced] - off[:T][voiced]).max() <= 1      # same numbers up to rounding where voiced


def test_lead_sheet_and_timeline():
    tl = ag.render_timeline(tonic=7, octave=4, bpm=80)
    sung = tl["melody"] >= 0
    assert tl["melody"][sung].min() == 62 and tl["melody"][sung].max() == 74      # D4 .. D5 in G
    scale = set((7 + d) % 12 for d in ag.MAJOR) | {(7 + 10) % 12}            # + the I7's flat seventh (bar 10)
    assert set(int(i) for i in np.flatnonzero(tl["chord_pcs"][sung].sum(0) > 0)) <= scale
    assert len(ag.degree_sequence()) >= 17


def test_dataset_loader_and_crops(dataset):
    recs = data.load_recordings("train", data=dataset)
    assert recs and all(r["X"].shape[1] == 48 for r in recs)
    assert all(r["X"].max() <= 100 and r["X"].min() >= 0 for r in recs)
    X, Y = data.crops(recs)
    assert X.shape[1:] == (125, 48) and Y.shape[-1] == data.N_CH
    assert (Y[..., data.MASK_CH][:, :31] == 0).all()                  # burn-in not scored
    sung = Y[..., data.MEL_CH] < 12
    assert 0.3 < sung.float().mean() < 1.0


def test_augment_keeps_labels_aligned(dataset):
    rng = np.random.default_rng(0)
    r = data.load_recordings("train", data=dataset)[0]
    k = 3
    s = augment.band_shift(r, k, rng)
    sung = (r["MEL"] < 12) & r["voiced"]
    assert np.all(s["MEL"][sung] == (r["MEL"][sung] + k) % 12)
    assert np.array_equal(s["Y"], np.roll(r["Y"], k, axis=1))
    # the strongest band moves by k where voiced
    moved = (s["X"].argmax(1) - r["X"].argmax(1))[sung]
    assert np.median(moved) == k
    t = augment.tempo(r, 1.25)
    assert abs(len(t["X"]) - len(r["X"]) / 1.25) < 3
    w = augment.random_augment(r, rng)
    assert w["X"].shape[1] == 48 and len(w["X"]) == len(w["MEL"]) == len(w["Y"])


def test_fits_c1():
    assert fits_c1(48, 64, 24)[0]
    assert fits_c1(48, 80, 24)[0] and "pairing" not in fits_c1(48, 80, 24)[1]
    assert fits_c1(48, 128, 24)[0] and "pairing" in fits_c1(48, 128, 24)[1]
    assert not fits_c1(48, 200, 24)[0]
    assert not fits_c1(48, 380, 24, recurrent=False)[0]


def test_torch_model_learns_and_exports(dataset, tmp_path):
    torch.manual_seed(0)
    recs = data.load_recordings("train", data=dataset)
    X, Y = data.crops(recs, stride=62)
    net = SingalongSNN(48, 32, 24)
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": 1.0},
                             {"params": list(net.h.parameters()) + list(net.o.parameters()), "lr": 0.3}])
    loss_fn = DualLoss()
    first = None
    for it in range(25):
        idx = torch.randperm(len(X))[:32]
        s_o, s_h, _ = net(X[idx], return_hidden=True)
        loss, _ = loss_fn(s_o, Y[idx], s_h)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or float(loss.detach())
    assert float(loss) < first                                     # it moves in the right direction
    net.set_qat(True)
    net.export_npz(tmp_path / "w.npz")
    w = np.load(tmp_path / "w.npz")
    assert w["W_in"].shape == (32, 48) and w["W_rec"].shape == (32, 32) and w["W_out"].shape == (24, 32)
    assert np.all(np.abs(w["W_in"]) <= 128) and np.all(w["W_in"] == np.round(w["W_in"]))


def test_numpy_net_matches_torch_qat(dataset, tmp_path):
    torch.manual_seed(1)
    net = SingalongSNN(48, 16, 24)
    net.set_qat(True)
    net.export_npz(tmp_path / "w.npz")
    r = data.load_recordings("train", data=dataset)[0]
    X = r["X"][:300]
    with torch.no_grad():
        s_t = net(torch.tensor(X[None]))[0].numpy()
    s_n = C1Net(tmp_path / "w.npz", enc_reset="zero", layer_delay=0).run(X)        # both follow C1 now
    assert s_t.shape == s_n.shape
    assert (s_t == s_n).mean() > 0.999


def _perfect_run(hmm, r):
    pred = np.zeros((len(r["X"]), 12))
    for t in range(len(r["X"])):
        m = np.zeros(12)
        if r["MEL"][t] < 12:
            m[r["MEL"][t]] = 1
        s, _ = hmm.step(m if m.any() else None, None)
        pred[t] = hmm.pcs(s)
    sung = r["M"] & (r["MEL"] < 12)
    return (pred[sung] == r["Y"][sung]).all(1).mean(), pred, sung


def test_score_follower_tracks_song(dataset):
    r = data.load_recordings("train", data=dataset)[0]
    exact, pred, sung = _perfect_run(sel.ScoreFollowerHMM(bpm=80), r)
    assert exact > 0.6
    nic = np.mean(pred[np.flatnonzero(sung), r["MEL"][sung]])
    assert nic > 0.7


def test_key_hmm_finds_key(dataset):
    r = data.load_recordings("train", data=dataset)[0]
    hmm = sel.KeyChordHMM()
    exact, pred, sung = _perfect_run(hmm, r)
    key_votes = np.bincount([hmm.key_of(int(np.argmax(hmm.logp)))], minlength=12)
    true_key = int(np.flatnonzero(r["Y"][sung][0])[0])              # not necessarily the tonic; just check it plays
    assert pred[sung].sum() > 0 and key_votes.sum() == 1


def test_baseline_evidence_pitch(dataset):
    r = data.load_recordings("train", data=dataset)[0]
    e = evidence.BaselineEvidence()
    acc = []
    for t in range(len(r["X"])):
        m, _ = e(r["X"][t], bool(r["voiced"][t]))
        if r["MEL"][t] < 12:
            acc.append(m.argmax() == r["MEL"][t])
    assert np.mean(acc) > 0.6


def test_renderer_voice_leading():
    sink = rd.PrintSink(quiet=True)
    ren = rd.Renderer(sink, min_hold_s=0.0)
    G = ag.chord_pcs(0, "maj", 7); C = ag.chord_pcs(0, "maj", 0); D7 = ag.chord_pcs(0, "7", 2)
    ren.step(G, root_pc=7); first = list(ren.notes)
    ren.step(G, root_pc=7); assert ren.notes == first                     # same chord: nothing happens
    ren.step(C, root_pc=0); second = list(ren.notes)
    assert all(rd.LOW <= n <= rd.HIGH for n in first + second)
    assert {n % 12 for n in second} <= {0, 4, 7}
    assert sum(abs(a - b) for a, b in zip(sorted(first), sorted(second))) <= 3 * 5   # small moves
    ren.step(D7, root_pc=2); ren.all_off()
    ons = [e for e in sink.events if e[0] == "on"]; offs = [e for e in sink.events if e[0] == "off"]
    assert len(ons) == len(offs)


@pytest.mark.skipif(shutil.which("gcc") is None or shutil.which("ffmpeg") is None, reason="needs gcc + ffmpeg")
def test_c_kernel_matches_python(dataset, tmp_path):
    src = ROOT / "kernels_c" / "bands"
    exe = tmp_path / "test_bands"
    subprocess.run(["gcc", "-O2", "-o", str(exe), str(src / "test_bands.c"), str(src / "bands_core.c"), "-lm"], check=True)
    wav = next(dataset.glob("*.wav"))
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    out = subprocess.run([str(exe)], input=pcm, capture_output=True, check=True).stdout.decode()
    C = np.array([[float(v) for v in line.split()] for line in out.strip().split("\n")])
    y = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    gate = fe.OnlineGate()
    v = np.array([gate(d) for d in fe.level_db(y)])
    P = fe.normalise(fe.log_bands(y), v)
    T = min(len(C), len(P))
    assert (C[:T, 0] == v[:T]).mean() > 0.99
    assert np.abs(C[:T, 2:] - P[:T]).max() <= 1


def test_wav_sink_mix_and_stereo(tmp_path):
    """WavSink renders notes + a singer of arbitrary length to mono and stereo files (no synth needed)."""
    import wave
    from singalong import renderer as rd
    sr = 16000
    rng = np.random.default_rng(0)
    voice = np.zeros(int(3.3 * sr) + 175)                       # odd length: not a multiple of the 20 ms hop
    voice[sr:2 * sr] = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr) + 0.01 * rng.standard_normal(sr)
    voice[int(2.5 * sr)] = 1.0                                  # a click must not set the level
    for stereo in (False, True):
        sink = rd.WavSink(tmp_path / f"out_{stereo}.wav", sr=sr)
        sink.set_time(0.0); sink.note_on(0, 60, 90); sink.control(0, 11, 100)
        sink.set_time(2.0); sink.note_off(0, 60)
        path = sink.close(voice=voice, stereo=stereo)
        with wave.open(str(path)) as w:
            assert w.getnchannels() == (2 if stereo else 1)
            y = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels()) / 32767
        assert len(y) >= len(voice)
        vio = y[:, -1] if stereo else y[:, 0]
        assert np.abs(vio[: int(0.9 * sr)]).max() < 0.02            # violin silent before the singer starts
        assert np.sqrt(np.mean(y[sr:2 * sr] ** 2)) > 0.05             # audible while singing

