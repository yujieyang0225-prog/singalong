#!/usr/bin/env python3
"""Training curves of train_torch.py runs as one HTML page (standard library only, no numpy/matplotlib).

Reads data/runs/<name>/log.csv (+ metrics.json for the baseline) and writes data/runs/runs.html:
pitch accuracy, chord F1 (thr 0.2 and 0.5), loss, dead hidden neurons and hidden rate over epochs,
one line per run, rounded (QAT) epochs as filled dots, the constant-chord baseline dashed.

Fetch the logs from the cluster first (small files only):
  rsync -av --include='*/' --include='log.csv' --include='metrics.json' --exclude='*' \
        <user>@<cluster-host>:MNH26SVR/data/runs/ data/runs/
Then:
  python3 simon/plot_runs.py                      # all runs
  python3 simon/plot_runs.py 'B_*' 'C_*'          # only runs matching these patterns
"""
import csv
import fnmatch
import html
import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "data" / "runs"
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d", "#475569", "#ea580c"]
PANELS = [  # column, title, y range (None = from the data), higher is better
    ("pitch_acc", "Pitch accuracy (melody head)", (0, 1), True),
    ("val_f1_0.2", "Chord F1 @ threshold 0.2", (0, 1), True),
    ("val_f1", "Chord F1 @ threshold 0.5", (0, 1), True),
    ("loss", "Training loss", None, False),
    ("dead", "Dead hidden neurons", None, False),
    ("hidden_rate", "Hidden firing rate", None, None),
]
W, H, PAD = 420, 230, (44, 14, 16, 30)  # left, right, top, bottom


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load(run):
    """Rows of the last training in log.csv (the trainer appends, so a reused run name holds several)."""
    rows = []
    for r in csv.reader(open(run / "log.csv")):
        if r and r[0] == "epoch":
            header, rows = r, []
        elif r:
            rows.append(dict(zip(header, r)))
    base = None
    if (run / "metrics.json").exists():
        b = json.load(open(run / "metrics.json")).get("baseline") or {}
        base = {"val_f1_0.2": b.get("f1"), "val_f1": b.get("f1")}
    return rows, base


def panel(col, title, yr, runs):
    pts = {name: [(int(r["epoch"]), num(r.get(col)), r.get("qat") == "1") for r in rows] for name, rows, _ in runs}
    pts = {n: [p for p in ps if p[1] is not None and p[1] == p[1]] for n, ps in pts.items()}  # drop missing / nan
    xs = [p[0] for ps in pts.values() for p in ps] or [1]
    ys = [p[1] for ps in pts.values() for p in ps] or [0]
    x0, x1 = 0, max(max(xs), 1)
    y0, y1 = yr if yr else (min(0, min(ys)), max(ys) * 1.08 or 1)
    sx = lambda x: PAD[0] + (x - x0) / (x1 - x0) * (W - PAD[0] - PAD[1])
    sy = lambda y: H - PAD[3] - (y - y0) / (y1 - y0 or 1) * (H - PAD[2] - PAD[3])
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    for i in range(5):  # grid + y labels
        y = y0 + (y1 - y0) * i / 4
        out.append(f'<line class="grid" x1="{PAD[0]}" x2="{W - PAD[1]}" y1="{sy(y):.1f}" y2="{sy(y):.1f}"/>'
                   f'<text class="ax" x="{PAD[0] - 6}" y="{sy(y) + 4:.1f}" text-anchor="end">{y:.2f}</text>')
    step = max(1, round(x1 / 8 / 5) * 5) if x1 > 10 else 1
    for x in range(0, x1 + 1, step):
        out.append(f'<text class="ax" x="{sx(x):.1f}" y="{H - 10}" text-anchor="middle">{x}</text>')
    for i, (name, rows, base) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        if base and base.get(col) is not None:
            out.append(f'<line x1="{PAD[0]}" x2="{W - PAD[1]}" y1="{sy(base[col]):.1f}" y2="{sy(base[col]):.1f}" '
                       f'stroke="{c}" stroke-dasharray="4 4" opacity=".6"><title>{html.escape(name)} baseline '
                       f'{base[col]:.3f}</title></line>')
        ps = pts[name]
        if not ps:
            continue
        out.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="'
                   + " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in ps) + '"/>')
        for x, y, q in ps:
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{3 if q else 2}" '
                       f'fill="{c if q else "var(--bg)"}" stroke="{c}"><title>{html.escape(name)} · epoch {x}'
                       f'{" (int8)" if q else ""}: {y:.3f}</title></circle>')
    out.append("</svg>")
    return f'<figure><figcaption>{html.escape(title)}</figcaption>{"".join(out)}</figure>'


def summary(runs):
    tr = ['<tr><th>Run</th><th>Epochs</th><th>Pitch (last)</th><th>Pitch (best)</th>'
          '<th>F1@0.2 (last)</th><th>Baseline F1</th><th>Dead (last)</th></tr>']
    for i, (name, rows, base) in enumerate(runs):
        last = rows[-1] if rows else {}
        pitch = [num(r.get("pitch_acc")) for r in rows if num(r.get("pitch_acc")) == num(r.get("pitch_acc"))]
        f = lambda v: "–" if v is None or v != v else f"{v:.3f}"
        tr.append(f'<tr><td><span class="dot" style="background:{COLORS[i % len(COLORS)]}"></span>{html.escape(name)}</td>'
                  f'<td>{last.get("epoch", "–")}</td><td>{f(num(last.get("pitch_acc")))}</td>'
                  f'<td>{f(max(pitch) if pitch else None)}</td><td>{f(num(last.get("val_f1_0.2")))}</td>'
                  f'<td>{f(base and base.get("val_f1_0.2"))}</td><td>{last.get("dead", "–")}</td></tr>')
    return f'<table>{"".join(tr)}</table>'


def main(patterns):
    names = sorted(p.parent.name for p in RUNS.glob("*/log.csv"))
    if patterns:
        names = [n for n in names if any(fnmatch.fnmatch(n, p) for p in patterns)]
    if not names:
        sys.exit(f"no runs with log.csv in {RUNS}" + (f" matching {patterns}" if patterns else ""))
    runs = [(n, *load(RUNS / n)) for n in names]
    legend = "".join(f'<span><span class="dot" style="background:{COLORS[i % len(COLORS)]}"></span>{html.escape(n)}</span>'
                     for i, (n, _, _) in enumerate(runs))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Singalong runs</title><style>
:root{{--bg:#fff;--fg:#111827;--muted:#6b7280;--grid:#e5e7eb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f172a;--fg:#e5e7eb;--muted:#94a3b8;--grid:#1e293b}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:0;padding:16px}}
h1{{font-size:18px;margin:0 0 4px}} p{{color:var(--muted);margin:0 0 12px}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 16px;margin:8px 0 12px}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
.grid-panels{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,400px),1fr));gap:16px}}
figure{{margin:0}} figcaption{{font-weight:600;margin-bottom:2px}} svg{{width:100%;height:auto}}
line.grid{{stroke:var(--grid)}} text.ax{{fill:var(--muted);font-size:10px}}
table{{border-collapse:collapse;margin:16px 0;display:block;overflow-x:auto}}
th,td{{padding:4px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
</style></head><body>
<h1>Singalong training runs</h1>
<p>Filled dots = rounded int8 (QAT) epochs, the model that gets saved. Dashed = constant-chord baseline. Hover a dot for its value.</p>
<div class="legend">{legend}</div>
{summary(runs)}
<div class="grid-panels">{"".join(panel(c, t, yr, runs) for c, t, yr, _ in PANELS)}</div>
</body></html>"""
    (RUNS / "runs.html").write_text(page)
    print(f"{len(runs)} runs -> {RUNS / 'runs.html'}")


if __name__ == "__main__":
    main(sys.argv[1:])
