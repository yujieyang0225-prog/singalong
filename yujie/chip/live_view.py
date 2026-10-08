#!/usr/bin/env python3
"""Live view for the sing-along loop. Wraps live_chip.py without changing it.

Takes the same options as live_chip.py and serves one local page, in the look of the team's
slide deck, that shows while the loop runs: the microphone level, the tracked note and what the
tracker is doing, the twelve pitch-class values, a piano roll of the notes played in the last
15 seconds with the raw output under it, the active mode (chip or laptop only) and, in a footer,
the round trip, the delay from voice to note and the error counts. It only observes: the sink,
the selector and the loop are the existing ones and get the same calls as before.

    python -u yujie/chip/live_view.py \
      --source "mic:MacBook Pro Microphone" --mode chip --worker-debug --model A_vsP_chip_shift3 --banner \
      --selector pitch --playout-delay 1.25 --start-attempts 5 --slow-limit 6 --sink synth
    then open http://127.0.0.1:8765

Extra options: --view-port N (default 8765), --open-browser.
"""
from __future__ import annotations

import argparse
from collections import deque
import http.server
import json
import re
import sys
import threading
import time
import webbrowser

import numpy as np

import live_chip

NAMES = live_chip.fe.NAMES
WINDOW = 15                                   # seconds of notes and chip output the page shows


class View:
    """What the page shows. Written by the loop's threads, read by the HTTP thread."""

    def __init__(self):
        self.started = time.time()
        self.phase, self.error = 'starting', None
        self.run = self.args = None
        self.mic_db, self.voiced = -120.0, False
        self.evidence = [0.0] * 12
        self.label, self.confidence, self.melody_pc = None, 0.0, None
        self.tracking, self.tracked = False, None
        self.notes, self.cc11, self.note_ons, self.port = [], None, 0, None
        self.events = deque(maxlen=8)
        self.segments = deque(maxlen=120)     # notes sent: dict(start, end, note, velocity); end is None while it sounds
        self.runs = deque(maxlen=400)         # strongest pitch class per frame, run-length encoded: [start, end, class]
        self.selector = None

    def now(self):
        """Seconds on the page's clock, the one `elapsed` uses."""
        return time.time() - self.started

    def mic(self, pcm, row):
        db = 20 * float(np.log10(np.sqrt(np.mean(np.square(pcm))) + 1e-9))
        self.mic_db = max(db, self.mic_db - 0.6)              # fast attack, about 40 dB/s release
        if row is not None:
            self.voiced = bool(row[3])

    def follow(self, hmm, melody, state, post):
        self.evidence = [0.0] * 12 if melody is None else [round(float(v), 3) for v in melody]
        self.label, self.confidence = hmm.names[state], float(post[state])
        note_pc = getattr(hmm, 'note_pc', None)               # only the score follower knows the melody note
        self.melody_pc = None if note_pc is None else int(note_pc[state])
        self.tracking = hasattr(hmm, 'note')                  # direct pitch tracking: show its note
        self.tracked = getattr(hmm, 'note', None)
        self.selector = hmm                                   # read again in snapshot(), never written
        top = max(self.evidence)
        top = self.evidence.index(top) if top > 0 else None   # the class the bars label; None when all are zero
        t = self.now()
        if self.runs and self.runs[-1][2] == top:
            self.runs[-1][1] = t
        else:
            self.runs.append([t, t, top])

    def midi(self, kind, note, velocity=None):
        t = self.now()
        open_segment = next((s for s in reversed(self.segments) if s['note'] == note and s['end'] is None), None)
        if open_segment:                                      # a note-off ends it; so does the same note sent again
            open_segment['end'] = t
        if kind == 'on':
            self.note_ons += 1
            self.notes = sorted(set(self.notes) | {note})
            self.segments.append(dict(start=t, end=None, note=note, velocity=velocity))
        else:
            self.notes = [n for n in self.notes if n != note]
        self.events.appendleft(dict(seconds=round(t, 2), kind=kind, note=note, velocity=velocity))

    def decoder(self):
        """What the pitch tracker is doing, read from its attributes. None for the other selectors."""
        hmm = self.selector
        if not hasattr(hmm, 'note'):
            return None
        count = lambda name: getattr(hmm, name, None) or 0
        part = lambda done, needed: round(min(1.0, done / needed), 2) if needed else None
        state = dict(state='silence', progress=None, candidate=None, held=None)
        if hmm.note is not None and count('quiet') > 0:
            state.update(state='releasing', progress=part(count('quiet'), count('release')))
        elif count('lead') > 0:
            candidate = getattr(hmm, 'candidate', None)
            state.update(state='candidate', progress=part(count('lead'), count('switch')),
                         candidate=None if candidate is None else NAMES[int(candidate) % 12])
        elif hmm.note is not None:
            state.update(state='note', held=round(count('dwell') / (count('fps') or live_chip.fe.FPS), 2))
        return state

    def state(self):
        """The /state response. Compact, because the page fetches it ten times a second."""
        return json.dumps(self.snapshot(), separators=(',', ':'))

    def snapshot(self):
        now, run = self.now(), self.run
        left = now - WINDOW                   # notes and raw runs are sent in seconds from the window's left edge,
        shown = lambda t: None if t is None else round(t - left, 2)     # so /state stays small however long the run
        raw = [[shown(start), shown(end), top] for start, end, top in list(self.runs) if end >= left]
        notes = []
        for s in list(self.segments):
            end = s['end']                    # read once: the loop's thread may end the note meanwhile
            if end is None or end >= left:
                notes.append([shown(s['start']), shown(end), s['note'], s['velocity']])
        m = run.metrics if run else {}
        windows = m.get('windows') or []
        last = windows[-1] if windows else None
        latency = m.get('output_latency_seconds')
        latency = latency[-300:] if isinstance(latency, list) else []
        summary = m.get('output_latency_seconds_summary') or {}
        age = None
        if last and run.epoch is not None and self.phase == 'running':
            age = time.perf_counter() - (run.epoch + last['submitted_seconds'] + last['host_roundtrip_seconds'])
        phase, error = self.phase, self.error
        if phase == 'running' and run and run.stop.is_set() and not run.errors.empty():
            failure = run.errors.queue[0]             # the loop has failed; shutdown may still be running
            phase, error = 'error', f'{type(failure).__name__}: {failure}'
        position = re.fullmatch(r'(.+) in (.+) @note(\d+)', self.label or '')
        mode = getattr(self.args, 'mode', None)
        return dict(
            phase=phase, error=error, elapsed=round(now, 1),
            mode=mode, source=getattr(self.args, 'source', None),
            model=(getattr(self.args, 'model', None) or live_chip.DEFAULT_MODEL) if mode == 'chip' else None,
            sink=getattr(self.args, 'sink', None), port=self.port,
            audio_seconds=round(run.total / live_chip.fe.FPS, 1) if run else 0,
            mic=dict(db=round(self.mic_db, 1), voiced=self.voiced),
            evidence=self.evidence,
            follower=dict(label=self.label, confidence=round(self.confidence, 2),
                          melody=None if self.melody_pc is None else NAMES[self.melody_pc],
                          tracking=self.tracking, tracked=self.tracked,
                          chord=position.group(1) if position else None,
                          key=position.group(2) if position else None,
                          note_index=int(position.group(3)) if position else None),
            midi=dict(notes=self.notes, cc11=self.cc11, note_ons=self.note_ons, events=list(self.events)),
            timeline=dict(now=round(now, 2), window=WINDOW, notes=notes), raw=raw, decoder=self.decoder(),
            snn=dict(windows=len(windows),
                     roundtrip=last['host_roundtrip_seconds'] if last else None,
                     recent_roundtrips=[w['host_roundtrip_seconds'] for w in windows[-60:]],
                     exact=last.get('exact_reference') if last else None,
                     age=None if age is None else round(age, 2),
                     boot_seconds=m.get('boot_seconds')),
            latency=dict(last=latency[-1] if latency else None,
                         median=float(np.median(latency)) if latency else summary.get('median'),
                         playout_delay=getattr(self.args, 'playout_delay', None),
                         late=m.get('late_outputs', 0), skipped=m.get('skipped_updates', 0),
                         slow=m.get('slow_round_trips', 0),
                         connection_failures=m.get('connection_failures', 0),
                         delivered=m.get('delivered_frames', 0)))


VIEW = View()


def tap_sink(sink):
    """Record what the existing sink is asked to send, after it sent it. The sink keeps its class."""
    VIEW.port = getattr(sink, 'label', None) or getattr(getattr(sink, 'out', None), 'name', type(sink).__name__)
    note_on, note_off, control = sink.note_on, sink.note_off, sink.control

    def on(ch, n, vel):
        note_on(ch, n, vel)
        VIEW.midi('on', int(n), int(vel))

    def off(ch, n):
        note_off(ch, n)
        VIEW.midi('off', int(n))

    def cc(ch, number, value):
        control(ch, number, value)
        if number == 11:
            VIEW.cc11 = int(value)

    sink.note_on, sink.note_off, sink.control = on, off, cc
    VIEW.phase = 'running'                    # the sink is made once the board (if any) is ready
    return sink


def tap_selector(hmm):
    step = hmm.step
    VIEW.selector = hmm                       # known from the start, so the page can show its state before the first frame

    def tapped(melody=None, chord=None, onset=True):
        state, post = step(melody, chord, onset=onset)
        VIEW.follow(hmm, melody, state, post)
        return state, post

    hmm.step = tapped
    return hmm


class ViewRun(live_chip.LiveRun):
    def __init__(self, args):
        super().__init__(args)
        VIEW.run, VIEW.args = self, args

    def add_pcm(self, front, pcm):
        super().add_pcm(front, pcm)
        VIEW.mic(pcm, self.history[-1] if self.history else None)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/state'):
            body, kind = VIEW.state().encode(), 'application/json'
        elif self.path == '/':
            body, kind = PAGE.encode(), 'text/html; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--view-port', type=int, default=8765)
    own.add_argument('--open-browser', action='store_true')
    a, sys.argv[1:] = own.parse_known_args()
    server = http.server.ThreadingHTTPServer(('127.0.0.1', a.view_port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{a.view_port}'
    print(f'LIVE VIEW: {url}', flush=True)
    if a.open_browser:
        webbrowser.open(url)
    host = live_chip.host
    def make_sink(spec, make=host.make_sink):
        kind, _, arg = spec.partition(':')
        if kind == 'synth':                   # --sink synth[:output device number]: built-in, nothing to install
            from synth_sink import LiveSynthSink
            return tap_sink(LiveSynthSink(int(arg) if arg else None))
        return tap_sink(make(spec))
    host.make_sink = make_sink
    live_chip.make_selector = lambda a, make=live_chip.make_selector: tap_selector(make(a))
    live_chip.LiveRun = ViewRun
    linger = 3
    try:
        live_chip.main()
        VIEW.phase = 'stopped'
    except SystemExit:
        linger = 0
        raise
    except KeyboardInterrupt:
        VIEW.phase = 'stopped'
    except BaseException as error:
        VIEW.phase, VIEW.error, linger = 'error', f'{type(error).__name__}: {error}', 5
        raise
    finally:
        time.sleep(linger)                    # let the page show how the run ended
        server.shutdown()


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SofiNNa live</title>
<style>
/* The look of the SofiNNa deck. Blue is what the chip hears, green what the decoder decides, rust-orange what is played. */
:root{color-scheme:light;--page:#f3f1ec;--card:#fcfcfb;--line:#e8e5de;--ink:#16202e;--text:#4a5363;--muted:#5f6673;
  --hears:#2563c9;--hears-soft:#e6eefb;--decides:#177e6b;--plays:#c2531b;--accent:#a8441a;--rule:#000000;
  --serif:"Goudy Old Style","Sorts Mill Goudy","Iowan Old Style","Hoefler Text",Baskerville,Georgia,serif;
  --sans:system-ui,-apple-system,sans-serif}
*{box-sizing:border-box}
html{font-size:min(1.25vw,2.2222vh)}          /* 16 px at 1280 x 720: the page scales with the screen, like a slide */
body{margin:0;height:100vh;display:flex;flex-direction:column;background:var(--page);color:var(--text);
  font:1rem/1.35 var(--serif)}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;padding:1.375rem 3.5rem .75rem}
.kicker{font-weight:700;font-size:.78rem;letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}
h1{margin:.2rem 0 0;font-weight:400;font-size:2.5rem;line-height:1;color:var(--ink)}
.tags{display:flex;gap:.5rem;padding-bottom:.15rem}
.tag{padding:.2rem .65rem .25rem;border:1px solid transparent;border-radius:4px;font-weight:600;font-size:.94rem;
  line-height:1.2;white-space:nowrap}
.tag.chip{background:var(--hears);color:#fff}
.tag.laptop,.tag.off{border-color:var(--muted);color:var(--muted)}
.tag.ok{background:var(--ink);color:#fff}
.tag.wait{border-color:var(--ink);color:var(--ink)}
.tag.bad{background:var(--accent);color:#fff}
.rule{flex:none;height:1px;background:var(--rule)}
#banner{display:none;align-items:baseline;gap:.6rem;margin:.75rem 3.5rem 0;padding:.5rem .75rem;border:1px solid var(--accent);
  border-radius:10px;background:var(--card);color:var(--ink);overflow-wrap:anywhere}
/* Three small monitors on top; the piano roll gets the rest of the screen. */
main{flex:1;min-height:0;display:grid;grid-template-rows:minmax(0,3fr) minmax(0,7fr);gap:.875rem;padding:.875rem 3.5rem 0}
.trio{display:grid;grid-template-columns:minmax(0,2.4fr) minmax(0,5fr) minmax(0,6.6fr);gap:.875rem;min-height:0}
.card{display:flex;flex-direction:column;min-width:0;min-height:0;padding:.75rem 1rem;background:var(--card);
  border:1px solid var(--line);border-radius:10px}
.cardhead{display:flex;align-items:baseline;justify-content:space-between;gap:1rem}
h2{margin:0 0 .3rem;font-weight:700;font-size:.75rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
.caption{font:.72rem/1.3 var(--sans);color:var(--muted)}
.notebox{flex:1;min-height:0;display:flex;align-items:center;gap:1rem}
#note{flex:none;min-width:12.6rem;font-weight:700;font-size:6rem;line-height:1;color:var(--muted)}
#note.on{color:var(--plays)}
#notesub:empty{display:none}
.side{flex:1;min-width:0;display:flex;flex-direction:column;gap:.6rem}
#decoder{min-height:1.75rem;color:var(--decides);font-weight:600;white-space:nowrap}
#decoder .meter{height:.25rem;margin-top:.2rem}
#decoder .meter i{background:var(--decides)}
.foot{margin-top:auto}
.meter{height:.375rem;margin-top:.25rem;border-radius:3px;background:var(--line);overflow:hidden}
.meter i{display:block;height:100%;width:0;background:var(--plays);transition:width .08s linear}
#micdb{font-weight:700;font-size:2.25rem;line-height:1.1;color:var(--ink);font-variant-numeric:tabular-nums}
#micsub{display:flex;align-items:center;gap:.4rem;font-size:.94rem;color:var(--muted)}
#micsub::before{content:"";flex:none;width:.55rem;height:.55rem;border:1.5px solid var(--muted);border-radius:50%}
#micsub.on{color:var(--ink);font-weight:600}
#micsub.on::before{border-color:var(--ink);background:var(--ink)}
.meter.mic{height:.625rem}
.meter.mic i{background:var(--ink)}           /* the input is neither chip, decoder nor note: no meaning colour */
.scale{display:flex;justify-content:space-between;margin-top:.2rem;font:.69rem/1.2 var(--sans);color:var(--muted)}
#bars{flex:1;min-height:0;display:grid;grid-template-columns:repeat(12,1fr);gap:.5rem;align-items:end;padding-top:1.4rem;
  border-bottom:1px solid var(--ink)}
#bars div{position:relative;height:100%;display:flex;align-items:flex-end;justify-content:center}
#bars i{display:block;width:62%;height:0;background:var(--hears);border-radius:3px 3px 0 0;transition:height .08s linear}
#bars b{position:absolute;left:50%;transform:translateX(-50%);padding:0 .3rem;border-radius:4px;background:var(--hears-soft);
  color:var(--hears);font:600 .72rem/1.5 var(--sans);font-variant-numeric:tabular-nums}
#bars b:empty{display:none}
#barlabels{display:grid;grid-template-columns:repeat(12,1fr);gap:.5rem;margin-top:.2rem;text-align:center;font-size:.94rem;
  color:var(--muted)}
#barlabels .top{color:var(--ink);font-weight:700}
body.laptop #bars i{background:var(--muted)}  /* blue is kept for what comes from the chip */
body.laptop #bars b{background:var(--line);color:var(--text)}
canvas{display:block;width:100%}
.stage{flex:1;min-height:0;position:relative}
#timeline{position:absolute;inset:0;height:100%}
#rollstate{display:none;position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);background-color:var(--card)}
#rollstate.bad{background-color:var(--accent)}
.sub{margin-top:.5rem}
.sub h2{color:var(--muted)}
#rawstrip{height:5.25rem}                     /* twelve rows of 7 px */
body.idle canvas{opacity:.4}                  /* not running: the last picture stays, faded, under a label that says why */
body.idle #rollstate{display:block}
footer{display:flex;flex-wrap:wrap;padding:.5rem 3.5rem .75rem;font:.6875rem/1.3 var(--sans);color:var(--muted);white-space:nowrap}
footer>span+span::before{content:"·";margin:0 .4em}
footer b{font-weight:600;color:var(--text);font-variant-numeric:tabular-nums}
footer b.bad{color:var(--accent)}
</style></head><body>
<header>
  <div><div class="kicker">Live demo</div><h1><b>S</b>ofi<b>NN</b>a</h1></div>
  <div class="tags"><span id="mode" class="tag off">Connecting</span><span id="status" class="tag wait">Connecting</span></div>
</header>
<div class="rule"></div>
<div id="banner"><span class="tag bad">Error</span><span id="bannertext"></span></div>
<main>
  <div class="trio">
    <section class="card"><h2>Microphone</h2>
      <div id="micdb">–</div><div id="micsub">No audio yet</div>
      <div class="foot"><div class="meter mic" title="Input level, dB relative to full scale"><i id="micbar"></i></div>
        <div class="scale"><span>−70 dB</span><span>0 dB</span></div></div></section>
    <section class="card"><h2>Note</h2>
      <div class="notebox"><div id="note">–</div>
        <div class="side"><div class="caption" id="notesub"></div>
          <div id="decoder"><div id="dectext"></div><div class="meter" id="decmeter"><i id="decbar"></i></div></div>
          <div><div class="caption">Loudness (expression)</div><div class="meter"><i id="ccbar"></i></div></div></div></div></section>
    <section class="card"><div class="cardhead"><h2 id="evtitle">What the chip hears</h2><span class="caption" id="evcaption"></span></div>
      <div id="bars"></div><div id="barlabels"></div></section>
  </div>
  <section class="card">
    <div class="cardhead"><h2>MIDI piano roll</h2><span class="caption">notes played, last 15 seconds</span></div>
    <div class="stage"><canvas id="timeline"></canvas><span id="rollstate" class="tag wait"></span></div>
    <div class="cardhead sub"><h2 id="rawtitle">Raw SNN output</h2><span class="caption" id="rawlabel"></span></div>
    <canvas id="rawstrip"></canvas></section>
</main>
<footer>
  <span id="f-model">–</span><span id="f-octave">octave chosen on the laptop</span>
  <span>chip round trip <b id="f-rt">–</b></span>
  <span>voice to note <b id="f-lat">–</b></span><span>late frames <b id="f-late">–</b></span>
  <span>skipped updates <b id="f-skipped">–</b></span><span>slow round trips <b id="f-slow">–</b></span>
  <span>connection errors <b id="f-conn">–</b></span>
</footer>
<script>
const NAMES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const $=id=>document.getElementById(id), name=n=>NAMES[n%12]+(Math.floor(n/12)-1);
const bars=$('bars'), labels=$('barlabels');
NAMES.forEach(n=>{const d=document.createElement('div');d.innerHTML='<b></b><i></i>';bars.appendChild(d);
  const l=document.createElement('div');l.textContent=n;labels.appendChild(l);});
const css=getComputedStyle(document.documentElement), colour=n=>css.getPropertyValue('--'+n).trim();
let last=null, shown=null, was=false, clock=0;   // the last answer of /state; what the roll shows; was the loop live; its clock
function tag(id,cls,text){const e=$(id);e.className='tag '+cls;e.textContent=text;}
function sheet(c){             // a cleared canvas at the screen's resolution; sizes come back in CSS pixels
  const r=c.getBoundingClientRect(), k=window.devicePixelRatio||1, w=Math.round(r.width*k), h=Math.round(r.height*k);
  if(c.width!==w||c.height!==h){c.width=w;c.height=h;}
  const g=c.getContext('2d');g.setTransform(k,0,0,k,0,0);g.clearRect(0,0,r.width,r.height);return [g,r.width,r.height];
}
function draw(s,chip){         // once per poll: a piano roll of the notes played and, under it on the same time axis, the raw output
  const rem=parseFloat(css.fontSize), keys=2.75*rem, sans=colour('sans'), win=s.timeline.window, notes=s.timeline.notes;
  const black=n=>[1,3,6,8,10].includes(n%12), px=Math.round;
  let [g,W,H]=sheet($('timeline'));
  const x=t=>px(keys+Math.max(0,Math.min(1,t/win))*(W-keys));       // t: seconds from the left edge; now is at the right
  const shade=(y,h)=>{g.globalAlpha=.55;g.fillStyle=colour('line');g.fillRect(keys,y,W-keys,h);g.globalAlpha=1;};
  const seconds=(y,h)=>{for(let t=1;t<win;t++){g.globalAlpha=t%5?.8:.4;g.fillStyle=colour(t%5?'line':'muted');g.fillRect(x(t),y,1,h);}
    g.globalAlpha=1;};                                               // a line every second, stronger every five
  const now=(y,h)=>{g.fillStyle=colour('ink');g.fillRect(W-2,y,2,h);};
  const keyboard=(tops,noteOf,label)=>{                              // one key per row; white keys meet behind the black ones
    const n=tops.length-1;g.fillStyle='#fff';g.fillRect(0,tops[0],keys,tops[n]-tops[0]);g.textBaseline='middle';g.textAlign='right';
    for(let i=0;i<n;i++){const note=noteOf(i), y=tops[i], h=tops[i+1]-y, text=label(note);
      if(black(note)){g.fillStyle=colour('line');g.fillRect(keys*.62,px(y+h/2),keys*.38,1);g.fillStyle=colour('ink');
        g.beginPath();g.roundRect?g.roundRect(0,y+.5,keys*.62,h-1,[0,2,2,0]):g.rect(0,y+.5,keys*.62,h-1);g.fill();}
      else if(!black(note+11)&&i<n-1){g.fillStyle=colour('line');g.fillRect(0,tops[i+1]-1,keys,1);}   // E|F and B|C
      if(text&&h>=.5*rem){g.fillStyle=colour('ink');g.font=`600 ${Math.min(.62*rem,h-1)}px ${sans}`;g.fillText(text,keys-.25*rem,y+h/2+.5);}}
    g.textAlign='left';g.fillStyle=colour('muted');g.fillRect(keys-1,tops[0],1,tops[n]-tops[0]);};
  let lo=48, hi=72;                                                  // C3 to C5 while nothing is visible
  if(notes.length){lo=Math.min(...notes.map(n=>n[2]))-2;hi=Math.max(...notes.map(n=>n[2]))+2;
    if(hi-lo<12){lo=Math.floor((lo+hi)/2)-6;hi=lo+12;}}             // at least one octave
  const ruler=px(1.05*rem), count=hi-lo+1, tops=[...Array(count+1)].map((_,i)=>ruler+px(i*(H-ruler)/count));
  const top=n=>tops[hi-n], tall=n=>tops[hi-n+1]-tops[hi-n];          // one row per semitone, on whole pixels
  g.globalAlpha=.5;g.fillStyle=colour('line');g.fillRect(keys,0,W-keys,ruler);g.globalAlpha=1;   // the ruler: seconds before now
  g.fillStyle=colour('muted');g.fillRect(keys,ruler-1,W-keys,1);g.textBaseline='middle';
  for(let t=0;t<win;t++){const five=t%5===0;g.fillStyle=colour('muted');g.fillRect(x(t),five?0:ruler-5,1,five?ruler:5);
    g.font=`${five?'600 ':''}${.62*rem}px ${sans}`;g.fillStyle=colour(five?'text':'muted');
    g.fillText((win-t)+(five?' s ago':''),x(t)+.25*rem,ruler/2);}
  for(let n=lo;n<=hi;n++){if(black(n))shade(top(n),tall(n));        // the rows of the black keys are shaded
    if(n%12===0){g.fillStyle=colour('line');g.fillRect(keys,top(n)+tall(n)-1,W-keys,1);}}
  seconds(ruler,H-ruler);
  for(const [start,end,note] of notes){                              // every note sent, in its row; none is merged or dropped
    const x0=x(start), w=Math.max((end===null?W-2:x(end))-x0,2), y=top(note), h=tall(note), size=Math.min(.69*rem,h-4);
    g.globalAlpha=end!==null&&end-start<.15?.6:1;g.fillStyle=colour('plays');
    g.beginPath();g.roundRect?g.roundRect(x0,y+.5,w,h-1,Math.min(2,w/2)):g.rect(x0,y+.5,w,h-1);g.fill();
    if(w>3){g.strokeStyle='rgba(0,0,0,.22)';g.lineWidth=1;g.strokeRect(x0+.5,y+1,w-1,h-2);}
    g.globalAlpha=1;g.font=`600 ${size}px ${sans}`;                  // the name only where it can be read and fits its bar
    if(size>=.56*rem&&g.measureText(name(note)).width+.5*rem<=w){g.fillStyle='#fff';g.fillText(name(note),x0+.25*rem,y+h/2+.5);}
  }
  keyboard(tops,i=>hi-i,n=>n%12===0?name(n):null);
  now(0,H);g.beginPath();g.roundRect?g.roundRect(W-2.5*rem,1,2.5*rem,ruler-2,2):g.rect(W-2.5*rem,1,2.5*rem,ruler-2);g.fill();
  g.fillStyle='#fff';g.font=`700 ${.6*rem}px ${sans}`;g.textAlign='center';g.fillText('NOW',W-1.25*rem,ruler/2+.5);g.textAlign='left';
  [g,W,H]=sheet($('rawstrip'));                                      // twelve rows, C at the bottom, with the same keys and seconds
  const rows=[...Array(13)].map((_,i)=>px(i*H/12));
  for(let c=0;c<12;c++)if(black(c))shade(rows[11-c],rows[12-c]-rows[11-c]);
  seconds(0,H);
  g.fillStyle=colour(chip?'hears':'muted');
  for(const [start,end,cls] of s.raw)if(cls!==null)g.fillRect(x(start),rows[11-cls]+1,Math.max(x(end)-x(start),2),Math.max(rows[12-cls]-rows[11-cls]-2,2));
  keyboard(rows,i=>11-i,n=>n===0?'C':null);
  now(0,H);
}
function figure(id,text,bad){const e=$(id);e.textContent=text;e.className=bad?'bad':'';}
function render(s){
  if(s.elapsed<clock)shown=null;                                     // the loop's clock went back: a new run, which starts with an empty roll
  clock=s.elapsed;
  const chip=s.mode==='chip', live=s.phase==='running';             // only a running loop has live data to show
  document.body.className=(chip?'chip':'laptop')+(live?'':' idle');
  if(s.mode)tag('mode',chip?'chip':'laptop',chip?'CHIP · Pulsar C1':'LAPTOP ONLY');
  $('banner').style.display=s.error?'flex':'none';
  $('bannertext').textContent=s.error||'';
  if(s.phase==='lost')tag('status','bad','Not connected: the pipeline is not running');
  else if(s.phase==='error')tag('status','bad','Stopped with an error');
  else if(s.phase==='stopped')tag('status','off','Stopped');
  else if(s.phase==='starting')tag('status','wait',chip?'Starting the board (about 20 s)':'Starting');
  else if(chip&&s.snn.age!==null&&s.snn.age>2)tag('status','wait','No chip result for '+s.snn.age.toFixed(1)+' s');
  else if(chip&&!s.snn.windows)tag('status','wait','Waiting for the first 2 s of audio');
  else tag('status','ok',chip?'Running on the chip':'Running on the laptop');
  tag('rollstate',$('status').classList.contains('bad')?'bad':'wait',$('status').textContent);   // shown over the faded roll
  const f=s.follower, m=s.midi, l=s.latency, ev=live?s.evidence:s.evidence.map(()=>0), top=Math.max(...ev), topi=ev.indexOf(top);
  const tracked=live?f.tracked:null;                                 // a stopped tracker keeps its last note: do not show it
  if(f.tracking){$('note').textContent=tracked===null?'–':name(tracked);$('note').className=tracked===null?'':'on';
    $('notesub').textContent='';}
  else{$('note').textContent=top>0?NAMES[topi]:'–';$('note').className=live&&m.notes.length?'on':'';
    $('notesub').textContent=!live?'':(top>0?'Strongest pitch class heard':'Nothing heard')+(f.label?' · '
      +(f.note_index?`song position: note ${f.note_index}, key ${f.key}, chord ${f.chord}`+(f.melody?`, expects ${f.melody}`:''):f.label)
      +` · confidence ${f.confidence.toFixed(2)}`:' · waiting for the score follower');}
  $('ccbar').style.width=live?((m.cc11||0)/127*100)+'%':'0';
  const d=s.decoder;
  $('decoder').style.display=d?'':'none';
  $('decoder').style.visibility=live?'visible':'hidden';
  if(d){$('dectext').textContent={releasing:'Releasing',candidate:'Hearing '+(d.candidate||''),
      note:'Note held '+(d.held||0).toFixed(1)+' s',silence:'Silence'}[d.state];
    $('decmeter').style.visibility=live&&d.progress!==null?'visible':'hidden';
    $('decbar').style.width=((d.progress||0)*100)+'%';}
  $('evtitle').textContent=chip?'What the chip hears':'What the laptop hears (no chip)';
  $('evcaption').textContent=chip?'spike rate per pitch class, last 8 frames':'pitch estimate per pitch class';
  $('rawtitle').textContent=chip?'Raw SNN output':'Raw laptop estimate (no chip)';
  $('rawlabel').textContent='strongest pitch class per frame · C at the bottom, B at the top';
  const scale=Math.max(1,top);
  [...bars.children].forEach((d,i)=>{const v=ev[i]/scale;d.lastChild.style.height=(v*100)+'%';
    d.title=NAMES[i]+': '+ev[i].toFixed(2);d.firstChild.textContent=(top>0&&i===topi)?ev[i].toFixed(2):'';
    d.firstChild.style.bottom='calc('+(v*100)+'% + .2rem)';labels.children[i].className=(top>0&&i===topi)?'top':'';});
  $('f-model').textContent=chip?(s.model||'–')+' · 104 spiking neurons':'laptop pitch estimate, no chip';
  $('f-octave').style.display=f.tracking?'':'none';
  const heard=live&&s.audio_seconds>0, db=Math.max(-70,Math.min(0,s.mic.db));
  $('micdb').textContent=heard?db.toFixed(0).replace('-','−')+' dB':'–';
  $('micbar').style.width=heard?((db+70)/70*100)+'%':'0';
  $('micsub').textContent=heard?(s.mic.voiced?'Voice detected':'Silence'):(live?'No audio yet':'Not listening');
  $('micsub').className=heard&&s.mic.voiced?'on':'';
  $('f-rt').textContent=chip?(s.snn.roundtrip?Math.round(s.snn.roundtrip*1000)+' ms':'–'):'n/a';
  $('f-lat').textContent=l.median?l.median.toFixed(2)+' s':'–';
  figure('f-late',l.late,l.late>0.02*l.delivered);
  $('f-skipped').textContent=l.skipped;$('f-slow').textContent=l.slow;
  figure('f-conn',l.connection_failures,l.connection_failures>0);
  if(live||was||!shown){        // drawn while live, once more on stopping (the last note is closed), then kept as it is;
    shown=[live||was?s:Object.assign({},s,{timeline:Object.assign({},s.timeline,{notes:[]}),raw:[]}),chip];draw(...shown);}
  was=live;                     // an empty roll if the page opens on a loop that is not running
}
addEventListener('resize',()=>{if(shown)draw(...shown);});            // a kept picture is redrawn sharp when the window changes
async function tick(){
  let s=null;
  try{s=await (await fetch('/state',{cache:'no-store'})).json();}catch(e){}
  try{
    if(s)render(last=s);
    else if(last)render(['error','stopped'].includes(last.phase)?last:Object.assign({},last,{phase:'lost'}));
    else tag('status','bad','Not connected: the pipeline is not running');
  }finally{setTimeout(tick,s?100:700);}
}
tick();
</script></body></html>
'''


if __name__ == '__main__':
    main()
