#!/usr/bin/env python3
"""Live view for the sing-along loop. Wraps live_chip.py without changing it.

Takes the same options as live_chip.py and serves one local page that shows, while the loop
runs: microphone level, pitch evidence, the score follower's note, the MIDI notes sent, the
active mode (chip or baseline), SNN round-trip times, latency and errors. It only observes:
the sink, the selector and the loop are the existing ones and get the same calls as before.

    python -u yujie/chip/live_view.py \
      --source mic:1 --mode chip --worker-debug --sink midi:GarageBand --playout-delay 1.25
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

    def midi(self, kind, note, velocity=None):
        if kind == 'on':
            self.note_ons += 1
            self.notes = sorted(set(self.notes) | {note})
        else:
            self.notes = [n for n in self.notes if n != note]
        self.events.appendleft(dict(seconds=round(time.time() - self.started, 2), kind=kind,
                                    note=note, velocity=velocity))

    def snapshot(self):
        run = self.run
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
        return dict(
            phase=phase, error=error, elapsed=round(time.time() - self.started, 1),
            mode=getattr(self.args, 'mode', None), source=getattr(self.args, 'source', None),
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
            body, kind = json.dumps(VIEW.snapshot()).encode(), 'application/json'
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
<title>Sing-along live</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
  --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--seq:#2a78d6;--track:#e1e0d9;
  --good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--black-key:#52514e}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;
  --ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--seq:#3987e5;--track:#2c2c2a;
  --black-key:#383835}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 32px}
header{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;margin-bottom:16px}
h1{font-size:20px;margin:0 auto 0 0;font-weight:650}
.pill{display:inline-flex;align-items:center;gap:7px;padding:5px 12px;border-radius:999px;border:1px solid var(--border);
  background:var(--surface);font-weight:600;white-space:nowrap}
.dot{width:10px;height:10px;border-radius:50%;background:var(--muted);flex:none}
.good .dot{background:var(--good)}.warn .dot{background:var(--warn)}.crit .dot{background:var(--crit)}
#mode{font-size:16px;letter-spacing:.02em}
#banner{display:none;margin-bottom:14px;padding:12px 14px;border-radius:10px;border:1px solid var(--crit);
  background:var(--surface);font-weight:600;overflow-wrap:anywhere}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}
.card{grid-column:span 4;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px;min-width:0}
.wide{grid-column:span 8}.full{grid-column:span 12}.tile{grid-column:span 2}
@media (max-width:860px){.card,.wide{grid-column:span 12}.tile{grid-column:span 6}}
h2{margin:0 0 8px;font-size:12px;font-weight:600;color:var(--ink2);text-transform:uppercase;letter-spacing:.06em}
.hero{font-size:54px;font-weight:700;line-height:1.05;min-height:58px}
.big{font-size:30px;font-weight:700;line-height:1.1}
.sub{color:var(--ink2);font-size:13px;margin-top:4px;overflow-wrap:anywhere}
.meter{height:12px;border-radius:6px;background:var(--track);overflow:hidden;margin-top:10px}
.meter i{display:block;height:100%;width:0;background:var(--seq);border-radius:0 4px 4px 0;transition:width .08s linear}
.scale{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:3px}
#bars{display:grid;grid-template-columns:repeat(12,1fr);gap:8px;height:170px;padding-top:22px;align-items:end;border-bottom:1px solid var(--axis)}
#bars div{position:relative;height:100%;display:flex;align-items:flex-end;justify-content:center}
#bars i{display:block;width:60%;max-width:34px;background:var(--seq);border-radius:4px 4px 0 0;height:0;transition:height .08s linear}
#bars b{position:absolute;left:0;right:0;text-align:center;font-size:12px;font-weight:600;color:var(--ink)}
#barlabels{display:grid;grid-template-columns:repeat(12,1fr);gap:8px;margin-top:5px;text-align:center;color:var(--ink2);font-size:13px}
#barlabels .top{color:var(--ink);font-weight:700}
#keys{display:flex;gap:2px;height:74px}
#keys div{flex:1;border-radius:0 0 4px 4px;border:1px solid var(--axis);background:var(--surface);position:relative;min-width:0}
#keys .black{background:var(--black-key);height:62%}
#keys .on{background:var(--seq);border-color:var(--seq)}
#keys span{position:absolute;bottom:-20px;left:50%;transform:translateX(-50%);font-size:12px;font-weight:700;color:var(--ink)}
#keyrange{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:24px}
.chips{display:flex;flex-wrap:wrap;gap:6px;min-height:40px;align-items:center}
.chip{padding:4px 10px;border-radius:8px;border:1px solid var(--border);font-size:20px;font-weight:700}
#events{margin:0;padding:0;list-style:none;font-size:13px;color:var(--ink2);font-variant-numeric:tabular-nums}
#events li{display:flex;gap:10px;padding:2px 0;border-bottom:1px solid var(--grid)}
#events li span:first-child{color:var(--muted);width:64px}
</style></head><body><main>
<header>
  <h1>Sing-along live</h1>
  <span id="mode" class="pill">…</span>
  <span id="status" class="pill"><span class="dot"></span><span id="statustext">Connecting</span></span>
</header>
<div id="banner"></div>
<div class="grid">
  <section class="card"><h2>Microphone</h2>
    <div class="big" id="micdb">–</div><div class="sub" id="micsub">&nbsp;</div>
    <div class="meter" title="Input level, dB relative to full scale"><i id="micbar"></i></div>
    <div class="scale"><span>−70 dB</span><span>0 dB</span></div></section>
  <section class="card"><h2>Detected note</h2>
    <div class="hero" id="note">–</div><div class="sub" id="notesub">&nbsp;</div></section>
  <section class="card"><h2>MIDI notes sounding</h2>
    <div class="chips" id="midinotes"></div><div class="sub" id="midisub">&nbsp;</div>
    <div class="meter" title="Expression (MIDI CC 11): follows how loudly you sing"><i id="ccbar"></i></div>
    <div class="scale"><span>expression 0</span><span>127</span></div></section>
  <section class="card wide"><h2 id="evtitle">Pitch evidence per pitch class</h2>
    <div id="bars"></div><div id="barlabels"></div></section>
  <section class="card"><h2>Last MIDI messages</h2><ul id="events"></ul></section>
  <section class="card full"><h2>Notes on the keyboard</h2>
    <div id="keys"></div><div id="keyrange"><span>E2</span><span>E6</span></div></section>
  <section class="card tile"><h2>SNN round trip</h2><div class="big" id="rt">–</div><div class="sub" id="rtsub">&nbsp;</div></section>
  <section class="card tile"><h2>Sound to MIDI</h2><div class="big" id="lat">–</div><div class="sub" id="latsub">&nbsp;</div></section>
  <section class="card tile"><h2>Late frames</h2><div class="big" id="late">–</div><div class="sub">more than 50 ms late</div></section>
  <section class="card tile"><h2>Skipped updates</h2><div class="big" id="skipped">–</div><div class="sub">windows dropped to catch up</div></section>
  <section class="card tile"><h2>Connection errors</h2><div class="big" id="conn">–</div><div class="sub" id="connsub">&nbsp;</div></section>
  <section class="card tile"><h2>Audio time</h2><div class="big" id="clock">–</div><div class="sub" id="clocksub">&nbsp;</div></section>
</div></main>
<script>
const NAMES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'], LOW=40, HIGH=88;
const $=id=>document.getElementById(id), name=n=>NAMES[n%12]+(Math.floor(n/12)-1);
const bars=$('bars'), labels=$('barlabels'), keys=$('keys');
NAMES.forEach(n=>{const d=document.createElement('div');d.innerHTML='<b></b><i></i>';bars.appendChild(d);
  const l=document.createElement('div');l.textContent=n;labels.appendChild(l);});
for(let n=LOW;n<=HIGH;n++){const k=document.createElement('div');k.title=name(n);
  if(NAMES[n%12].includes('#'))k.className='black';keys.appendChild(k);}
function status(cls,text){$('status').className='pill '+cls;$('statustext').textContent=text;}
function render(s){
  const chip=s.mode==='chip';
  $('mode').textContent=s.mode?(chip?'CHIP · physical Pulsar board':'BASELINE · laptop only, no chip'):'…';
  $('banner').style.display=s.error?'block':'none';
  $('banner').textContent=s.error?'✕ '+s.error:'';
  if(s.phase==='error')status('crit','✕ Stopped with an error');
  else if(s.phase==='stopped')status('','■ Stopped');
  else if(s.phase==='starting')status('warn',chip?'… Starting the board (about 20 s)':'… Starting');
  else if(chip&&s.snn.age!==null&&s.snn.age>2)status('warn','! No chip result for '+s.snn.age.toFixed(1)+' s');
  else if(chip&&!s.snn.windows)status('warn','… Waiting for the first 2 s of audio');
  else status('good',chip?'✓ Running on the chip':'✓ Running on the laptop');
  const db=Math.max(-70,Math.min(0,s.mic.db));
  $('micdb').textContent=s.audio_seconds?db.toFixed(0)+' dB':'–';
  $('micbar').style.width=((db+70)/70*100)+'%';
  $('micsub').textContent=s.audio_seconds?(s.mic.voiced?'● Voice detected':'○ Quiet'):'No audio yet';
  const f=s.follower, ev=s.evidence, top=Math.max(...ev), topi=ev.indexOf(top);
  if(f.tracking){$('note').textContent=f.tracked===null?'–':name(f.tracked);
    $('notesub').textContent=(f.tracked===null?'Silence: no note':'Tracked note (octave chosen on the laptop)')+(top>0?` · strongest pitch class now: ${NAMES[topi]}`:'');}
  else{
  $('note').textContent=top>0?NAMES[topi]:'–';
  $('notesub').textContent=(top>0?'Strongest pitch class heard':'Nothing heard')+(f.label?' · '
    +(f.note_index?`song position: note ${f.note_index}, key ${f.key}, chord ${f.chord}`+(f.melody?`, expects ${f.melody}`:''):f.label)
    +` · confidence ${f.confidence.toFixed(2)}`:' · waiting for the score follower');
  }
  $('evtitle').textContent=chip?'Chip output: spike rate per pitch class (last 8 frames)':'Laptop baseline: pitch evidence per pitch class';
  const scale=Math.max(1,top);
  [...bars.children].forEach((d,i)=>{const v=ev[i]/scale;d.lastChild.style.height=(v*100)+'%';
    d.title=NAMES[i]+': '+ev[i].toFixed(2);d.firstChild.textContent=(top>0&&i===topi)?ev[i].toFixed(2):'';
    d.firstChild.style.bottom='calc('+(v*100)+'% + 3px)';labels.children[i].className=(top>0&&i===topi)?'top':'';});
  const m=s.midi;
  $('midinotes').replaceChildren(...(m.notes.length?m.notes:[null]).map(n=>{const c=document.createElement('span');
    c.className=n===null?'sub':'chip';c.textContent=n===null?'No note sounding':name(n);return c;}));
  $('midisub').textContent=`${m.note_ons} note-ons sent · to ${s.port||s.sink||'–'}`;
  $('ccbar').style.width=((m.cc11||0)/127*100)+'%';
  [...keys.children].forEach((k,i)=>{const on=m.notes.includes(LOW+i);k.classList.toggle('on',on);
    k.replaceChildren();if(on){const t=document.createElement('span');t.textContent=name(LOW+i);k.appendChild(t);}});
  $('events').replaceChildren(...m.events.map(e=>{const li=document.createElement('li');
    const t=document.createElement('span');t.textContent=e.seconds.toFixed(2)+' s';
    const d=document.createElement('span');d.textContent=(e.kind==='on'?'note on  ':'note off ')+name(e.note)+(e.velocity?'  velocity '+e.velocity:'');
    li.append(t,d);return li;}));
  $('rt').textContent=chip?(s.snn.roundtrip?Math.round(s.snn.roundtrip*1000)+' ms':'–'):'n/a';
  $('rtsub').textContent=chip?`${s.snn.windows} windows`+(s.snn.exact===true?' · matches reference':s.snn.exact===false?' · DIFFERS from reference':''):'baseline does not use the chip';
  const l=s.latency;
  $('lat').textContent=l.median?l.median.toFixed(2)+' s':'–';
  $('latsub').textContent=l.playout_delay?`includes ${l.playout_delay} s planned delay`:'';
  $('late').textContent=l.late;$('skipped').textContent=l.skipped;$('conn').textContent=l.connection_failures;
  $('connsub').textContent=l.connection_failures?'see the message above':'none';
  $('clock').textContent=s.audio_seconds.toFixed(0)+' s';
  $('clocksub').textContent=`${l.delivered} frames played`;
}
async function tick(){
  try{const r=await fetch('/state',{cache:'no-store'});render(await r.json());setTimeout(tick,100);}
  catch(e){status('crit','✕ Not connected: the pipeline is not running');setTimeout(tick,700);}
}
tick();
</script></body></html>
'''


if __name__ == '__main__':
    main()
