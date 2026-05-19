"""
webapp.py — lightweight FastAPI web UI for live LSDJ generation control.

Serves:
  GET  /        HTML control panel with embedded Game Boy screen
  GET  /video   MJPEG stream of the PyBoy framebuffer (30 fps)
  GET  /events  SSE stream of current session state (10 Hz)
  POST /control JSON {action, value} — same actions as keyboard controls

Integration (once session.py wiring is added):
  webapp = StreamingWebApp(session, pyboy, port=8765)
  webapp.start()                      # launches uvicorn in a daemon thread
  # call webapp.update_frame() after each pyboy.tick() in the main loop

Dependencies: pip install fastapi uvicorn[standard]
"""

import asyncio
import io
import json
import threading

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from .session import MIN_TEMP, MAX_TEMP, MIN_XFADE, _crossfade_models


# ── embedded HTML / JS ────────────────────────────────────────────────────────

_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>pe-lsdj</title>
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #cfc; font-family: monospace; font-size: 14px; }
.layout { display: flex; gap: 16px; padding: 16px; align-items: flex-start; flex-wrap: wrap; }
.screen img { display: block; width: 480px; height: 432px;
              image-rendering: pixelated; border: 2px solid #484; background: #000; }
.controls { display: flex; flex-direction: column; gap: 10px; min-width: 220px; }
section { background: #1a1a1a; border: 1px solid #333; border-radius: 4px; padding: 10px; }
h3 { color: #8d8; margin-bottom: 8px; font-size: 11px; letter-spacing: 0.08em; }
.val { color: #fff; font-size: 20px; font-weight: bold; display: inline-block; min-width: 64px; }
.sub { color: #666; font-size: 11px; margin-left: 4px; }
.btn-row { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
button { background: #1e2e1e; color: #cfc; border: 1px solid #4a4; border-radius: 3px;
         padding: 5px 10px; cursor: pointer; font-family: monospace; font-size: 12px; }
button:hover { background: #2a3e2a; }
button.on { background: #3a6a3a; color: #fff; border-color: #6d6; }
input[type=range] { width: 100%; margin-top: 8px; accent-color: #4a4; cursor: pointer; }
.status { font-size: 11px; color: #555; padding: 6px 10px; }
.rec { color: #f55; }
.keys { border: 1px solid #252; border-radius: 4px; padding: 10px;
        font-size: 11px; color: #555; line-height: 1.9; }
.keys b { color: #7b7; }
</style>
</head>
<body>
<div class="layout">

  <div class="screen">
    <img src="/video" alt="LSDJ">
  </div>

  <div class="controls" x-data="app()">

    <section>
      <h3>TEMPERATURE</h3>
      <span class="val" x-text="tempVal.toFixed(2)">—</span>
      <input type="range" min="0" max="4" step="0.05"
             :value="tempVal"
             @mousedown="tempDrag = true"
             @input="tempVal = +$event.target.value"
             @change="tempDrag = false; ctl('temp_set', tempVal)"
             @touchstart="tempDrag = true"
             @touchend="tempDrag = false; ctl('temp_set', tempVal)">
      <div class="btn-row">
        <button @click="ctl('temp_nudge',-0.25)" title="[ key">[ &minus;0.25</button>
        <button @click="ctl('temp_nudge',+0.25)" title="] key">] +0.25</button>
      </div>
    </section>

    <section x-show="s.xfade_enabled">
      <h3>CROSSFADE</h3>
      <span class="val" x-text="xfadeVal.toFixed(2)">—</span>
      <span class="sub" x-text="'/ ' + s.xfade_max"></span>
      <input type="range" min="0" :max="s.xfade_max" step="0.1"
             :value="xfadeVal"
             @mousedown="xfadeDrag = true"
             @change="xfadeVal = +$event.target.value; xfadeDrag = false; ctl('xfade_set', xfadeVal)"
             @touchstart="xfadeDrag = true"
             @touchend="xfadeVal = +$event.target.value; xfadeDrag = false; ctl('xfade_set', xfadeVal)">
      <div class="btn-row">
        <button @click="ctl('xfade_nudge',-0.1)" title="{ key">{ &minus;0.1</button>
        <button @click="ctl('xfade_nudge',+0.1)" title="} key">} +0.1</button>
      </div>
    </section>

    <section>
      <h3>PROGRESS</h3>
      <span class="val" x-text="pct()">—</span>
      <span class="sub" x-text="s.progress_locked ? '(locked)' : '(advancing)'"></span>
      <div class="btn-row">
        <button @click="ctl('progress_toggle')" :class="s.progress_locked ? 'on' : ''"
                title="p key">p lock</button>
        <button @click="ctl('progress_nudge',-0.05)" title="< key">&lt; &minus;5%</button>
        <button @click="ctl('progress_nudge',+0.05)" title="> key">&gt; +5%</button>
      </div>
    </section>

    <div class="status">
      step <b x-text="s.step_idx">0</b>
      &nbsp;&nbsp;
      <span class="rec" x-show="s.recording">&#9679; REC</span>
    </div>

    <div class="keys">
      <b>generation controls</b><br>
      <b>[ ]</b> temperature &nbsp; <b>{ }</b> crossfade<br>
      <b>&lt; &gt;</b> nudge progress &nbsp; <b>p</b> lock/unlock<br>
      <b>game boy controls</b><br>
      <b>↑↓←→</b> D-pad &nbsp; <b>z</b> A &nbsp; <b>x</b> B<br>
      <b>↵</b> start &nbsp; <b>⌫</b> select
    </div>

  </div>
</div>

<script>
function app() {
  return {
    s: {
      temp: 0.9, xfade: 0.0, xfade_max: 0, xfade_enabled: false,
      progress: 0.0, progress_locked: false, step_idx: 0, recording: false,
    },
    tempVal: 0.9, xfadeVal: 0.0,
    tempDrag: false, xfadeDrag: false,
    init() {
      const es = new EventSource('/events');
      es.onmessage = e => {
        this.s = JSON.parse(e.data);
        if (!this.tempDrag)  this.tempVal  = this.s.temp;
        if (!this.xfadeDrag) this.xfadeVal = this.s.xfade;
      };
      document.addEventListener('keydown', e => this.onkey(e, true));
      document.addEventListener('keyup',   e => this.onkey(e, false));
    },
    pct() { return (this.s.progress * 100).toFixed(0) + '%'; },
    ctl(action, value = null) {
      fetch('/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, value }),
      });
    },
    gp(btn, down) { this.ctl(down ? 'gamepad_press' : 'gamepad_release', btn); },
    onkey(e, down) {
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
      const map = {
        ']': ['temp_nudge',     +0.25],
        '[': ['temp_nudge',     -0.25],
        '}': ['xfade_nudge',    +0.1 ],
        '{': ['xfade_nudge',    -0.1 ],
        '>': ['progress_nudge', +0.05],
        '<': ['progress_nudge', -0.05],
        'p': ['progress_toggle', null],
      };
      const gpmap = {
        'ArrowUp': 'up', 'ArrowDown': 'down', 'ArrowLeft': 'left', 'ArrowRight': 'right',
        'z': 'a', 'Z': 'a', 'x': 'b', 'X': 'b',
        'Enter': 'start', 'Backspace': 'select', 'Delete': 'select',
      };
      if (down && map[e.key]) { e.preventDefault(); this.ctl(...map[e.key]); }
      if (gpmap[e.key] && !e.repeat) { e.preventDefault(); this.gp(gpmap[e.key], down); }
    },
  };
}
</script>
</body>
</html>
"""


# ── FastAPI request model ──────────────────────────────────────────────────────

class ControlRequest(BaseModel):
    action: str
    value: str | float | None = None


# ── main class ────────────────────────────────────────────────────────────────

class StreamingWebApp:
    """
    Lightweight web UI for monitoring and controlling a StreamingSession.

    The Game Boy screen is streamed as MJPEG; session state is pushed via SSE;
    controls are plain JSON POSTs — keyboard shortcuts in the browser mirror
    the terminal controls exactly.
    """

    def __init__(self, session, pyboy, port: int = 8765):
        self._session    = session
        self._pyboy      = pyboy
        self._port       = port
        self._frame_lock = threading.Lock()
        self._latest_frame: bytes = b""
        self._app = self._build_app()

    def update_frame(self) -> None:
        """Capture the current PyBoy screen. Call after each pyboy.tick()."""
        try:
            buf = io.BytesIO()
            self._pyboy.screen.image.convert("RGB").save(buf, format="JPEG", quality=80)
            with self._frame_lock:
                self._latest_frame = buf.getvalue()
        except Exception:
            pass

    def start(self) -> None:
        """Launch uvicorn in a daemon thread; returns immediately."""
        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # keep signals on main thread
        threading.Thread(target=server.run, daemon=True, name="pe-lsdj-web").start()
        print(f"Web UI → http://localhost:{self._port}")

    # ── routes ────────────────────────────────────────────────────────────────

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return _HTML

        @app.get("/video")
        async def video():
            return StreamingResponse(
                self._mjpeg_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @app.get("/events")
        async def events():
            return StreamingResponse(
                self._sse_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @app.post("/control")
        async def control(req: ControlRequest):
            self._dispatch(req.action, req.value)
            return {"ok": True}

        return app

    # ── streaming generators ──────────────────────────────────────────────────

    async def _mjpeg_stream(self):
        while True:
            with self._frame_lock:
                frame = self._latest_frame
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(1 / 30)

    async def _sse_stream(self):
        while True:
            yield f"data: {json.dumps(self._state())}\n\n"
            await asyncio.sleep(0.1)

    # ── state snapshot ────────────────────────────────────────────────────────

    def _state(self) -> dict:
        s = self._session
        n = len(s._models)
        return {
            "temp":           round(s._temp, 4),
            "xfade":          round(s._xfade, 4),
            "xfade_max":      n - 1,
            "xfade_enabled":  n > 1,
            "progress":       round(
                s._loop_progress if s._loop_progress is not None
                else min(1.0, (s._W + s._progress_step_idx) / s._song_length),
                4,
            ),
            "progress_locked": s._loop_progress is not None,
            "step_idx":       s._step_idx,
            "recording":      s._record_path is not None,
        }

    # ── control dispatch ──────────────────────────────────────────────────────

    def _dispatch(self, action: str, value) -> None:
        s = self._session
        if action == "temp_nudge":
            s._nudge_temp(float(value))
        elif action == "temp_set":
            s._temp = max(MIN_TEMP, min(MAX_TEMP, float(value)))
            print(f"[temp → {s._temp:.2f}]", flush=True)
            s._record_event("temp", s._temp)
        elif action == "xfade_nudge":
            s._nudge_xfade(float(value))
        elif action == "xfade_set":
            v = max(MIN_XFADE, min(float(len(s._models) - 1), float(value)))
            s._xfade = v
            print(f"[xfade → {s._xfade:.2f}]", flush=True)
            s._record_event("xfade", s._xfade)
            threading.Thread(
                target=lambda: setattr(s, "_model", _crossfade_models(s._models, v)),
                daemon=True,
            ).start()
        elif action == "progress_nudge":
            s._nudge_progress(float(value))
        elif action == "progress_set":
            v = max(0.05, min(1.0, float(value)))
            s._loop_progress = v
            s._record_event("progress", v)
        elif action == "progress_toggle":
            s._toggle_progress_lock()
        elif action == "gamepad_press":
            if value in {"up", "down", "left", "right", "a", "b", "start", "select"}:
                self._pyboy.button_press(str(value))
        elif action == "gamepad_release":
            if value in {"up", "down", "left", "right", "a", "b", "start", "select"}:
                self._pyboy.button_release(str(value))
