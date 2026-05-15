# pe-lsdj streaming

Real-time generation of LSDJ song data using a JAX transformer, streamed live
into a running PyBoy emulator instance.

## How it works

1. **Prompt** — a `.lsdsng` file is loaded and tokenized. The first N steps
   (default: 64) are fed into the model as a prompt via a KV-cache prefill.
2. **Generation** — the model autoregressively generates new song steps.
   Each step covers all four LSDJ channels (PU1, PU2, WAV, NOI).
3. **Streaming** — generated steps are written into LSDJ's SRAM ahead of the
   playhead. The system tracks the playhead position and keeps a configurable
   number of phrases buffered ahead of it.
4. **Playback** — LSDJ plays the written data in real time via PyBoy, producing
   audio normally.

The generator runs on a dedicated thread so JAX computation and PyBoy ticking
never block each other.

## Entry point

```
python3 scripts/lsdj_stream.py \
    --rom   lsdj.gb \
    --sav   songs.sav \
    --song  path/to/prompt.lsdsng \
    --weights path/to/weights.eqx \
    [--headless] [--web-port 8765] [--record session.pelsdj]
```

Multiple weight files can be passed to `--weights` to enable live crossfade
between models (see below).

## Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--write-ahead-phrases` | 2 | Phrases to keep buffered ahead of the playhead |
| `--prompt-steps` | 64 | Steps of the prompt song used for KV-cache prefill |
| `--song-length` | prompt file length | Song-length hint for the progress embedding |
| `--temp` | 0.9 | Initial sampling temperature |
| `--lock-progress` | — | Pin progress fraction (e.g. `0.4` = always ~40% through) |
| `--exclude-fx` | — | Hard-exclude FX commands, e.g. `H,M,T` (post-temperature, so effect is T-invariant) |
| `--channel-mask` | — | Freeze channels from generation (e.g. `2,3` = WAV, NOI) |
| `--seed` | 42 | RNG seed |
| `--headless` | off | Run without SDL2 window (window is on by default; required for audio) |
| `--web-port` | — | Enable browser UI on this port (e.g. `8765`) |
| `--record` | — | Save session to a `.pelsdj` file for later replay |

## Live controls

Controls are available in the terminal and mirrored exactly in the browser UI
(see Web UI below).

| Key | Action |
|-----|--------|
| `]` / `[` | Temperature ±0.25 |
| `p` | Toggle progress lock (freeze/unfreeze song position) |
| `>` / `<` | Nudge locked progress ±5% |
| `}` / `{` | Crossfade between models ±0.1 (when multiple `--weights` given) |
| `s` | Save recording snapshot to disk (when `--record` is active) |

## Song progress

The model receives a progress signal (`position / song_length`) at each step,
which influences the character of generation. Two modes:

- **Advancing** — progress moves forward naturally as steps are generated.
- **Locked** — progress is pinned to a fixed fraction, producing continuous
  generation at a consistent point in the song arc. Toggle with `p`;
  adjust with `<` / `>`.

Unlocking resumes from the fraction at the time of locking, not from wherever
the absolute step count has reached.

## Crossfade

Passing multiple files to `--weights` loads all of them and enables live
crossfade between the weight sets:

```
--weights weights_a.eqx weights_b.eqx
```

`{` / `}` interpolate piecewise-linearly along the list (`0.0` = first model,
`1.0` = second, etc.). For more than two models the position moves between
adjacent pairs. Interpolation runs on a background thread so generation is not
interrupted.

The KV cache is built from the first model (`xfade = 0`). Shifting xfade
mid-session introduces a mild inconsistency between cached and new context,
which in practice sounds like a gradual style shift.

## Recording

Passing `--record session.pelsdj` records the full token stream and all control
events to a compressed NumPy archive (`.pelsdj.npz`). The file contains:

- `tokens` — `(N, 4, 21) uint16` array of every generated step
- `config` — JSON metadata (weights, song, seed, thresholds, etc.)
- `events` — JSON log of every live control change (temp, progress, xfade)
  with the step index at which it occurred

Press `s` at any time to save a snapshot without stopping the session. On
Ctrl-C, the recording is saved automatically (SIGINT is briefly suppressed
during the write to prevent truncation).

## Replay

A recorded session can be played back without the model:

```
python3 scripts/lsdj_replay.py session.pelsdj \
    --rom lsdj.gb \
    --sav songs.sav \
    [--window]
```

This feeds the recorded token stream directly into LSDJ's SRAM — no JAX
required. Control events are printed inline as playback reaches the step at
which they originally occurred (for reference only; they have no effect on the
token stream, which already encodes their influence).

## Web UI

Passing `--web-port 8765` launches a browser-based control panel alongside the
terminal. Open `http://localhost:8765` after startup.

The UI provides:
- **Live LSDJ screen** — MJPEG stream of the PyBoy framebuffer (requires
  `--window`; audio also comes from PyBoy)
- **Generation controls** — temperature, crossfade, progress lock/nudge,
  with real-time state display via SSE
- **REC indicator** — shows when a recording is active
- **Keyboard shortcuts** — identical to the terminal, plus Game Boy controls:

| Key | Game Boy button |
|-----|----------------|
| `↑` `↓` `←` `→` | D-pad |
| `z` | A |
| `x` | B |
| `↵` | Start |
| `⌫` | Select |

## Module layout

| File | Role |
|------|------|
| `session.py` | `StreamingSession` — main orchestration, threading, live controls |
| `buffer.py` | `StreamingBuffer` — SRAM write-ahead ring buffer |
| `alloc.py` | `AllocationManager` — phrase/chain allocation tracking |
| `sram.py` | Low-level SRAM read/write helpers |
| `webapp.py` | `StreamingWebApp` — FastAPI browser UI (MJPEG, SSE, controls) |
