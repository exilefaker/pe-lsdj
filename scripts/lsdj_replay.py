#!/usr/bin/env python3
"""
lsdj_replay.py: Replay a recorded pe-lsdj session from a .pelsdj file.

Feeds the recorded token stream directly into LSDJ SRAM — no model or JAX
required, exact reproduction of the original session.

The .pelsdj file also contains the original control events (temp, xfade,
progress changes) as a JSON log, printed on load for reference.

Usage:
    python3 scripts/lsdj_replay.py session.pelsdj \\
        --rom  lsdj.gb \\
        --sav  songs.sav \\
        [--window]
"""

import argparse
import json
import os
import queue as stdlib_queue
import threading
import time

import numpy as np
from pyboy import PyBoy

from pe_lsdj.constants import NUM_CHANNELS
from pe_lsdj.streaming import AllocationManager, StreamingBuffer
from pe_lsdj.streaming.session import find_first_empty_row, read_playheads

_INIT_FRAMES    = 180
_PHRASE_CURSOR  = 0xC74B
_FRAME_DURATION = 1.0 / 60.0


def _feed_tokens(tokens, queue: stdlib_queue.Queue, stop: threading.Event) -> None:
    """Thread body: push token arrays into the queue, then send a None sentinel."""
    for tok in tokens:
        if stop.is_set():
            return
        while not stop.is_set():
            try:
                queue.put(tok, timeout=0.05)
                break
            except stdlib_queue.Full:
                continue
    queue.put(None)  # end-of-stream sentinel


def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded pe-lsdj session from a .pelsdj file."
    )
    parser.add_argument("recording",
                        help=".pelsdj (or .pelsdj.npz) file to replay.")
    parser.add_argument("--rom", "-r", required=True, help="LSDJ .gb ROM path.")
    parser.add_argument("--sav", "-s", required=True, help=".sav file to boot from.")
    parser.add_argument("--write-ahead-phrases", type=int, default=2,
                        help="Phrases to keep buffered ahead of the playhead (default: 2).")
    parser.add_argument("--window", action="store_true",
                        help="Show SDL2 window (default: headless null).")
    args = parser.parse_args()

    # ── load recording ────────────────────────────────────────────────────────
    path = args.recording
    if not os.path.exists(path) and not path.endswith(".npz"):
        path = path + ".npz"
    data   = np.load(path, allow_pickle=False)
    tokens = data["tokens"]                              # (N, 4, 21) uint16
    config = json.loads(bytes(data["config"]).decode())
    events = json.loads(bytes(data["events"]).decode())

    print(f"Recording : {path}")
    print(f"Steps     : {len(tokens)}")
    print(f"Song      : {config.get('song', 'unknown')}")
    print(f"Weights   : {config.get('weights', 'unknown')}")
    print(f"Seed      : {config.get('seed', '?')}  "
          f"temp={config.get('initial_temp', '?')}  "
          f"song_length={config.get('song_length', '?')}")
    if events:
        print(f"Events    : {len(events)}")
        for ev in events:
            print(f"  step {ev['step']:>6}  {ev['type']} → {ev['value']}")
    print()

    num_phrases_per_chain = config.get("num_phrases_per_chain", 4)

    # ── parse channel mask from config ────────────────────────────────────────
    channel_mask = [False] * NUM_CHANNELS
    for tok in config.get("channel_mask", "").split(","):
        tok = tok.strip()
        if tok:
            channel_mask[int(tok)] = True

    # ── boot PyBoy ────────────────────────────────────────────────────────────
    window_mode = "SDL2" if args.window else "null"
    print(f"Booting LSDJ ({window_mode}, {_INIT_FRAMES} init frames) ...")
    with open(args.sav, "rb") as sav_fh:
        pyboy = PyBoy(args.rom, window=window_mode, ram_file=sav_fh)
    for _ in range(_INIT_FRAMES):
        pyboy.tick(render=args.window)

    # ── initialise streaming objects ──────────────────────────────────────────
    alloc = AllocationManager(pyboy)
    print(f"Alloc: {alloc.free_phrase_count}/255 phrases, "
          f"{alloc.free_chain_count}/128 chains free")

    phrases_per_row   = NUM_CHANNELS * num_phrases_per_chain
    chains_per_row    = NUM_CHANNELS
    safe_from_phrases = alloc.free_phrase_count // phrases_per_row - 1
    safe_from_chains  = alloc.free_chain_count  // chains_per_row  - 1
    max_rows          = max(1, min(safe_from_phrases, safe_from_chains))

    first_row = find_first_empty_row(pyboy)
    print(f"Streaming from song row 0x{first_row:02X} ({first_row}), max_rows={max_rows}")

    buf = StreamingBuffer(
        pyboy, alloc,
        next_song_row         = first_row,
        max_rows              = max_rows,
        num_phrases_per_chain = num_phrases_per_chain,
        channel_mask          = channel_mask,
    )

    # ── pre-fill buffer ───────────────────────────────────────────────────────
    npp = num_phrases_per_chain
    write_ahead_rows = 1 + (args.write_ahead_phrases + npp - 1) // npp
    pregen_steps     = min(write_ahead_rows * buf.steps_per_row, len(tokens))
    print(f"Pre-filling {write_ahead_rows} row(s) ({pregen_steps} steps) ...")

    rows_committed = buf.committed_rows
    token_idx      = 0
    for _ in range(pregen_steps):
        committed = buf.push_step(tokens[token_idx])
        token_idx += 1
        if committed is not None:
            rows_committed += 1
            print(f"  pre-fill row 0x{committed.song_row:02X}")

    # ── start feeder thread for remaining tokens ──────────────────────────────
    queue_depth  = args.write_ahead_phrases * 16 * 2
    token_queue  = stdlib_queue.Queue(maxsize=queue_depth)
    stop_event   = threading.Event()
    feeder       = threading.Thread(
        target=_feed_tokens,
        args=(tokens[token_idx:], token_queue, stop_event),
        daemon=True,
    )
    feeder.start()

    tokens_consumed = token_idx  # total tokens pushed to SRAM so far
    event_ptr = 0
    while event_ptr < len(events) and events[event_ptr]["step"] < tokens_consumed:
        event_ptr += 1  # skip events that occurred before pre-fill

    # ── main loop ─────────────────────────────────────────────────────────────
    print(f"\nStarting LSDJ playback — {len(tokens)} steps total.  Ctrl-C to stop.\n")
    pyboy.button("start")
    pyboy.tick(render=args.window)

    prev_cursor       = pyboy.memory[_PHRASE_CURSOR]
    cursor_total      = 0
    playheads         = read_playheads(pyboy)
    initial_min_ph    = min(playheads)
    prev_min_ph       = initial_min_ph
    cursor_at_row_adv = 0
    _c74b_samples: list[float] = []
    _c74b_per_row: float | None = None
    stream_done = False

    try:
        while True:
            frame_start = time.perf_counter()
            pyboy.tick(render=args.window)

            curr  = pyboy.memory[_PHRASE_CURSOR]
            delta = (curr - prev_cursor) % 256
            if delta:
                cursor_total += delta
                prev_cursor   = curr

            playheads  = read_playheads(pyboy)
            cur_min_ph = min(playheads)
            row_delta  = (cur_min_ph - prev_min_ph) % 256
            if row_delta > 0:
                ticks_this = cursor_total - cursor_at_row_adv
                if ticks_this > 0:
                    _c74b_samples.append(ticks_this / row_delta)
                    _c74b_per_row = sum(_c74b_samples[-8:]) / len(_c74b_samples[-8:])
                cursor_at_row_adv = cursor_total
                prev_min_ph       = cur_min_ph
                c74b_str = f"{_c74b_per_row:.1f}" if _c74b_per_row else "?"
                pct = min(100, round(100 * tokens_consumed / len(tokens)))
                print(
                    f"[row+{row_delta}] ph={cur_min_ph:#04x}  "
                    f"c74b/row={c74b_str}  "
                    f"rows_in_buf={buf.committed_rows}  "
                    f"replayed≈{pct}%"
                )

            rows_advanced = (cur_min_ph - initial_min_ph) % 256
            intra_frac    = 0.0
            if _c74b_per_row:
                intra_frac = min((cursor_total - cursor_at_row_adv) / _c74b_per_row, 1.0)
            phrases_consumed = (rows_advanced + intra_frac) * buf.num_phrases_per_chain
            phrases_ahead    = buf.phrases_committed - phrases_consumed

            while phrases_ahead < args.write_ahead_phrases and not stream_done:
                try:
                    tok = token_queue.get_nowait()
                except stdlib_queue.Empty:
                    break
                if tok is None:
                    stream_done = True
                    print("Token stream exhausted — playing out remaining buffer.")
                    break
                buf.push_step(tok)
                tokens_consumed += 1
                while event_ptr < len(events) and events[event_ptr]["step"] <= tokens_consumed:
                    ev = events[event_ptr]
                    print(f"  [step {ev['step']:>6}] {ev['type']} → {ev['value']}", flush=True)
                    event_ptr += 1
                phrases_ahead = buf.phrases_committed - phrases_consumed

            if not args.window:
                elapsed   = time.perf_counter() - frame_start
                remainder = _FRAME_DURATION - elapsed
                if remainder > 0:
                    time.sleep(remainder)

    except KeyboardInterrupt:
        print()
    finally:
        stop_event.set()
        feeder.join(timeout=1.0)
        print(f"Stopped after {buf.committed_rows} rows ({tokens_consumed}/{len(tokens)} steps).")
        pyboy.stop(save=False)


if __name__ == "__main__":
    main()
