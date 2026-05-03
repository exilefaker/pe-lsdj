#!/usr/bin/env python3
"""
lsdj_stream.py: Real-time model-driven LSDJ generation via PyBoy.

Boots LSDJ in PyBoy, prefills the KV cache from a .lsdsng prompt,
then streams generated song steps into SRAM in sync with the playhead.

The model generates one step at a time; steps accumulate into phrases
(16 steps), then chains (num_phrases_per_chain phrases), then song rows.
StreamingBuffer handles SRAM allocation and write timing. When the
buffer is full (max_rows), the oldest row is recycled.

Architecture: a background thread runs _jit_step continuously and puts
numpy tokens into a queue. The main thread ticks PyBoy at a steady 60 fps
and drains tokens from the queue into StreamingBuffer as needed. JAX
releases the Python GIL during C++ computation, so both threads make
progress concurrently without audio stalls.

Usage:
    python3 scripts/lsdj_stream.py \\
        --rom   lsdj.gb \\
        --sav   songs.sav \\
        --song  data/gen_test/my_song.lsdsng \\
        --weights data/weights/v13/step_011950.eqx \\
        [--write-ahead 8] [--temp 0.9] [--window]
"""

import argparse
import json
import os
import queue as stdlib_queue
import threading
import time

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
from pyboy import PyBoy

from pe_lsdj import SongFile
from pe_lsdj.embedding import SongBanks
from pe_lsdj.generation import generate_step_cached
from pe_lsdj.models import LSDJTransformer
from pe_lsdj.constants import (
    SONG_CHAINS_ADDR,
    NUM_CHANNELS,
    EMPTY,
)
from pe_lsdj.streaming import AllocationManager, StreamingBuffer
from pe_lsdj.streaming.sram import read_sram

# Per-channel WRAM playhead addresses (0=PU1, 1=PU2, 2=WAV, 3=NOI)
_PLAYHEAD_ADDRS = [0xC39F, 0xC3A0, 0xC3A1, 0xC3A2]

_INIT_FRAMES    = 180          # ~3 s at 60 fps; lets LSDJ finish initialisation
_FRAME_DURATION = 1.0 / 60.0  # null-mode rate cap; SDL2 mode self-limits via display


# ── helpers ───────────────────────────────────────────────────────────────────

def find_first_empty_row(pyboy) -> int:
    """Return the first song row where all 4 channel slots are EMPTY (0xFF)."""
    for row in range(256):
        if all(
            read_sram(pyboy, SONG_CHAINS_ADDR.start + row * NUM_CHANNELS + ch) == EMPTY
            for ch in range(NUM_CHANNELS)
        ):
            return row
    return 0  # fully packed; wrap and overwrite from row 0


def read_playheads(pyboy) -> list[int]:
    """Return current song-row playhead for each channel (0–255)."""
    return [pyboy.memory[addr] for addr in _PLAYHEAD_ADDRS]


_jit_step = eqx.filter_jit(generate_step_cached)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stream model-generated steps live into LSDJ via PyBoy."
    )
    parser.add_argument("--rom",     "-r", required=True, help="LSDJ .gb ROM path")
    parser.add_argument("--sav",     "-s", required=True, help=".sav file to boot from")
    parser.add_argument("--song",    "-g", required=True, help=".lsdsng prompt file")
    parser.add_argument("--weights", "-w", required=True, help="Model weights (.eqx)")
    parser.add_argument("--params",  "-p", default=None,  help="model_hyperparams.json (default: weights dir)")

    parser.add_argument("--write-ahead", type=int, default=8,
                        help="Target rows ahead of slowest playhead (default: 8)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Rolling buffer depth (default: auto from alloc headroom)")
    parser.add_argument("--num-phrases-per-chain", type=int, default=4,
                        help="Phrases per chain per row (default: 4 → 64 steps/row)")
    parser.add_argument("--prompt-steps", type=int, default=64,
                        help="Steps of the .lsdsng to use as prompt (default: 64)")
    parser.add_argument("--song-length", type=int, default=None,
                        help="Song-length hint for progress embedding (default: prompt + 1024)")
    parser.add_argument("--temp", type=float, default=0.9,
                        help="Sampling temperature (default: 0.9)")
    parser.add_argument("--instr-threshold",     type=float, default=0.5)
    parser.add_argument("--table-threshold",     type=float, default=0.5)
    parser.add_argument("--groove-threshold",    type=float, default=0.1)
    parser.add_argument("--softsynth-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", action="store_true",
                        help="Show SDL2 window (default: headless null)")
    parser.add_argument("--channel-mask", type=str, default="",
                        help="Comma-separated channel indices to freeze, e.g. '2,3'")
    args = parser.parse_args()

    # ── channel mask ──────────────────────────────────────────────────────────
    channel_mask = [False] * NUM_CHANNELS
    for tok in args.channel_mask.split(","):
        tok = tok.strip()
        if tok:
            channel_mask[int(tok)] = True
    if any(channel_mask):
        ch_names = ["PU1", "PU2", "WAV", "NOI"]
        frozen = [ch_names[i] for i, m in enumerate(channel_mask) if m]
        print(f"Frozen channels: {frozen}")

    # ── load model ────────────────────────────────────────────────────────────
    params_path = args.params or os.path.join(os.path.dirname(args.weights), "model_hyperparams.json")
    print(f"Loading model from {args.weights} ...")
    with open(params_path) as f:
        params = json.load(f)
    key = jr.PRNGKey(args.seed)
    model_key, gen_key = jr.split(key)
    ref   = LSDJTransformer(model_key, **params)
    model = eqx.tree_deserialise_leaves(args.weights, like=ref)
    model = eqx.nn.inference_mode(model)
    print("Model loaded.")

    # ── load prompt and prefill KV cache ─────────────────────────────────────
    sf     = SongFile(args.song)
    banks  = SongBanks.from_songfile(sf)
    tokens = jnp.asarray(sf.song_tokens[:args.prompt_steps], dtype=jnp.uint16)
    W      = tokens.shape[0]
    song_length = args.song_length or (W + 1024)
    print(f"Prompt: {W} steps from {os.path.basename(args.song)}  |  song_length={song_length}")

    print("Pre-filling KV cache ...")
    last_hidden, k_cache, v_cache = eqx.filter_jit(model.prefill)(
        tokens, banks, song_length=song_length
    )
    print("KV cache ready.")

    # ── boot PyBoy ────────────────────────────────────────────────────────────
    window_mode = "SDL2" if args.window else "null"
    print(f"Booting LSDJ ({window_mode} window, {_INIT_FRAMES} init frames) ...")
    with open(args.sav, "rb") as sav_fh:
        pyboy = PyBoy(args.rom, window=window_mode, ram_file=sav_fh)

    for _ in range(_INIT_FRAMES):
        pyboy.tick(render=args.window)

    # ── initialise streaming objects ──────────────────────────────────────────
    alloc = AllocationManager(pyboy)
    print(f"Alloc: {alloc.free_phrase_count}/{255} phrases free, "
          f"{alloc.free_chain_count}/128 chains free")

    # Compute safe max_rows from available headroom.
    # Peak allocation = (max_rows + 1) rows (buffer + one building chain).
    phrases_per_row = NUM_CHANNELS * args.num_phrases_per_chain
    chains_per_row  = NUM_CHANNELS
    safe_from_phrases = alloc.free_phrase_count // phrases_per_row - 1
    safe_from_chains  = alloc.free_chain_count  // chains_per_row  - 1
    safe_max_rows = max(1, min(safe_from_phrases, safe_from_chains))

    if args.max_rows is None:
        max_rows = safe_max_rows
        print(f"Auto max_rows={max_rows} (headroom: "
              f"{alloc.free_phrase_count} phrases, {alloc.free_chain_count} chains)")
    else:
        if args.max_rows > safe_max_rows:
            print(f"Warning: --max-rows {args.max_rows} exceeds alloc headroom "
                  f"({safe_max_rows} safe); clamping.")
        max_rows = min(args.max_rows, safe_max_rows)

    if args.write_ahead >= max_rows:
        print(f"Warning: write_ahead={args.write_ahead} >= max_rows={max_rows}; "
              f"clamping write_ahead to {max(1, max_rows - 1)}")
        write_ahead = max(1, max_rows - 1)
    else:
        write_ahead = args.write_ahead

    first_row = find_first_empty_row(pyboy)
    print(f"Streaming from song row 0x{first_row:02X} ({first_row})")

    buf = StreamingBuffer(
        pyboy, alloc,
        next_song_row         = first_row,
        max_rows              = max_rows,
        num_phrases_per_chain = args.num_phrases_per_chain,
        channel_mask          = channel_mask,
    )

    # ── pre-generate write_ahead rows before starting playback ───────────────
    gen_key  = jr.PRNGKey(args.seed + 1)
    step_idx = 0

    print(f"Pre-generating {write_ahead} rows ({write_ahead * buf.steps_per_row} steps) ...")

    while buf.committed_rows < write_ahead:
        gen_key, step_key = jr.split(gen_key)
        carry = (last_hidden, banks, k_cache, v_cache)
        carry, next_token = _jit_step(
            carry, (step_key, jnp.int32(step_idx)),
            model, W, song_length,
            args.instr_threshold, args.table_threshold,
            args.groove_threshold, args.softsynth_threshold,
            args.temp,
        )
        last_hidden, banks, k_cache, v_cache = carry
        committed = buf.push_step(np.array(next_token))
        if committed is not None:
            print(f"  pre-gen row 0x{committed.song_row:02X}")
        step_idx += 1

    # ── launch generator thread ───────────────────────────────────────────────
    # The generator thread owns all JAX state. The main thread owns PyBoy and
    # StreamingBuffer. JAX releases the GIL during C++ computation, so both
    # threads run concurrently — PyBoy ticks at steady 60 fps while the model
    # generates in the background.
    #
    # Queue maxsize bounds look-ahead memory: 2× write_ahead rows of tokens.
    _step_queue = stdlib_queue.Queue(maxsize=write_ahead * buf.steps_per_row * 2)
    _stop_gen   = threading.Event()

    def _generator_loop():
        nonlocal gen_key, last_hidden, banks, k_cache, v_cache, step_idx
        while not _stop_gen.is_set():
            gen_key, step_key = jr.split(gen_key)
            carry = (last_hidden, banks, k_cache, v_cache)
            carry, next_token = _jit_step(
                carry, (step_key, jnp.int32(step_idx)),
                model, W, song_length,
                args.instr_threshold, args.table_threshold,
                args.groove_threshold, args.softsynth_threshold,
                args.temp,
            )
            last_hidden, banks, k_cache, v_cache = carry
            step_idx += 1
            token_np = np.array(next_token)
            # Block until space is available — natural backpressure.
            while not _stop_gen.is_set():
                try:
                    _step_queue.put(token_np, timeout=0.05)
                    break
                except stdlib_queue.Full:
                    continue

    gen_thread = threading.Thread(target=_generator_loop, daemon=True, name="pe-lsdj-gen")
    gen_thread.start()
    print("Generator thread started.")

    # ── start playback ────────────────────────────────────────────────────────
    print("\nStarting LSDJ playback — Ctrl-C to stop.\n")
    pyboy.button("start")
    pyboy.tick(render=args.window)

    rows_committed = buf.committed_rows
    log_every      = max(1, write_ahead // 2)

    try:
        while True:
            frame_start = time.perf_counter()

            # Tick the emulator one frame. No JAX here — runs at steady 60 fps.
            pyboy.tick(render=args.window)

            playheads = read_playheads(pyboy)
            ahead     = buf.rows_ahead_of(playheads)

            # Drain as many pre-generated tokens as needed to stay write_ahead rows
            # ahead of the playhead. Each token takes only a few µs (SRAM write).
            while ahead < write_ahead:
                try:
                    next_token = _step_queue.get_nowait()
                except stdlib_queue.Empty:
                    break  # generator hasn't caught up; try again next frame
                committed = buf.push_step(next_token)
                if committed is not None:
                    rows_committed += 1
                    if rows_committed % log_every == 0:
                        ph_str = " ".join(f"{p:02X}" for p in playheads)
                        print(f"row 0x{committed.song_row:02X} committed  "
                              f"total={rows_committed}  "
                              f"playhead=[{ph_str}]  "
                              f"ahead={buf.rows_ahead_of(playheads)}  "
                              f"free={alloc.free_phrase_count}ph/{alloc.free_chain_count}ch")
                ahead = buf.rows_ahead_of(playheads)

            # In null mode, cap to 60 fps so PyBoy doesn't flood the audio buffer.
            # SDL2 mode self-limits via display refresh; no sleep needed there.
            if not args.window:
                elapsed   = time.perf_counter() - frame_start
                remainder = _FRAME_DURATION - elapsed
                if remainder > 0:
                    time.sleep(remainder)

    except KeyboardInterrupt:
        print()
    finally:
        _stop_gen.set()
        gen_thread.join(timeout=2.0)
        print(f"Stopped after {rows_committed} rows ({step_idx} steps).")
        pyboy.stop(save=False)


if __name__ == "__main__":
    main()
