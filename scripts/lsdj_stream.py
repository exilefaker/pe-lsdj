#!/usr/bin/env python3
"""
lsdj_stream.py: Real-time model-driven LSDJ generation via PyBoy.

Boots LSDJ in PyBoy, prefills the KV cache from a .lsdsng prompt,
then streams model-generated steps into SRAM in sync with the playhead.

See pe_lsdj.streaming.session for the generation/playback implementation.

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

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from pyboy import PyBoy

from pe_lsdj import SongFile
from pe_lsdj.constants import NUM_CHANNELS
from pe_lsdj.embedding import SongBanks
from pe_lsdj.models import LSDJTransformer
from pe_lsdj.streaming import AllocationManager, StreamingBuffer, StreamingSession
from pe_lsdj.streaming.session import find_first_empty_row

_INIT_FRAMES = 180  # ~3 s at 60 fps; lets LSDJ finish initialisation


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
    model_key, _ = jr.split(key)
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

    # Peak allocation = (max_rows + 1) rows (buffer + one building chain).
    phrases_per_row   = NUM_CHANNELS * args.num_phrases_per_chain
    chains_per_row    = NUM_CHANNELS
    safe_from_phrases = alloc.free_phrase_count // phrases_per_row - 1
    safe_from_chains  = alloc.free_chain_count  // chains_per_row  - 1
    safe_max_rows     = max(1, min(safe_from_phrases, safe_from_chains))

    if args.max_rows is None:
        max_rows = safe_max_rows
        print(f"Auto max_rows={max_rows} (headroom: "
              f"{alloc.free_phrase_count} phrases, {alloc.free_chain_count} chains)")
    else:
        if args.max_rows > safe_max_rows:
            print(f"Warning: --max-rows {args.max_rows} exceeds alloc headroom "
                  f"({safe_max_rows} safe); clamping.")
        max_rows = min(args.max_rows, safe_max_rows)

    write_ahead = args.write_ahead
    if write_ahead >= max_rows:
        write_ahead = max(1, max_rows - 1)
        print(f"Warning: write_ahead clamped to {write_ahead} (max_rows={max_rows})")

    first_row = find_first_empty_row(pyboy)
    print(f"Streaming from song row 0x{first_row:02X} ({first_row})")

    buf = StreamingBuffer(
        pyboy, alloc,
        next_song_row         = first_row,
        max_rows              = max_rows,
        num_phrases_per_chain = args.num_phrases_per_chain,
        channel_mask          = channel_mask,
    )

    # ── run ───────────────────────────────────────────────────────────────────
    session = StreamingSession(
        pyboy            = pyboy,
        model            = model,
        alloc            = alloc,
        buf              = buf,
        last_hidden      = last_hidden,
        banks            = banks,
        k_cache          = k_cache,
        v_cache          = v_cache,
        W                = W,
        song_length      = song_length,
        write_ahead      = write_ahead,
        seed             = args.seed + 1,
        window           = args.window,
        instr_threshold  = args.instr_threshold,
        table_threshold  = args.table_threshold,
        groove_threshold = args.groove_threshold,
        softsynth_threshold = args.softsynth_threshold,
        temp             = args.temp,
    )
    session.run()


if __name__ == "__main__":
    main()
