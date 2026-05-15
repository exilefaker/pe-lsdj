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
        [--write-ahead 8] [--temp 0.9] [--headless]
"""

import argparse
import json
import os

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from pyboy import PyBoy

from pe_lsdj import SongFile
from pe_lsdj.constants import NUM_CHANNELS, NUM_FX_COMMANDS, CMD_H, CMD_M, CMD_T
from pe_lsdj.embedding import SongBanks
from pe_lsdj.models import LSDJTransformer
from pe_lsdj.streaming import AllocationManager, StreamingBuffer, StreamingSession
from pe_lsdj.streaming.session import find_first_empty_row
from pe_lsdj.streaming.webapp import StreamingWebApp

_INIT_FRAMES = 180  # ~3 s at 60 fps; lets LSDJ finish initialisation


def main():
    parser = argparse.ArgumentParser(
        description="Stream model-generated steps live into LSDJ via PyBoy."
    )
    parser.add_argument("--rom",     "-r", required=True, help="LSDJ .gb ROM path")
    parser.add_argument("--sav",     "-s", required=True, help=".sav file to boot from")
    parser.add_argument("--song",    "-g", required=True, help=".lsdsng prompt file")
    parser.add_argument("--weights", "-w", required=True, nargs="+",
                        help="Model weights (.eqx) — one file, or multiple for "
                             "live crossfade between models ({ / } keys).")
    parser.add_argument("--params",  "-p", default=None,
                        help="model_hyperparams.json (default: dir of first --weights file)")

    parser.add_argument("--write-ahead-phrases", type=int, default=2,
                        help="Target phrases ahead of step-clock (default: 2, ≈1–2 s)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Rolling buffer depth (default: auto from alloc headroom)")
    parser.add_argument("--num-phrases-per-chain", type=int, default=4,
                        help="Phrases per chain per row (default: 4 → 64 steps/row)")
    parser.add_argument("--prompt-steps", type=int, default=64,
                        help="Steps of the .lsdsng to use as prompt (default: 64)")
    parser.add_argument("--song-length", type=int, default=None,
                        help="Song-length hint for progress embedding "
                             "(default: full length of --song file)")
    parser.add_argument("--lock-progress", type=float, default=None,
                        help="Pin progress fraction for continuous generation "
                             "(e.g. 0.4 = always ~40%% done). "
                             "Toggle live with Space (SDL2) or p (terminal).")
    parser.add_argument("--temp", type=float, default=0.9,
                        help="Sampling temperature (default: 0.9)")
    parser.add_argument("--exclude-fx", type=str, default="",
                        help="Comma-separated FX commands to hard-exclude, e.g. 'H,M,T'. "
                             "Bias applied post-temperature so effect is T-invariant.")
    parser.add_argument("--instr-threshold",     type=float, default=0.5)
    parser.add_argument("--table-threshold",     type=float, default=0.5)
    parser.add_argument("--groove-threshold",    type=float, default=0.1)
    parser.add_argument("--softsynth-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true",
                        help="Run without SDL2 window (default: window on)")
    parser.add_argument("--channel-mask", type=str, default="",
                        help="Comma-separated channel indices to freeze, e.g. '2,3'")
    parser.add_argument("--record", type=str, default=None, metavar="FILE",
                        help="Save session to a .pelsdj file for later replay.")
    parser.add_argument("--web-port", type=int, default=None, metavar="PORT",
                        help="Enable web UI on this port (e.g. 8765). Off by default.")
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

    # ── FX logit biases ───────────────────────────────────────────────────────
    _FX_NAME_TO_CMD = {"H": CMD_H, "M": CMD_M, "T": CMD_T}
    logit_biases = None
    excluded_names = [n.strip().upper() for n in args.exclude_fx.split(",") if n.strip()]
    if excluded_names:
        fx_bias = jnp.zeros(NUM_FX_COMMANDS)
        valid = []
        for name in excluded_names:
            if name not in _FX_NAME_TO_CMD:
                print(f"Warning: unknown FX command '{name}' (supported: {list(_FX_NAME_TO_CMD)})")
                continue
            fx_bias = fx_bias.at[_FX_NAME_TO_CMD[name]].set(-jnp.inf)
            valid.append(name)
        if valid:
            logit_biases = {"fx_cmd": fx_bias}
            print(f"Excluding FX commands: {', '.join(valid)}")

    # ── load model(s) ─────────────────────────────────────────────────────────
    params_path = args.params or os.path.join(
        os.path.dirname(args.weights[0]), "model_hyperparams.json"
    )
    with open(params_path) as f:
        params = json.load(f)
    key = jr.PRNGKey(args.seed)
    model_key, _ = jr.split(key)
    ref = LSDJTransformer(model_key, **params)
    models = []
    for wpath in args.weights:
        print(f"Loading {wpath} ...")
        m = eqx.tree_deserialise_leaves(wpath, like=ref)
        models.append(eqx.nn.inference_mode(m))
    print(f"{len(models)} model(s) loaded.")

    # ── load prompt and prefill KV cache ─────────────────────────────────────
    sf     = SongFile(args.song)
    banks  = SongBanks.from_songfile(sf)
    tokens = jnp.asarray(sf.song_tokens[:args.prompt_steps], dtype=jnp.uint16)
    W      = tokens.shape[0]
    S      = sf.song_tokens.shape[0]
    song_length = args.song_length or S
    print(f"Prompt: {W}/{S} steps from {os.path.basename(args.song)}  |  song_length={song_length}")

    print("Pre-filling KV cache ...")
    last_hidden, k_cache, v_cache = eqx.filter_jit(models[0].prefill)(
        tokens, banks, song_length=song_length
    )
    print("KV cache ready.")

    # ── boot PyBoy ────────────────────────────────────────────────────────────
    window_mode = "null" if args.headless else "SDL2"
    print(f"Booting LSDJ ({window_mode} window, {_INIT_FRAMES} init frames) ...")
    with open(args.sav, "rb") as sav_fh:
        pyboy = PyBoy(args.rom, window=window_mode, ram_file=sav_fh)

    for _ in range(_INIT_FRAMES):
        pyboy.tick(render=not args.headless)

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
        models           = models,
        alloc            = alloc,
        buf              = buf,
        last_hidden      = last_hidden,
        banks            = banks,
        k_cache          = k_cache,
        v_cache          = v_cache,
        W                = W,
        song_length      = song_length,
        write_ahead_phrases = args.write_ahead_phrases,
        seed             = args.seed + 1,
        window           = not args.headless,
        loop_progress    = args.lock_progress,
        instr_threshold  = args.instr_threshold,
        table_threshold  = args.table_threshold,
        groove_threshold = args.groove_threshold,
        softsynth_threshold = args.softsynth_threshold,
        temp             = args.temp,
        logit_biases     = logit_biases,
        record_path      = args.record,
        record_config    = {
            "weights":            args.weights,
            "song":               args.song,
            "params":             params_path,
            "seed":               args.seed,
            "prompt_steps":       args.prompt_steps,
            "song_length":        song_length,
            "num_phrases_per_chain": args.num_phrases_per_chain,
            "initial_temp":       args.temp,
            "initial_lock_progress": args.lock_progress,
            "exclude_fx":         args.exclude_fx,
            "channel_mask":       args.channel_mask,
            "instr_threshold":    args.instr_threshold,
            "table_threshold":    args.table_threshold,
            "groove_threshold":   args.groove_threshold,
            "softsynth_threshold": args.softsynth_threshold,
        } if args.record else None,
    )
    if args.web_port is not None:
        webapp = StreamingWebApp(session, pyboy, port=args.web_port)
        webapp.start()
        session._webapp = webapp
    session.run()


if __name__ == "__main__":
    main()
