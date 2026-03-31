"""
generate_one.py — single-model generation CLI

Single song:
    python3 generate_one.py --weights data/weights/v10_helix/step_005350.eqx \
                            --song data/orbital-final.lsdsng \
                            --output data/generated \
                            --num-steps 1024 --num-samples 2 --seed 42

All songs in a folder (model loaded once):
    python3 generate_one.py --weights data/weights/v10_helix/step_005350.eqx \
                            --songs-dir data/train \
                            --output data/generated \
                            --num-steps 1024 --num-samples 2
"""
import argparse
import glob
import json
import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from pe_lsdj import SongFile
from pe_lsdj.embedding import SongBanks
from pe_lsdj.generation import generate
from pe_lsdj.models import LSDJTransformer


def _infer_params_path(weights_path: str) -> str:
    base = weights_path if os.path.isdir(weights_path) else os.path.dirname(weights_path)
    return os.path.join(base, "model_hyperparams.json")


def _load_model(key, weights_path: str, params_path: str) -> eqx.Module:
    with open(params_path) as f:
        params = json.load(f)
    ref_model = LSDJTransformer(key, **params)
    return eqx.tree_deserialise_leaves(weights_path, like=ref_model)


def _load_swa_model(key, weights_dir: str, params_path: str,
                    swa_start: int, swa_stop: int | None, swa_step: int) -> eqx.Module:
    with open(params_path) as f:
        params = json.load(f)
    ref_model = LSDJTransformer(key, **params)

    all_ckpts = sorted(glob.glob(os.path.join(glob.escape(weights_dir), "step_*.eqx")))
    def _step(p):
        return int(os.path.basename(p).removeprefix("step_").removesuffix(".eqx"))
    ckpt_paths = [
        p for p in all_ckpts
        if _step(p) >= swa_start
        and (swa_stop is None or _step(p) <= swa_stop)
        and (_step(p) - swa_start) % swa_step == 0
    ]
    if not ckpt_paths:
        stop_str = str(swa_stop) if swa_stop is not None else "end"
        print(f"Error: no checkpoints found in {weights_dir} "
              f"for steps {swa_start}..{stop_str} (stride {swa_step})", file=sys.stderr)
        sys.exit(1)

    steps = [_step(p) for p in ckpt_paths]
    print(f"SWA: averaging {len(ckpt_paths)} checkpoints "
          f"(steps {steps[0]}..{steps[-1]}, stride {swa_step}) ...")
    checkpoints = [eqx.tree_deserialise_leaves(p, like=ref_model) for p in ckpt_paths]

    dynamic_list, static = [], None
    for ckpt in checkpoints:
        dyn, stat = eqx.partition(ckpt, eqx.is_array)
        dynamic_list.append(dyn)
        if static is None:
            static = stat

    avg_dynamic = jax.tree.map(
        lambda *arrs: jnp.mean(jnp.stack(arrs), axis=0),
        *dynamic_list,
    )
    return eqx.combine(avg_dynamic, static)


def _generate_song(model, song_path, gen_key, args, base_name=None):
    sf = SongFile(song_path)
    banks = SongBanks.from_songfile(sf)
    base_name = base_name or sf.name
    input_tokens = sf.song_tokens[:args.prompt_steps, :, :]
    print(f"Prompt: {args.prompt_steps} steps from {song_path}")

    print(f"Generating {args.num_samples} sample(s), {args.num_steps} steps each ...")
    gen_tokens, gen_banks = eqx.filter_jit(generate)(
        model,
        input_tokens=input_tokens,
        key=gen_key,
        banks=banks,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        temperature=args.temperature,
        instr_match_threshold=args.instr_threshold,
        groove_match_threshold=args.groove_threshold,
        table_match_threshold=args.table_threshold,
        softsynth_match_threshold=args.softsynth_threshold,
        use_kv_cache=not args.no_kv_cache,
    )
    print("Generation complete.")

    # LSDJ song names are max 8 chars; no underscores. Reserve space for sample
    # index suffix when generating multiple samples.
    suffix_width = len(str(args.num_samples - 1)) if args.num_samples > 1 else 0
    truncated_base = base_name[:8 - suffix_width].upper()

    for i in range(args.num_samples):
        tokens_i = gen_tokens[i]
        banks_i = jax.tree.map(lambda x: x[i], gen_banks)
        out_name = f"{truncated_base}{i}" if args.num_samples > 1 else truncated_base
        out_path = os.path.join(args.output, f"{out_name}.lsdsng")
        out_file = SongFile.from_tokens(tokens_i, banks_i, name=out_name)
        out_file.to_lsdsng(out_path)
        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LSDJ songs from a trained model.")

    parser.add_argument("--weights", "-w", required=True,
                        help="Path to .eqx checkpoint file, or a weights folder when using --swa-start/stop.")
    parser.add_argument("--params", "-p", default=None,
                        help="Path to model_hyperparams.json. "
                             "Defaults to model_hyperparams.json in the same directory as --weights.")
    parser.add_argument("--swa-start", type=int, default=None,
                        help="Enable SWA: first checkpoint step to average. "
                             "--weights must be a folder. Steps are selected by stride --swa-step.")
    parser.add_argument("--swa-stop", type=int, default=None,
                        help="Last checkpoint step for SWA. If omitted, uses all available from --swa-start.")
    parser.add_argument("--swa-step", type=int, default=50,
                        help="Stride between SWA checkpoints. (default: 50)")

    song_group = parser.add_mutually_exclusive_group(required=True)
    song_group.add_argument("--song", "-s",
                            help="Path to a single input .lsdsng file used as prompt.")
    song_group.add_argument("--songs-dir", "-d",
                            help="Directory of .lsdsng files; model is loaded once and run on each.")

    parser.add_argument("--output", "-o", default="data/generated",
                        help="Output directory for .lsdsng files. (default: data/generated)")
    parser.add_argument("--name", default=None,
                        help="Base name for output files (single-song mode only). "
                             "Defaults to the project name embedded in the .lsdsng.")
    parser.add_argument("--num-steps", "-n", type=int, default=1024,
                        help="Number of generation steps. (default: 1024)")
    parser.add_argument("--prompt-steps", type=int, default=64,
                        help="Number of song steps to use as prompt. (default: 64)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of independent samples to generate. (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. (default: 42)")
    parser.add_argument("--instr-threshold", type=float, default=0.5,
                        help="Instrument match threshold. (default: 0.5)")
    parser.add_argument("--groove-threshold", type=float, default=0.1,
                        help="Groove match threshold. (default: 0.1)")
    parser.add_argument("--table-threshold", type=float, default=0.5,
                        help="Table match threshold. (default: 0.5)")
    parser.add_argument("--softsynth-threshold", type=float, default=0.5,
                        help="Softsynth match threshold. (default: 0.5)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for categorical distributions. "
                             "< 1.0 = more conservative, > 1.0 = more random. (default: 1.0)")
    parser.add_argument("--no-kv-cache", action="store_true",
                        help="Disable KV-cache (slower, useful for debugging).")

    args = parser.parse_args()

    params_path = args.params or _infer_params_path(args.weights)
    if not os.path.exists(params_path):
        print(f"Error: params file not found: {params_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    key = jr.PRNGKey(args.seed)
    model_key, gen_key = jr.split(key)

    if args.swa_start is not None:
        model = _load_swa_model(model_key, args.weights, params_path,
                                args.swa_start, args.swa_stop, args.swa_step)
    else:
        print(f"Loading model from {args.weights} ...")
        model = _load_model(model_key, args.weights, params_path)
    print("Model loaded.")

    if args.song:
        _generate_song(model, args.song, gen_key, args, base_name=args.name)
    else:
        song_paths = sorted(glob.glob(os.path.join(args.songs_dir, "*.lsdsng")))
        if not song_paths:
            print(f"No .lsdsng files found in {args.songs_dir}", file=sys.stderr)
            sys.exit(1)
        for song_path in song_paths:
            print(f"\n=== {os.path.basename(song_path)} ===")
            gen_key, use_key = jr.split(gen_key)
            _generate_song(model, song_path, use_key, args)


if __name__ == "__main__":
    main()
