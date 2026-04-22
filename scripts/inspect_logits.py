#!/usr/bin/env python3
"""
Diagnostic: capture fx_cmd logits across independent autoregressive trajectories.

Runs one full generation loop per temperature in --temps, with each trajectory
sharing the same prompt, KV-cache prefill, and random keys.  Temperature is the
only variable, so the comparison directly reflects how it changes the model's
behaviour (both the sampling distribution and the context history it produces).

Raw logits for each trajectory are saved to a .npz file, and a per-temperature
summary table is printed inline.

Usage:
    python3 inspect_logits.py \\
        --weights data/weights/v13_1024/step_011950.eqx \\
        --song    data/gen_test/my_song.lsdsng \\
        --num-steps 256 \\
        --temps 0.7 1.0 1.4 \\
        --output inspect_out/logits.npz
"""
import argparse
import json
import os
import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np

from pe_lsdj import SongFile
from pe_lsdj.embedding import SongBanks
from pe_lsdj.generation import resolve_step
from pe_lsdj.models import LSDJTransformer
from pe_lsdj.constants import (
    CMD_NULL, CMD_A, CMD_C, CMD_D, CMD_E, CMD_F, CMD_G,
    CMD_H, CMD_K, CMD_L, CMD_M, CMD_O, CMD_P, CMD_R,
    CMD_S, CMD_T, CMD_V, CMD_W, CMD_Z,
)

CMD_NAMES = {
    CMD_NULL: "---",
    CMD_A: "A", CMD_C: "C", CMD_D: "D", CMD_E: "E", CMD_F: "F",
    CMD_G: "G", CMD_H: "H", CMD_K: "K", CMD_L: "L", CMD_M: "M",
    CMD_O: "O", CMD_P: "P", CMD_R: "R", CMD_S: "S", CMD_T: "T",
    CMD_V: "V", CMD_W: "W", CMD_Z: "Z",
}
FX_VOCAB = 19  # token indices 0..18


# ── JIT-compiled single generation step ──────────────────────────────────────

@eqx.filter_jit
def _capture_step(
    model,
    last_hidden,   # (4, d_model)
    k_cache,
    v_cache,
    banks,
    key,
    abs_pos,
    song_length,
    temperature,
    instr_threshold,
    table_threshold,
    groove_threshold,
    softsynth_threshold,
):
    logits_dict, latents = jax.vmap(model.output_heads.generation_outputs)(last_hidden)
    fx_cmd_logits = logits_dict['fx_cmd']  # (4, 19) — raw pre-temperature logits

    next_token, banks_out = resolve_step(
        model.output_heads, banks, key, logits_dict, latents,
        instr_threshold, table_threshold, groove_threshold, softsynth_threshold,
        temperature,
    )

    x_new = model.embedder(
        next_token[None], banks_out,
        positions=jnp.asarray(abs_pos)[None],
        song_length=song_length,
    )
    new_hidden, k_cache, v_cache = model._encode_one_cached(x_new, k_cache, v_cache, abs_pos)

    return new_hidden, banks_out, k_cache, v_cache, next_token, fx_cmd_logits


def _run_trajectory(model, last_hidden_init, k_cache_init, v_cache_init,
                    banks_init, keys, W, song_length, temperature, args):
    """Run one full generation loop at a given temperature; return (num_steps, 4, 19) logits."""
    last_hidden, k_cache, v_cache, banks = (
        last_hidden_init, k_cache_init, v_cache_init, banks_init
    )
    all_logits = []

    for step_idx in range(args.num_steps):
        last_hidden, banks, k_cache, v_cache, _tok, fx_logits = _capture_step(
            model, last_hidden, k_cache, v_cache, banks,
            keys[step_idx], W + step_idx, song_length,
            temperature,
            args.instr_threshold, args.table_threshold,
            args.groove_threshold, args.softsynth_threshold,
        )
        # jax.block_until_ready((last_hidden, k_cache, v_cache))
        all_logits.append(np.array(fx_logits))  # discard next_token (_tok)

        if (step_idx + 1) % 32 == 0:
            print(f"    step {step_idx + 1}/{args.num_steps}")

    return np.stack(all_logits, axis=0)  # (num_steps, 4, 19)


# ── summary printing ──────────────────────────────────────────────────────────

def _mean_probs(logits_np):
    """logits_np: (N, 4, 19) → mean softmax prob per cmd, averaged over steps & channels."""
    shifted = logits_np - logits_np.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs.mean(axis=(0, 1))  # (19,)


def print_summary(logits_by_temp):
    """logits_by_temp: {temperature: (num_steps, 4, 19)}"""
    temps = sorted(logits_by_temp)
    per_temp = {t: _mean_probs(logits_by_temp[t]) for t in temps}

    col_w = 11
    header = f"{'CMD':<5}" + "".join(f"T={t:<{col_w}.2f}" for t in temps)
    print()
    print("── fx_cmd mean sampling probability (independent trajectories) ──")
    print(header)
    print("─" * len(header))

    sort_probs = per_temp[temps[len(temps) // 2]]
    for idx in np.argsort(-sort_probs):
        name = CMD_NAMES.get(int(idx), f"?{idx}")
        row = f"{name:<5}" + "".join(f"{per_temp[t][idx]:.5f}     " for t in temps)
        print(row)

    print()
    for cmd_idx, label in [(CMD_M, "M (master-vol)"), (CMD_T, "T (tempo)")]:
        print(f"  {label}:")
        for t in temps:
            p = per_temp[t][cmd_idx]
            print(f"    T={t:.2f}  {p*100:.4f}%")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run independent generation trajectories per temperature and compare fx_cmd distributions."
    )
    parser.add_argument("--weights",  "-w", required=True)
    parser.add_argument("--params",   "-p", default=None)
    parser.add_argument("--song",     "-s", required=True)
    parser.add_argument("--num-steps", "-n", type=int, default=128)
    parser.add_argument("--prompt-steps", type=int, default=64)
    parser.add_argument("--temps", type=float, nargs="+", default=[0.7, 1.0, 1.4],
                        help="Temperatures to run separate trajectories for. (default: 0.7 1.0 1.4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", default=None,
                        help="Path for output .npz. Defaults to inspect_logits_<timestamp>.npz.")
    parser.add_argument("--instr-threshold",     type=float, default=0.5)
    parser.add_argument("--table-threshold",     type=float, default=0.5)
    parser.add_argument("--groove-threshold",    type=float, default=0.1)
    parser.add_argument("--softsynth-threshold", type=float, default=0.5)
    args = parser.parse_args()

    # ── load model ────────────────────────────────────────────────────────────
    params_path = args.params or os.path.join(os.path.dirname(args.weights), "model_hyperparams.json")
    print(f"Loading model from {args.weights} ...")
    with open(params_path) as f:
        params = json.load(f)
    key = jr.PRNGKey(args.seed)
    model_key, gen_key = jr.split(key)
    ref = LSDJTransformer(model_key, **params)
    model = eqx.tree_deserialise_leaves(args.weights, like=ref)
    model = eqx.nn.inference_mode(model)
    print("Model loaded.")

    # ── load song & prompt ────────────────────────────────────────────────────
    sf    = SongFile(args.song)
    banks = SongBanks.from_songfile(sf)
    prompt_tokens = jnp.asarray(sf.song_tokens[:args.prompt_steps], dtype=jnp.uint16)
    W = prompt_tokens.shape[0]
    song_length = W + args.num_steps
    print(f"Prompt: {W} steps from {args.song}")

    # ── prefill once; reuse across all temperatures ───────────────────────────
    print("Pre-filling KV cache ...")
    last_hidden_init, k_cache_init, v_cache_init = eqx.filter_jit(model.prefill)(
        prompt_tokens, banks, song_length=song_length
    )

    # Same random keys for every trajectory so temperature is the only variable
    keys = jr.split(gen_key, args.num_steps)

    # ── one trajectory per temperature ───────────────────────────────────────
    logits_by_temp = {}
    for temp in args.temps:
        print(f"\nT={temp:.2f}: running {args.num_steps} steps ...")
        logits_by_temp[temp] = _run_trajectory(
            model, last_hidden_init, k_cache_init, v_cache_init,
            banks, keys, W, song_length, temp, args,
        )
        print(f"  done — logits shape {logits_by_temp[temp].shape}")

    # ── save ──────────────────────────────────────────────────────────────────
    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"inspect_logits_{ts}.npz"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_dict = {f"logits_T{t}": logits_by_temp[t] for t in args.temps}
    save_dict["temps"] = np.array(args.temps)
    save_dict["num_steps"] = np.int32(args.num_steps)
    save_dict["prompt_steps"] = np.int32(args.prompt_steps)
    np.savez(args.output, **save_dict)
    print(f"\nSaved → {args.output}")

    # ── print summary ─────────────────────────────────────────────────────────
    print_summary(logits_by_temp)


if __name__ == "__main__":
    main()
