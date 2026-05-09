#!/usr/bin/env python3
"""
average_weights.py: Offline stochastic weight averaging (SWA).

Averages a selection of checkpoints and saves the result to a single .eqx
file, which can then be passed directly to lsdj_stream.py (or generate.py)
via --weights.

Two selection modes (mutually exclusive):

  --weights-dir   Scan a folder for step_<N>.eqx files; filter by
                  --swa-start / --swa-stop / --swa-step. Files are
                  processed in ascending step order.

  --files         Provide an explicit list of .eqx files in any order.
                  The mixing-weight index corresponds directly to the
                  order given on the command line.

Usage:
    # SWA over a folder
    python3 scripts/average_weights.py \
        --weights-dir data/weights/v13_reg \
        --out         data/weights/v13_reg_swa.eqx \
        [--params     data/weights/v13_reg/model_hyperparams.json] \
        [--swa-start  9000] [--swa-stop 12000] [--swa-step 50] \
        [--mixing-weights 0.3,0.7,...]

    # Blend arbitrary files
    python3 scripts/average_weights.py \
        --files  data/weights/v13/step_9000.eqx \
                 data/weights/v13_reg/step_8500.eqx \
        --out    data/weights/blend.eqx \
        --mixing-weights 0.4,0.6
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

from pe_lsdj.models import LSDJTransformer


def _step(path: str) -> int:
    return int(os.path.basename(path).removeprefix("step_").removesuffix(".eqx"))


def main():
    parser = argparse.ArgumentParser(
        description="Average LSDJ-transformer checkpoints (SWA) and save to a single .eqx file."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights-dir", "-w",
                     help="Folder containing step_*.eqx checkpoints.")
    src.add_argument("--files", "-f", nargs="+",
                     help="Explicit list of .eqx files to blend (any order; "
                          "mixing-weight indices follow command-line order).")

    parser.add_argument("--out", "-o", required=True,
                        help="Output path for the averaged weights (.eqx).")
    parser.add_argument("--params", "-p", default=None,
                        help="model_hyperparams.json "
                             "(default: <weights-dir>/model_hyperparams.json "
                             "or directory of first --files entry).")
    parser.add_argument("--swa-start", type=int, default=None,
                        help="First checkpoint step to include (--weights-dir only). "
                             "If omitted, start from the earliest.")
    parser.add_argument("--swa-stop", type=int, default=None,
                        help="Last checkpoint step to include (--weights-dir only). "
                             "If omitted, use all from --swa-start.")
    parser.add_argument("--swa-step", type=int, default=50,
                        help="Stride between selected checkpoints (--weights-dir only, default: 50).")
    parser.add_argument("--mixing-weights", type=str, default=None,
                        help="Comma-separated floats, one per checkpoint "
                             "(e.g. '0.3,0.7'). Automatically normalized to sum 1. "
                             "Default: uniform.")
    args = parser.parse_args()

    # ── resolve checkpoint list ───────────────────────────────────────────────
    if args.files:
        ckpt_paths = args.files
        for p in ckpt_paths:
            if not os.path.isfile(p):
                print(f"Error: file not found: {p}", file=sys.stderr)
                sys.exit(1)
        params_path = args.params or os.path.join(
            os.path.dirname(os.path.abspath(ckpt_paths[0])), "model_hyperparams.json"
        )
        print(f"Blending {len(ckpt_paths)} explicit file(s):")
        for p in ckpt_paths:
            print(f"  {p}")
    else:
        params_path = args.params or os.path.join(args.weights_dir, "model_hyperparams.json")
        all_ckpts = sorted(
            glob.glob(os.path.join(glob.escape(args.weights_dir), "step_*.eqx")),
            key=_step,
        )
        if not all_ckpts:
            print(f"Error: no step_*.eqx files found in {args.weights_dir}", file=sys.stderr)
            sys.exit(1)
        swa_start = args.swa_start if args.swa_start is not None else _step(all_ckpts[0])
        ckpt_paths = [
            p for p in all_ckpts
            if _step(p) >= swa_start
            and (args.swa_stop is None or _step(p) <= args.swa_stop)
            and (_step(p) - swa_start) % args.swa_step == 0
        ]
        if not ckpt_paths:
            stop_str = str(args.swa_stop) if args.swa_stop is not None else "end"
            print(f"Error: no checkpoints matched "
                  f"steps {swa_start}..{stop_str} (stride {args.swa_step})", file=sys.stderr)
            sys.exit(1)
        steps = [_step(p) for p in ckpt_paths]
        print(f"Averaging {len(ckpt_paths)} checkpoint(s): "
              f"steps {steps[0]}..{steps[-1]}, stride {args.swa_step}")

    if not os.path.isfile(params_path):
        print(f"Error: params file not found: {params_path}", file=sys.stderr)
        sys.exit(1)

    # ── resolve mixing weights ────────────────────────────────────────────────
    if args.mixing_weights is not None:
        raw_w = [float(x) for x in args.mixing_weights.split(",")]
        if len(raw_w) != len(ckpt_paths):
            print(f"Error: --mixing-weights has {len(raw_w)} value(s) "
                  f"but {len(ckpt_paths)} checkpoint(s) selected", file=sys.stderr)
            sys.exit(1)
        total = sum(raw_w)
        weights = [w / total for w in raw_w]
        print(f"Mixing weights (normalized): {[round(w, 6) for w in weights]}")
    else:
        weights = [1.0 / len(ckpt_paths)] * len(ckpt_paths)

    # ── load and average ──────────────────────────────────────────────────────
    with open(params_path) as f:
        params = json.load(f)
    ref = LSDJTransformer(jr.PRNGKey(0), **params)

    dynamic_list, static = [], None
    for i, path in enumerate(ckpt_paths):
        print(f"  [{i+1}/{len(ckpt_paths)}] loading {os.path.basename(path)} ...")
        ckpt = eqx.tree_deserialise_leaves(path, like=ref)
        dyn, stat = eqx.partition(ckpt, eqx.is_array)
        dynamic_list.append(dyn)
        if static is None:
            static = stat

    print("Averaging ...")
    avg_dynamic = jax.tree.map(
        lambda *arrs: sum(w * a for w, a in zip(weights, arrs)),
        *dynamic_list,
    )
    averaged = eqx.combine(avg_dynamic, static)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    eqx.tree_serialise_leaves(args.out, averaged)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
