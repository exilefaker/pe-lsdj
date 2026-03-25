import os
import json
import numpy as np
import datetime
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jaxtyping import Array, Key

from typing import Any, NamedTuple

from pe_lsdj import SongFile
from pe_lsdj.constants import NUM_NOTES
from pe_lsdj.embedding.song import SongBanks
from pe_lsdj.models.transformer import (
    TOKEN_HEADS, hard_targets, entity_loss, conditional_entity_loss, note_token_loss,
)


class TrainState(NamedTuple):
    model: Any
    opt_state: Any
    step: int


def save_checkpoint(path, model, opt_state, step):
    """Save model weights, optimizer state, and step as a single bundle."""
    eqx.tree_serialise_leaves(path, TrainState(model, opt_state, step))


def load_checkpoint(path, ref_model, ref_opt_state):
    """
    Load a checkpoint bundle (model, opt_state, step).

    If the file contains only weights (e.g. saved by an older version),
    loads the model and returns None for opt_state and 0 for step, with
    a warning. The caller should reinitialize opt_state in that case.
    """
    try:
        state = eqx.tree_deserialise_leaves(
            path, like=TrainState(ref_model, ref_opt_state, 0)
        )
        return state.model, state.opt_state, state.step
    except Exception:
        print(
            f"Warning: {path} does not contain a full TrainState "
            "(model + opt_state + step). Loading weights only — "
            "optimizer state will be reset to step 0."
        )
        model = eqx.tree_deserialise_leaves(path, like=ref_model)
        return model, None, 0


def _find_latest_checkpoint_in_session(session_path):
    ckpts = sorted(
        f for f in os.listdir(session_path)
        if f.startswith("step_") and f.endswith(".eqx")
    )
    if not ckpts:
        return session_path, None
    return session_path, os.path.join(session_path, ckpts[-1])


def _find_latest_checkpoint(checkpoint_path):
    """Return (session_path, checkpoint_file) for the most recent saved step.
    Returns (None, None) if no checkpoints exist."""
    sessions = sorted(
        d for d in os.listdir(checkpoint_path)
        if os.path.isdir(os.path.join(checkpoint_path, d))
    )
    if not sessions:
        return None, None
    
    session_path = os.path.join(checkpoint_path, sessions[-1])
    return _find_latest_checkpoint_in_session(session_path)


def load_songs(song_paths: list[str]) -> list[SongFile]:
    """Load all .lsdsng files upfront."""
    return [SongFile(p) for p in song_paths]

# NOTE: Legacy (assumes single-song batches) - keeping around in case
def sample_crops(song_tokens: Array, crop_len: int, batch_size: int, key: Key):
    """
    Sample random crops from one song's token sequence for teacher forcing.

    song_tokens: (S, 4, 21)
    Returns (inputs, targets):
        inputs:  (B, crop_len, 4, 21)
        targets: (B, crop_len, 4, 21)
    """
    S = song_tokens.shape[0]
    max_start = S - crop_len  # need crop_len + 1 total steps for shift
    starts = jr.randint(key, (batch_size,), 0, max_start)

    def _crop(start):
        full = jax.lax.dynamic_slice(
            song_tokens, (start, 0, 0), (crop_len + 1, 4, 21)
        )
        return full[:-1], full[1:]

    inputs, targets = jax.vmap(_crop)(starts)
    return inputs, targets


def sample_crop(song_tokens: Array, crop_len: int, key: Key):
    """
    Sample random crop from one song's token sequence for teacher forcing.

    song_tokens: (S, 4, 21)
    Returns (inputs, targets, crop_start):
        inputs:     (crop_len, 4, 21)
        targets:    (crop_len, 4, 21)
        crop_start: scalar int — absolute position of inputs[0] in the full song
    """
    S = song_tokens.shape[0]
    max_start = S - crop_len  # need crop_len + 1 total steps for shift
    start = jr.randint(key, (), 0, max_start)

    full = jax.lax.dynamic_slice(
        song_tokens, (start, 0, 0), (crop_len + 1, 4, 21)
    )
    return full[:-1], full[1:], start


def _transpose(tokens: Array, k) -> Array:
    """
    Shift all non-null note tokens by k semitones.

    tokens : (seq_len, 4, 21) float32
    k      : scalar int — semitones to shift (negative = down)

    Note tokens sit at position 0. Token 0 = NULL (left unchanged).
    Tokens 1..NUM_NOTES represent notes; out-of-range results are clamped
    to the boundary so no note disappears due to a large transposition.

    The noise channel (index 3) is excluded: noise "notes" are periodic
    noise frequency values whose perceptual meaning doesn't map to semitones —
    transposing them may degrade percussion sounds.
    """
    notes = tokens[:, :3, 0]  # PU1, PU2, WAV only — skip noise channel
    shifted = jnp.where(notes > 0, jnp.clip(notes + k, 1, NUM_NOTES), 0)
    return tokens.at[:, :3, 0].set(shifted)


def _swap_pulse(tokens: Array) -> Array:
    """Swap PU1 (channel 0) and PU2 (channel 1). All 21 token fields are swapped."""
    return tokens.at[:, :2, :].set(tokens[:, :2, :][:, ::-1, :])


def make_multi_track_batch(songs, all_banks, batch_size, crop_len, key,
                           transpose_range: int = 0,
                           swap_pulse: bool = False,
                           p_transpose: float = 0.2):
    """Sample one crop per batch item, each from a randomly chosen song.
    Returns (inputs, targets, batched_banks, idxs, crop_starts, song_lengths).
        crop_starts:  (B,) int — absolute position of each crop within its song
        song_lengths: (B,) int — full length of each selected song

    transpose_range: if > 0, each crop may be shifted by a non-zero number of
        semitones drawn from [-transpose_range, +transpose_range]. Only note 
        tokens (position 0) are affected; null notes and other fields unchanged. 
        Default 0 = no augmentation.
    p_transpose: probability of applying any transposition. The remaining
        (1 - p_transpose) mass is placed on zero (no shift). Non-zero offsets
        share p_transpose equally. Default 0.2. Ignored when transpose_range=0.
    swap_pulse: if True, each crop independently has a 50% chance of swapping
        PU1 and PU2 channels (all 21 token fields). Default False.
    """
    k1, k2, k3, k4 = jr.split(key, 4)
    idxs = jr.choice(k1, len(songs), shape=(batch_size,), replace=False)
    subkeys = jr.split(k2, batch_size)
    crops = [
        sample_crop(songs[i].song_tokens.astype(jnp.float32), crop_len, subkeys[j])
        for j, i in enumerate(idxs)
    ]

    if transpose_range > 0:
        n_shifts = 2 * transpose_range
        offsets = np.concatenate([[0], np.arange(-transpose_range, transpose_range + 1)[np.arange(-transpose_range, transpose_range + 1) != 0]])
        weights = np.array([1 - p_transpose] + [p_transpose / n_shifts] * n_shifts, dtype=np.float32)
        jax_offsets = jnp.array(offsets, dtype=jnp.int32)
        jax_weights = jnp.array(weights)
        transpose_keys = jr.split(k3, batch_size)
        def _apply(crop, tk):
            inp, tgt, start = crop
            k = jr.choice(tk, jax_offsets, p=jax_weights)
            return _transpose(inp, k), _transpose(tgt, k), start
        crops = [_apply(crop, tk) for crop, tk in zip(crops, transpose_keys)]

    if swap_pulse:
        swap_keys = jr.split(k4, batch_size)
        def _apply_swap(crop, sk):
            inp, tgt, start = crop
            do_swap = jr.bernoulli(sk)
            inp = jax.lax.cond(do_swap, _swap_pulse, lambda x: x, inp)
            tgt = jax.lax.cond(do_swap, _swap_pulse, lambda x: x, tgt)
            return inp, tgt, start
        crops = [_apply_swap(crop, sk) for crop, sk in zip(crops, swap_keys)]
    inputs = jnp.stack([c[0] for c in crops])
    targets = jnp.stack([c[1] for c in crops])
    crop_starts = jnp.stack([c[2] for c in crops])
    song_lengths = jnp.array([songs[i].song_tokens.shape[0] for i in idxs], dtype=jnp.int32)
    batched_banks = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *[all_banks[i] for i in idxs],
    )
    return inputs, targets, batched_banks, idxs, crop_starts, song_lengths


def sequence_loss(model, input_tokens: Array, target_tokens: Array, banks: SongBanks,
                  key: Key | None = None, crop_start=None, song_length=None,
                  label_smoothing: float = 0.0):
    """
    Teacher-forcing loss for one sequence: token CE + scalar entity CE + conditional entity CE.

    input_tokens:  (L, 4, 21)
    target_tokens: (L, 4, 21)
    banks:         SongBanks for the current song (null rows pre-included)
    key:           optional PRNGKey for embedding noise (None = no noise, e.g. at validation)
    crop_start:    absolute position of input_tokens[0] within the full song (int scalar).
                   Used together with song_length for the progress embedding.
    song_length:   total length of the source song (int scalar).
    Returns: scalar — mean loss per (channel × timestep)
    """
    L = input_tokens.shape[0]
    positions = jnp.arange(L) if crop_start is None else jnp.arange(L) + crop_start
    hiddens = model.encode(input_tokens, banks, key=key,
                           positions=positions, song_length=song_length)  # (L, 4, d_model)
    target_fx_cmd = target_tokens[:, :, 2]   # (L, 4) — teacher-forcing fx_cmd for val conditioning
    logits  = jax.vmap(jax.vmap(model.output_heads))(hiddens, target_fx_cmd)    # dict of (L, 4, ...)

    # Token cross-entropy (fx_cmd, fx values, transpose)
    targets  = jax.vmap(jax.vmap(hard_targets))(target_tokens)
    token_ce = 0.0
    for name, (_, vocab) in TOKEN_HEADS.items():
        log_probs = jax.nn.log_softmax(logits[name], axis=-1)
        t = targets[name]
        if label_smoothing > 0.0:
            t = (1.0 - label_smoothing) * t + label_smoothing / vocab
        token_ce -= jnp.sum(t * log_probs)

    # Factorized note loss: chroma CE (always) + octave CE (masked when NULL)
    note_tokens = jnp.int32(target_tokens[:, :, 0])   # (L, 4)
    token_ce += jnp.sum(jax.vmap(jax.vmap(note_token_loss))(
        logits['note_chroma'], logits['note_oct'], note_tokens,
    ))

    # Scalar entity loss: instrument scalars, table scalars, phrase groove.
    # Groove-slot and trace sub-entity losses are handled below.
    entity_preds = {k: logits[k] for k in ('instr', 'table', 'groove')}
    _per_step_channel = jax.vmap(
        jax.vmap(entity_loss, in_axes=(0, None, 0)),
        in_axes=(0, None, 0),
    )
    scalar_ce = jnp.sum(_per_step_channel(entity_preds, banks, target_tokens))

    # Conditional groove + trace loss: only computed for active entity slots.
    cond_ce = conditional_entity_loss(model.output_heads, hiddens, target_tokens, banks)

    L = input_tokens.shape[0]
    return (token_ce + scalar_ce + cond_ce) / (L * 4)


# NOTE: Legacy (assumes single-song batches) - keeping around in case
def batch_loss(model, input_batch: Array, target_batch: Array, banks: SongBanks):
    """Mean loss over a batch of sequences."""
    losses = jax.vmap(sequence_loss, in_axes=(None, 0, 0, None))(
        model, input_batch, target_batch, banks
    )
    return jnp.mean(losses)


def multi_track_batch_loss(model, input_batch: Array, target_batch: Array,
                           batched_banks: SongBanks, noise_keys=None,
                           crop_starts=None, song_lengths=None,
                           label_smoothing: float = 0.0):
    """Cross-track/song batch loss.
    noise_keys:      (B,) array of PRNGKeys for per-item embedding noise, or None for no noise.
    crop_starts:     (B,) int — absolute crop start positions within each song.
    song_lengths:    (B,) int — total length of each source song.
    label_smoothing: static float — token CE label smoothing ε (0 = disabled).
    """
    B = input_batch.shape[0]
    if crop_starts is None:
        crop_starts = jnp.zeros(B, dtype=jnp.int32)
    if song_lengths is None:
        song_lengths = jnp.full(B, input_batch.shape[1], dtype=jnp.int32)
    # label_smoothing is a static Python float — use partial to avoid vmap axis issues
    _seq_loss = partial(sequence_loss, label_smoothing=label_smoothing)
    if noise_keys is None:
        losses = jax.vmap(_seq_loss, in_axes=(None, 0, 0, 0, None, 0, 0))(
            model, input_batch, target_batch, batched_banks, None, crop_starts, song_lengths
        )
    else:
        losses = jax.vmap(_seq_loss, in_axes=(None, 0, 0, 0, 0, 0, 0))(
            model, input_batch, target_batch, batched_banks, noise_keys, crop_starts, song_lengths
        )
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(model, opt_state, optimizer, input_batch, target_batch, banks, key,
               crop_starts=None, song_lengths=None, label_smoothing: float = 0.0):
    """One gradient step. Returns (model, opt_state, loss).
    key is split into per-item noise keys for embedding perturbation.
    crop_starts / song_lengths: passed through to multi_track_batch_loss for
    the progress embedding (see sequence_loss for details).
    label_smoothing: static float — token CE label smoothing ε.
    """
    B = input_batch.shape[0]
    noise_keys = jr.split(key, B)
    loss, grads = eqx.filter_value_and_grad(multi_track_batch_loss)(
        model, input_batch, target_batch, banks, noise_keys, crop_starts, song_lengths,
        label_smoothing,
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def get_validate_sequences(val_songs, val_banks, crop_len):
    """
    Pre-gather non-overlapping crops from each validation song (remainder dropped).
    Returns a list of (input, target, banks, crop_start, song_length) tuples.
    """
    results = []
    for song, banks in zip(val_songs, val_banks):
        tokens = song.song_tokens.astype(jnp.float32)
        S = tokens.shape[0]
        B = (S - 1) // crop_len  # -1 ensures every crop has a target step
        for i in range(B):
            start = i * crop_len
            full = tokens[start : start + crop_len + 1]
            results.append((full[:-1], full[1:], banks, start, S))
    return results


def train(
    model,
    songs: list[SongFile],
    *,
    validation_songs: list[SongFile] | None = None,
    num_steps: int = 10_000,
    crop_len: int = 256,
    batch_size: int = 8,
    lr: float = 3e-4,
    key: Key,
    log_every: int = 50,
    checkpoint_path: str | None = None,
    resume_from_checkpoint: bool = False,
    transpose_range: int = 0,
    p_transpose: float = 0.2,
    swap_pulse: bool = False,
    label_smoothing: float = 0.0,
    weight_decay: float = 0.0,
):
    """
    Multi-song batching training loop.

    Each step samples one crop per batch item from a randomly chosen song,
    stacks per-song banks, and runs one gradient step.

    If resume_from_checkpoint=True and checkpoint_path is set, the most
    recent session and step are loaded automatically. Logs are appended to
    the existing session folder rather than creating a new one.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=num_steps // 20,
        decay_steps=num_steps,
    )
    def _wd_mask(params):
        # Apply weight decay only to weight matrices (ndim >= 2); skip biases,
        # LayerNorm scale/shift, and embeddings (all 1-D).
        return jax.tree.map(
            lambda x: eqx.is_inexact_array(x) and x.ndim >= 2,
            params,
            is_leaf=lambda x: x is None,
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=weight_decay, mask=_wd_mask),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    start_step = 0

    # Resume from checkpoint if requested
    if resume_from_checkpoint and checkpoint_path is not None:
        session_path, ckpt_file = _find_latest_checkpoint(checkpoint_path)
        if ckpt_file is not None:
            model, loaded_opt_state, start_step = load_checkpoint(
                ckpt_file, model, opt_state
            )
            if loaded_opt_state is not None:
                opt_state = loaded_opt_state
            print(f"Resumed from {ckpt_file} (step {start_step})")
        else:
            print("resume_from_checkpoint=True but no checkpoint found; starting fresh.")
            session_path = None  # fall through to create a new session below

    # Set up checkpointing + logging
    if checkpoint_path is not None:

        def write_train_params(filepath):
            with open(filepath, "w") as f:
                f.write(json.dumps({
                    "num_steps": num_steps,
                    "crop_len": crop_len,
                    "batch_size": batch_size,
                    "lr": lr,
                    "key": key.tolist(),
                    "transpose_range": transpose_range,
                    "p_transpose": p_transpose,
                    "swap_pulse": swap_pulse,
                    "label_smoothing": label_smoothing,
                    "weight_decay": weight_decay,
                }))

        resuming = resume_from_checkpoint and start_step > 0
        if not resuming:
            session_path = os.path.join(
                checkpoint_path,
                str(datetime.datetime.now())
                .replace(' ', '_')
                .replace('.', '_')
            )
            os.makedirs(session_path, exist_ok=True)
            if hasattr(model, "write_metadata"):
                model.write_metadata(
                    os.path.join(session_path, "model_hyperparams.json")
                )
            write_train_params(os.path.join(session_path, "train_params.json"))

        log_mode = "a" if resuming else "w"
        g = open(os.path.join(session_path, "losses.txt"), log_mode)
        if not resuming:
            header_str = "step,song,loss"
            if validation_songs is not None:
                header_str += ",val"
            header_str += "\n"
            g.write(header_str)

    assert batch_size <= len(songs), (
        f"batch_size ({batch_size}) must be <= number of training songs ({len(songs)}) "
        "for without-replacement sampling"
    )

    # Pre-compute banks for each song
    all_banks = [SongBanks.from_songfile(sf) for sf in songs]

    # Set up validation sequences, if applicable
    if validation_songs is not None:
        _val_seq_loss = eqx.filter_jit(sequence_loss)
        validation_banks = [SongBanks.from_songfile(vs) for vs in validation_songs]
        val_sequences = get_validate_sequences(validation_songs, validation_banks, crop_len)

    for step in range(start_step, num_steps):
        key, k_crop, k_noise = jr.split(key, 3)

        # Sample multi-song batch
        inputs, targets, banks, batch_idxs, crop_starts, song_lengths = make_multi_track_batch(
            songs, all_banks, batch_size, crop_len, k_crop,
            transpose_range, swap_pulse, p_transpose,
        )

        # Gradient step (k_noise is split per-item inside train_step)
        model, opt_state, loss = train_step(
            model, opt_state, optimizer, inputs, targets, banks, k_noise,
            crop_starts, song_lengths, label_smoothing,
        )

        if step % log_every == 0:
            batch_names = [songs[i].name for i in batch_idxs.tolist()]
            names_str = '+'.join(batch_names)
            loss_str = f"step {step:5d} | {names_str} | loss {loss:.4f}"

            # Compute validation loss if applicable
            if validation_songs is not None:
                val_losses = [
                    _val_seq_loss(model, inp, tgt, bnk, None, start, slen)
                    for inp, tgt, bnk, start, slen in val_sequences
                ]
                validation_loss = jnp.mean(jnp.stack(val_losses))
                loss_str += f" | val {validation_loss:.4f}"

            print(loss_str)

            # Save checkpoint + logs
            if checkpoint_path is not None:
                ckpt_file = os.path.join(session_path, f"step_{step:06d}.eqx")
                save_checkpoint(ckpt_file, model, opt_state, step)

                loss_log_str = f"{step:5d},{names_str},{loss:.4f}"
                if validation_songs is not None:
                    loss_log_str += f",{validation_loss:.4f}"
                loss_log_str += "\n"
                g.write(loss_log_str)
                g.flush()

    if checkpoint_path is not None:
        g.close()
    return model, opt_state
