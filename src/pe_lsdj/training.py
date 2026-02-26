import os
import json
import datetime
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jaxtyping import Array, Key

from pe_lsdj import SongFile
from pe_lsdj.embedding.song import SongBanks
from pe_lsdj.models.transformer import (
    TOKEN_HEADS, ENTITY_OUTPUT_HEADS, hard_targets, entity_loss,
)


def load_songs(song_paths: list[str]) -> list[SongFile]:
    """Load all .lsdsng files upfront."""
    return [SongFile(p) for p in song_paths]


def load_weights(reference_model, filepath):
    """
    Load checkpointed model weights, given an isomorphic model 'skeleton'
    i.e., first initialize a model with the same hyperparams
    """
    return eqx.tree_deserialise_leaves(filepath, like=reference_model)  


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


def sequence_loss(model, input_tokens: Array, target_tokens: Array, banks: SongBanks):
    """
    Teacher-forcing loss for one sequence: logit-group CE + entity param CE.

    input_tokens:  (L, 4, 21)
    target_tokens: (L, 4, 21)
    banks:         SongBanks for the current song (null rows pre-included)
    Returns: scalar — mean loss per (channel × timestep)
    """
    logits = model(input_tokens)   # dict of (L, 4, ...)

    # Logit-group cross-entropy
    targets = jax.vmap(jax.vmap(hard_targets))(target_tokens)
    token_ce = 0.0
    for name in TOKEN_HEADS:
        log_probs = jax.nn.log_softmax(logits[name], axis=-1)
        token_ce -= jnp.sum(targets[name] * log_probs)

    # Entity parameter cross-entropy: vmap entity_param_loss over (L, 4)
    # in_axes=(0, None, 0): batch over first axis of logit dicts and target_tokens,
    # keep banks constant (not batched).
    entity_logits = {name: logits[name] for name in ENTITY_OUTPUT_HEADS}
    _per_step_channel = jax.vmap(
        jax.vmap(entity_loss, in_axes=(0, None, 0)),
        in_axes=(0, None, 0),
    )
    entity_ce = jnp.sum(_per_step_channel(entity_logits, banks, target_tokens))

    L = input_tokens.shape[0]
    return (token_ce + entity_ce) / (L * 4)


def batch_loss(model, input_batch: Array, target_batch: Array, banks: SongBanks):
    """Mean loss over a batch of sequences."""
    losses = jax.vmap(sequence_loss, in_axes=(None, 0, 0, None))(
        model, input_batch, target_batch, banks
    )
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(model, opt_state, optimizer, input_batch, target_batch, banks):
    """One gradient step. Returns (model, opt_state, loss)."""
    loss, grads = eqx.filter_value_and_grad(batch_loss)(
        model, input_batch, target_batch, banks
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train(
    model,
    songs: list[SongFile],
    *,
    num_steps: int = 10_000,
    crop_len: int = 256,
    batch_size: int = 8,
    lr: float = 3e-4,
    key: Key,
    log_every: int = 50,
    checkpoint_path: str | None = None,
):
    """
    Per-song batching training loop.

    Each step picks a song (round-robin), swaps entity banks,
    samples random crops, and runs one gradient step.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=num_steps // 20,
        decay_steps=num_steps,
    )
    # optimizer = optax.adam(lr)
    # Trying a schedule to mitigate some oscillation
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Optionally set up model checkpointing + logging
    if checkpoint_path is not None:

        def write_train_params(filepath):
            with open(filepath, "w") as f:
                f.write(json.dumps({
                    "num_steps": num_steps,
                    "crop_len": crop_len,
                    "batch_size": batch_size,
                    "lr": lr,
                    "key": key.tolist(),
                }))

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
        write_train_params(
            os.path.join(session_path, "train_params.json")
        )
        g = open(os.path.join(session_path, "losses.txt"), "w")
        g.write("step,song,loss\n")

    # Pre-compute banks for each song
    all_banks = [SongBanks.from_songfile(sf) for sf in songs]
    all_tokens = [sf.song_tokens.astype(jnp.float32) for sf in songs]

    for step in range(num_steps):
        key, k_crop = jr.split(key)

        # Pick song (round-robin)
        song_idx = step % len(songs)
        tokens = all_tokens[song_idx]
        banks = all_banks[song_idx]

        # Swap entity banks
        model = model.with_banks(banks)

        # Sample crops
        inputs, targets = sample_crops(tokens, crop_len, batch_size, k_crop)

        # Gradient step
        model, opt_state, loss = train_step(
            model, opt_state, optimizer, inputs, targets, banks
        )

        if step % log_every == 0:
            song_name = songs[song_idx].name
            print(f"step {step:5d} | song {song_name} | loss {loss:.4f}")
            if checkpoint_path is not None:
                ckpt_file = os.path.join(session_path, f"step_{step:06d}.eqx")
                eqx.tree_serialise_leaves(ckpt_file, model)
                g.write(f"{step:5d},{song_name},{loss:.4f}\n") # TODO maybe log (step, loss) as csv?

    if checkpoint_path is not None:
        g.close()
    return model, opt_state
