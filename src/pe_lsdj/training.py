import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import orbax.checkpoint as ocp
import optax
from jaxtyping import Array, Key

from pe_lsdj import SongFile
from pe_lsdj.embedding.song import SongBanks
from pe_lsdj.models.transformer import hard_targets


def load_songs(song_paths: list[str]) -> list[SongFile]:
    """Load all .lsdsng files upfront."""
    return [SongFile(p) for p in song_paths]


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


def sequence_loss(model, input_tokens: Array, target_tokens: Array):
    """
    Teacher-forcing CE loss for one sequence.

    input_tokens:  (L, 4, 21)
    target_tokens: (L, 4, 21)
    Returns: scalar — mean CE per (channel × timestep)
    """
    logits = model(input_tokens)  # dict of (L, 4, vocab_i)
    targets = jax.vmap(jax.vmap(hard_targets))(target_tokens)  # dict of (L, 4, vocab_i)

    total = 0.0
    for name in logits:
        log_probs = jax.nn.log_softmax(logits[name], axis=-1)
        total -= jnp.sum(targets[name] * log_probs)

    L = input_tokens.shape[0]
    return total / (L * 4)


def batch_loss(model, input_batch: Array, target_batch: Array):
    """Mean loss over a batch of sequences."""
    losses = jax.vmap(sequence_loss, in_axes=(None, 0, 0))(
        model, input_batch, target_batch
    )
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(model, opt_state, optimizer, input_batch, target_batch):
    """One gradient step. Returns (model, opt_state, loss)."""
    loss, grads = eqx.filter_value_and_grad(batch_loss)(
        model, input_batch, target_batch
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
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_path, 
        options=ocp.CheckpointManagerOptions(
            max_to_keep=3, 
            create=True
        )
    ) if checkpoint_path is not None else None

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
            model, opt_state, optimizer, inputs, targets
        )

        if step % log_every == 0:
            print(f"step {step:5d} | song {song_idx} | loss {loss:.4f}")
            if checkpoint_manager is not None:
                checkpoint_manager.save(
                    step, 
                    args=ocp.args.StandardSave(model)
                )

    return model, opt_state
