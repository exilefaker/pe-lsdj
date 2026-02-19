import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from pe_lsdj.constants import NUM_CHANNELS, STEPS_PER_PHRASE
from pe_lsdj.embedding.base import EnumEmbedder
from pe_lsdj.embedding.song import SongStepEmbedder


class SinusoidalPositionEncoding(eqx.Module):
    """Fixed sinusoidal position encoding for global timestep."""
    out_dim: int

    def __call__(self, positions):
        """
        positions: (S,) integer timestep indices
        Returns:   (S, out_dim)
        """
        d = self.out_dim
        i = jnp.arange(d // 2)
        freq = 1.0 / (10000.0 ** (2 * i / d))
        # (S, 1) * (d//2,) -> (S, d//2)
        angles = positions[:, None] * freq[None, :]
        return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class PhrasePositionEmbedder(eqx.Module):
    """Learned embedding for position within a 16-step LSDJ phrase."""
    embedder: EnumEmbedder

    def __init__(self, out_dim, key):
        self.embedder = EnumEmbedder(STEPS_PER_PHRASE, out_dim, key)

    def __call__(self, positions):
        """
        positions: (S,) integer phrase-relative indices (0–15)
        Returns:   (S, out_dim)
        """
        return jax.vmap(lambda p: self.embedder(p[None]))(positions)


class ChannelPositionEmbedder(eqx.Module):
    """Learned embedding for channel identity (PU1, PU2, WAV, NOI)."""
    embedder: EnumEmbedder

    def __init__(self, out_dim, key):
        self.embedder = EnumEmbedder(NUM_CHANNELS, out_dim, key)

    def __call__(self):
        """Returns: (4, out_dim) — one embedding per channel."""
        return jax.vmap(lambda c: self.embedder(c[None]))(jnp.arange(NUM_CHANNELS))
