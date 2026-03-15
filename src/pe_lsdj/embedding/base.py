import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from enum import Enum
from jaxtyping import Array
from pe_lsdj.constants import *


class EntityType(str, Enum):
    """Names the SongBanks field each EntityEmbedder should look up at call time."""
    INSTRUMENTS = 'instruments'
    GROOVES     = 'grooves'
    TABLES      = 'tables'
    TRACES      = 'traces'
    SYNTH_WAVES = 'synth_waves'


class BaseEmbedder(eqx.Module):
    in_dim: int
    out_dim: int


def _soft_hot(x, size: int, soft: bool):
    if soft:
        return x
    return jax.nn.one_hot(x, size)

class EnumEmbedder(BaseEmbedder):
    projection: eqx.nn.Linear
    vocab_size: int

    def __init__(self, vocab_size, out_dim, key):
        self.vocab_size = vocab_size
        self.in_dim = 1
        self.out_dim = out_dim

        self.projection = eqx.nn.Linear(
            in_features=vocab_size,
            out_features=out_dim,
            use_bias=False,
            key=key,
        )

    def __call__(self, x, _banks=None, soft: bool = False):
        x = jnp.squeeze(x)
        soft_hot = _soft_hot(x, self.vocab_size, soft)
        return self.projection(soft_hot)


class GatedNormedEmbedder(BaseEmbedder):
    gate_embedder: EnumEmbedder
    projection: eqx.nn.Linear
    max_value: int = 255
    null_value: int = 0

    def __init__(
        self,
        out_dim,
        key,
        in_dim=1,
        null_value=0,
        max_value=255,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.null_value = null_value
        self.max_value = max_value

        k1, k2 = jr.split(key)

        self.gate_embedder = EnumEmbedder(
            2,
            out_dim,
            k1
        )

        self.projection = eqx.nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
            use_bias=False, # Ensure 0 -> 0 mapping
            key=k2,
        )

    def __call__(self, x, _banks=None):
        valid_event = jnp.all(
            (x > self.null_value) & (x <= self.max_value + 1)
        ).astype(jnp.float32)

        # 2-dim "event" one-hot
        gate_emb = self.gate_embedder(valid_event)

        # Normalize only if event; otherwise it's 0.0
        normed = jnp.where(
            valid_event > 0,
            (x - 1) / self.max_value,
            0.0
        )
        continuous_emb = self.projection(normed)

        return gate_emb + continuous_emb


class EntityEmbedder(BaseEmbedder):
    """
    Grabs one of a bank of discrete entities by ID
    (or using a soft mixture), and embeds it.

    Banks are passed at call time via a SongBanks argument; this embedder
    looks up the right field using entity_type (an EntityType enum value).
    Banks are expected to be null-prepended (index 0 = zero sentinel row).
    """
    entity_type: EntityType
    embedder: eqx.Module

    def __init__(self, entity_type: EntityType, embedder):
        self.entity_type = entity_type
        self.embedder = embedder
        self.in_dim = 1
        self.out_dim = embedder.out_dim

    def __call__(self, x, banks, soft: bool = False):
        bank = getattr(banks, self.entity_type)
        x = jnp.squeeze(x)
        soft_hot = _soft_hot(x, bank.shape[0], soft)
        return self.embedder(soft_hot @ bank, banks)


def _offsets(embedders):
    """Returns dict mapping embedder name → start position."""
    pos = 0
    offsets = {}
    for name, e in embedders.items():
        offsets[name] = pos
        pos += e.in_dim
    return offsets


class SumEmbedder(BaseEmbedder):
    """
    Aggregates embeddings by summing.
    """
    embedders: dict[str, BaseEmbedder]
    offsets: dict

    def __init__(self, embedders: dict[str, BaseEmbedder]):
        self.embedders = dict(sorted(embedders.items()))
        self.offsets = _offsets(self.embedders)
        out_dims = [e.out_dim for e in self.embedders.values()]
        assert all(d == out_dims[0] for d in out_dims), (
            f"SumEmbedder: all sub-embedder out_dims must match, got {out_dims}"
        )
        self.in_dim = sum(e.in_dim for e in embedders.values())
        self.out_dim = out_dims[0]

    def __call__(self, x, banks=None):
        embeddings = [
            e(x[self.offsets[name]:self.offsets[name] + e.in_dim], banks)
            for name, e in self.embedders.items()
        ]
        return jax.tree.reduce(jnp.add, embeddings)


class ConcatEmbedder(BaseEmbedder):
    """
    Aggregates embeddings by concatenating, then projects.
    """
    embedders: dict[str, BaseEmbedder]
    offsets: dict
    projection: eqx.nn.Linear

    def __init__(self, key, embedders: dict[str, BaseEmbedder], out_dim: int,
                 *, _projection=None):
        self.embedders = dict(sorted(embedders.items()))
        self.offsets = _offsets(self.embedders)
        self.in_dim = sum(e.in_dim for e in self.embedders.values())
        self.out_dim = out_dim
        concat_dim = sum(e.out_dim for e in self.embedders.values())
        if _projection is not None:
            self.projection = _projection
        else:
            self.projection = eqx.nn.Linear(
                in_features=concat_dim,
                out_features=out_dim,
                use_bias=False,
                key=key,
            )

    def __call__(self, x, banks=None):
        embeddings = jnp.concatenate([
            e(x[self.offsets[name]:self.offsets[name] + e.in_dim], banks)
            for name, e in self.embedders.items()
        ])
        return self.projection(embeddings)


class HelixEmbedder(eqx.Module):
    """
    Encodes a discrete cyclic token (e.g. a note) as a point on a helix:
        (cos(2π·chroma/period), sin(2π·chroma/period), octave/max_octave)

    The chroma dimension is circular (period-periodic), so adjacent values
    at period boundaries (e.g. B→C in music) have small distance. The octave
    dimension is linear and normalised to [0, 1].

    NULL token (value 0) maps to the zero vector, which lies at the origin
    and is never produced by the helix projection — geometrically distinct
    from all valid tokens.

    Input:  token — shape (1,) uint16, 0=NULL, 1..num_values=valid
    Output: (out_dim,)
    """
    proj: eqx.nn.Linear   # Linear(3, out_dim, use_bias=False)
    out_dim: int
    period: int
    num_values: int

    def __init__(self, out_dim, key, period=12, num_values=72):
        self.out_dim = out_dim
        self.period = period
        self.num_values = num_values
        self.proj = eqx.nn.Linear(3, out_dim, use_bias=False, key=key)

    def __call__(self, token, _banks=None):
        val = token[0].astype(jnp.float32) - 1.0   # 0-indexed semitone; NULL → -1
        is_null = token[0] == 0
        chroma = val % float(self.period)
        octave = val // float(self.period)
        max_octave = float((self.num_values - 1) // self.period)
        coords = jnp.array([
            jnp.cos(2.0 * jnp.pi * chroma / float(self.period)),
            jnp.sin(2.0 * jnp.pi * chroma / float(self.period)),
            octave / max_octave,
        ])
        return jnp.where(is_null, jnp.zeros(self.out_dim), self.proj(coords))


class DummyEmbedder(BaseEmbedder):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def __call__(self, _x, _banks=None):
        return jnp.zeros(self.out_dim)
