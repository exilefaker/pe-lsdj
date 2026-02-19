import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from itertools import accumulate
from jaxtyping import Array
from pe_lsdj.constants import *


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

    def __call__(self, x, soft: bool=False):
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

    def __call__(self, x):
        valid_event = jnp.all(
            (x > self.null_value) & (x <= self.max_value)
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
    """
    entity_bank: Array
    num_entities: int
    embedder: eqx.Module
    null_entry: bool

    def __init__(self, entity_bank, embedder, null_entry=False):
        """
        null_entry: if True, prepends a zero row so that index 0
        maps to a zero vector.  Tokens are assumed to already carry
        a +1 offset (0 = null).
        """
        self.null_entry = null_entry
        entity_dim = entity_bank.shape[-1]
        entity_bank = jnp.concatenate(
            [jnp.zeros((1, entity_dim), entity_bank.dtype), entity_bank],
            axis=0
        ) if null_entry else entity_bank

        self.entity_bank = entity_bank
        self.embedder = embedder

        self.num_entities = entity_bank.shape[0]
        self.in_dim = 1
        self.out_dim = embedder.out_dim

    def __call__(self, x, soft: bool=False):
        x = jnp.squeeze(x)
        soft_hot = _soft_hot(x, self.num_entities, soft)
        return self.embedder(
            soft_hot @ self.entity_bank
        )


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

    def __call__(self, x):
        embeddings = [
            e(x[self.offsets[name]:self.offsets[name] + e.in_dim])
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

    def __call__(self, x):
        embeddings = jnp.concatenate([
            e(x[self.offsets[name]:self.offsets[name] + e.in_dim])
            for name, e in self.embedders.items()
        ])
        return self.projection(embeddings)


class DummyEmbedder(BaseEmbedder):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def __call__(self, _x):
        return jnp.zeros(self.out_dim)
