import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn.initializers as init
from jaxtyping import Array
from jax import lax
from pe_lsdj.constants import *


class BaseEmbedder(eqx.Module):
    out_dim: int


def _soft_hot(x, size: int, soft: bool):
    return lax.cond(
        soft,
        lambda x: x,
        lambda x: jax.nn.one_hot(x, size),
        x
    )    

class EnumEmbedder(BaseEmbedder):
    projection: eqx.nn.Linear

    def __init__(self, vocab_size, out_dim, key):
        self.vocab_size = vocab_size
        self.out_dim = out_dim

        self.projection = eqx.nn.Linear(
            in_features=vocab_size,
            out_features=out_dim,
            use_bias=False,
            key=key,
        )

    def __call__(self, x, soft: bool=False):
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
    entity_bank: Array
    embedder: eqx.Module

    def __call__(self, x, soft: bool=False):
        """
        x: one-hot or categorical distribution
        """
        soft_hot = _soft_hot(x, self.in_dim, soft)

        return self.embedder(
            soft_hot @ self.entity_bank
        )


class SumEmbedder(BaseEmbedder):
    """
    Aggregates embeddings by summing
    """
    embedders: dict[str, eqx.Module]

    def __call__(self, **kwargs):
        embeddings = [
            e(kwargs[k]) for k, e in self.embedders.items()
        ]
        return jax.tree.reduce(jnp.add, embeddings)


class ConcatEmbedder(BaseEmbedder):
    """
    Aggregates embeddings by concatenating
    """
    embedders: dict[str, BaseEmbedder]
    projection: eqx.nn.Linear

    def __init__(self, key, embedders: dict[str, BaseEmbedder], out_dim: int):

        total_in_dim = sum([e.out_dim for e in embedders])
        self.projection = eqx.nn.Linear(
            in_features=total_in_dim,
            out_features=out_dim,
            use_bias=False,
            key=key,
        )
        self.embedders = embedders
        self.out_dim = out_dim

    def __call__(self, **kwargs):
        embeddings = jnp.concatenate(
            [e(kwargs[k]) for k, e in self.embedders.items()]
        )
        return self.projection(embeddings)



