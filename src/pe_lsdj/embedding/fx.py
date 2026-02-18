from pe_lsdj.constants import *
from pe_lsdj.embedding.base import (
    ConcatEmbedder,
    EnumEmbedder,
    EntityEmbedder,
    GatedNormedEmbedder,
    SumEmbedder,
)
import jax.numpy as jnp
import jax.random as jr


class GrooveEntityEmbedder(EntityEmbedder):
    def __init__(self, out_dim, key, grooves, **kwargs):
        super().__init__(
            grooves,
            GatedNormedEmbedder(
                out_dim,
                key,
                STEPS_PER_GROOVE * 2,
                0,
                255,
            ),
            **kwargs
        )


def build_fx_value_embedders(out_dim, key, groove_embedder):
    """
    Build 11 shared sub-embedders for FX value columns 1..11.

    Column order matches FX_VALUE_KEYS[1:]:
        GROOVE_FX(1), HOP_FX(1), PAN_FX(1), CHORD(2), ENV(2),
        RETRIG(2), VIBRATO(2), VOLUME_FX(1), WAVE_FX(1), RANDOM(2),
        CONTINUOUS_FX(1)
    """
    keys = jr.split(key, 10)
    ki = iter(range(10))

    return [
        groove_embedder,                                            # GROOVE_FX (1)
        GatedNormedEmbedder(out_dim, keys[next(ki)], 1, 0, 255),  # HOP_FX (1)
        EnumEmbedder(4, out_dim, keys[next(ki)]),                  # PAN_FX (1)
        GatedNormedEmbedder(out_dim, keys[next(ki)], 2, 0, 0x0F), # CHORD (2)
        GatedNormedEmbedder(out_dim, keys[next(ki)], 2, 0, 0x0F), # ENV (2)
        GatedNormedEmbedder(out_dim, keys[next(ki)], 2, 0, 0x0F), # RETRIG (2)
        GatedNormedEmbedder(out_dim, keys[next(ki)], 2, 0, 0x0F), # VIBRATO (2)
        GatedNormedEmbedder(out_dim, keys[next(ki)], 1, 0, 255),  # VOLUME_FX (1)
        EnumEmbedder(4, out_dim, keys[next(ki)]),                  # WAVE_FX (1)
        GatedNormedEmbedder(out_dim, keys[next(ki)], 2, 0, 0x0F), # RANDOM (2)
        GatedNormedEmbedder(out_dim, keys[next(ki)], 1, 0, 255),  # CONTINUOUS_FX (1)
    ]


class FXValueEmbedder(SumEmbedder):
    """
    17-column FX value embedder aligned with FX_VALUE_KEYS.
    Position 0 (TABLE_FX) is configurable per tier:
      - Tier 0: DummyEmbedder (base case, no table lookup)
      - Tier 1: EntityEmbedder over traces bank
      - Phrase:  EntityEmbedder over tables bank
    """
    def __init__(self, table_fx_embedder, shared_embedders):
        super().__init__([table_fx_embedder] + list(shared_embedders))


class FXEmbedder(ConcatEmbedder):
    """Combined FX cmd + FX value embedder."""
    def __init__(self, key, fx_value_embedder, out_dim=128, cmd_out_dim=32,
                 *, _projection=None):
        k1, k2 = jr.split(key)
        cmd_embedder = EnumEmbedder(19, cmd_out_dim, k1)
        embedders = [cmd_embedder, fx_value_embedder]
        super().__init__(k2, embedders, out_dim, _projection=_projection)


class TableEmbedder(ConcatEmbedder):
    """
    Embeds a table row: env_volume, env_duration, transpose, FX1, FX2.
    FX1 and FX2 share the same FXEmbedder with positional encoding.
    """
    fx_col_position: EnumEmbedder
    fx1_idx: int
    fx2_idx: int

    def __init__(self, out_dim, key, fx_embedder, *, _projection=None):
        keys = jr.split(key, 5)

        PARAMS = [
            (1, 0, 0x0F),  # env_volume
            (1, 0, 0x0F),  # env_duration
            (1, 0, 255),   # transpose
        ]
        env_embedders = [
            GatedNormedEmbedder(out_dim, keys[idx], *params)
            for idx, params in enumerate(PARAMS)
        ]

        embedders = list(env_embedders)
        self.fx1_idx = len(embedders)
        embedders.append(fx_embedder)
        self.fx2_idx = len(embedders)
        embedders.append(fx_embedder)

        self.fx_col_position = EnumEmbedder(2, fx_embedder.out_dim, keys[3])

        super().__init__(keys[4], embedders, out_dim, _projection=_projection)

    def __call__(self, x):
        embeddings = []
        for i, e in enumerate(self.embedders):
            emb = e(x[self.offsets[i]:self.offsets[i+1]])
            if i == self.fx1_idx:
                emb = emb + self.fx_col_position(jnp.array(0))
            elif i == self.fx2_idx:
                emb = emb + self.fx_col_position(jnp.array(1))
            embeddings.append(emb)
        return self.projection(jnp.concatenate(embeddings))
