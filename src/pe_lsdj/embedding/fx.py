from pe_lsdj.constants import *
from pe_lsdj.embedding.base import (
    BaseEmbedder,
    ConcatEmbedder,
    EnumEmbedder,
    EntityEmbedder,
    GatedNormedEmbedder,
    SumEmbedder,
    _offsets,
)
import equinox as eqx
import jax
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
    Build shared sub-embedders for FX value columns 1..11.
    Column order matches FX_VALUE_KEYS[1:].
    Returns dict[str, BaseEmbedder].
    """
    keys = jr.split(key, 10)
    ki = iter(range(10))

    return {
        'groove': groove_embedder,
        'hop': GatedNormedEmbedder(out_dim, keys[next(ki)], 1, 0, 255),
        'pan': EnumEmbedder(4, out_dim, keys[next(ki)]),
        'chord': GatedNormedEmbedder(out_dim, keys[next(ki)], 2, 0, 0x0F),
        'env': GatedNormedEmbedder(out_dim, keys[next(ki)], 2, 0, 0x0F),
        'retrig': GatedNormedEmbedder(out_dim, keys[next(ki)], 2, 0, 0x0F),
        'vibrato': GatedNormedEmbedder(out_dim, keys[next(ki)], 2, 0, 0x0F),
        'volume': GatedNormedEmbedder(out_dim, keys[next(ki)], 1, 0, 255),
        'wave': EnumEmbedder(4, out_dim, keys[next(ki)]),
        'random': GatedNormedEmbedder(out_dim, keys[next(ki)], 2, 0, 0x0F),
        'continuous': GatedNormedEmbedder(out_dim, keys[next(ki)], 1, 0, 255),
    }


class FXValueEmbedder(SumEmbedder):
    """
    17-column FX value embedder aligned with FX_VALUE_KEYS.
    Position 'table_fx' is configurable per tier:
      - Tier 0: DummyEmbedder (base case, no table lookup)
      - Tier 1: EntityEmbedder over traces bank
      - Phrase: EntityEmbedder over tables bank
    """
    def __init__(self, table_fx_embedder, shared_embedders):
        super().__init__({'table_fx': table_fx_embedder, **shared_embedders})


class FXEmbedder(ConcatEmbedder):
    """Combined FX cmd + FX value embedder."""
    def __init__(self, key, fx_value_embedder, out_dim=128, cmd_out_dim=32,
                 *, _projection=None):
        k1, k2 = jr.split(key)
        cmd_embedder = EnumEmbedder(19, cmd_out_dim, k1)
        embedders = {'cmd': cmd_embedder, 'value': fx_value_embedder}
        super().__init__(k2, embedders, out_dim, _projection=_projection)


class TableEmbedder(BaseEmbedder):
    """
    Embeds a full table (STEPS_PER_TABLE steps of TABLE_STEP_WIDTH features).
    Each step is embedded via shared sub-embedders (env×3, FX×2),
    then all step embeddings are concatenated and projected.

    Input:  (STEPS_PER_TABLE * TABLE_STEP_WIDTH,) = (624,) = (TABLE_WIDTH,)
    Output: (out_dim,)
    """
    embedders: dict[str, BaseEmbedder]
    offsets: dict
    projection: eqx.nn.Linear
    fx_col_position: EnumEmbedder

    def __init__(self, out_dim, key, fx_embedder, *, _projection=None):
        keys = jr.split(key, 6)

        PARAMS = [
            (1, 0, 0x0F),  # env_volume
            (1, 0, 0x0F),  # env_duration
            (1, 0, 255),   # transpose
        ]

        self.embedders = dict(sorted({
            'env_volume': GatedNormedEmbedder(out_dim, keys[0], *PARAMS[0]),
            'env_duration': GatedNormedEmbedder(out_dim, keys[1], *PARAMS[1]),
            'transpose': GatedNormedEmbedder(out_dim, keys[2], *PARAMS[2]),
            'fx1': fx_embedder,
            'fx2': fx_embedder,
        }.items()))

        self.offsets = _offsets(self.embedders)
        self.fx_col_position = EnumEmbedder(2, fx_embedder.out_dim, keys[3])

        step_concat_dim = sum(e.out_dim for e in self.embedders.values())
        if _projection is None:
            _projection = eqx.nn.Linear(
                in_features=STEPS_PER_TABLE * step_concat_dim,
                out_features=out_dim,
                use_bias=False,
                key=keys[5],
            )
        self.projection = _projection

        step_in_dim = sum(e.in_dim for e in self.embedders.values())
        self.in_dim = STEPS_PER_TABLE * step_in_dim
        self.out_dim = out_dim

    def _embed_step(self, x):
        """Embed one table step (TABLE_STEP_WIDTH,) -> (step_concat_dim,)."""
        embeddings = []
        for name, e in self.embedders.items():
            emb = e(x[self.offsets[name]:self.offsets[name] + e.in_dim])
            if name == 'fx1':
                emb = emb + self.fx_col_position(jnp.array(0))
            elif name == 'fx2':
                emb = emb + self.fx_col_position(jnp.array(1))
            embeddings.append(emb)
        return jnp.concatenate(embeddings)

    def __call__(self, x):
        steps = x.reshape(STEPS_PER_TABLE, -1)
        step_embs = jax.vmap(self._embed_step)(steps)
        return self.projection(step_embs.reshape(-1))
