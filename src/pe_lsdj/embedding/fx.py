from pe_lsdj.constants import *
from pe_lsdj.embedding.base import (
    ConcatEmbedder,
    EnumEmbedder, 
    EntityEmbedder, 
    GatedNormedEmbedder, 
    SumEmbedder,
)
from jaxtyping import Array, Key
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class GrooveEntityEmbedder(EntityEmbedder):
    def __init__(self, out_dim, key, grooves):
        self.entity_bank = grooves
        self.embedder = GatedNormedEmbedder(
            out_dim,
            key,
            STEPS_PER_GROOVE * 2,
            0, 
            255,
        )


class TableFXValueEmbedder(SumEmbedder):
    """
    Embed FX values for use insinde tables,
    without recursively calling table embedding.
    """
    def __init__(
        self, 
        out_dim: int,
        key: Key,
        groove_embedder: GrooveEntityEmbedder,
    ):
        self.out_dim = out_dim

        PARAMS = {
            "hop": (1, 0, 255),
            "chord": (2, 0, 0x0F), # (semitone 1, semitone 2)
            "env": (2, 0, 0x0F), # (volume, fade)
            "retrig": (2, 0, 0x0F), # (fade, rate)
            "vibrato": (2, 0, 0x0F), # (speed, depth)
            "volume": (1, 0, 255),
            "random": (2, 0, 0x0F), # (L digit, R digit)
            "continuous": (1, 0, 255) # Continuous byte
        }
        pan_key, wave_key, *keys = jr.split(key, 9)

        pan_embedder = EnumEmbedder(
            vocab_size=4,
            out_dim=out_dim,
            key=pan_key,
        )
        wave_embedder = EnumEmbedder(
            vocab_size=4,
            out_dim=out_dim,
            key=wave_key,
        )

        embedders = {
            "groove_embedder": groove_embedder,
            "pan_embedder": pan_embedder,
            "wave_embedder": wave_embedder,
        }

        for idx, (name, params) in enumerate(PARAMS.items()):
            embedders[f"{name}_embedder"] = GatedNormedEmbedder(
                out_dim,
                keys[idx],
                params[0],
                params[1],
                params[2],
            )
        
        self.embedders = embedders


class FXEmbedder(ConcatEmbedder):
    """
    Combined FX cmd / FX value embedder.
    Abstracts over TableFXEmbedder (which doesn't use a Table embedder internally)
    and PhraseFXEmbedder (which does).
    """
    def __init__(
        self,
        key: Key,
        fx_value_embedder: TableFXValueEmbedder,
        out_dim: int=128,
        cmd_out_dim: int=32,
    ):
        k1, k2 = jr.split(key)

        fx_cmd_embedder = EnumEmbedder(
            19,
            cmd_out_dim,
            k1
        )

        embedders = {
            "fx_cmd": fx_cmd_embedder,
            "fx_value": fx_value_embedder,
        }
        super().__init__(k2, embedders, out_dim)


class TableFXEmbedder(FXEmbedder):
    def __init__(
        self,
        key: Key,
        groove_embedder: GrooveEntityEmbedder,
        out_dim: int=128,
        value_out_dim:int=64,
        cmd_out_dim: int=32,
    ):
        k1, k2 = jr.split(key)
        fx_value_embedder = TableFXValueEmbedder(
            value_out_dim, 
            k1, 
            groove_embedder,
        )
        super().__init__(
            k2,
            fx_value_embedder,
            out_dim,
            cmd_out_dim,            
        )


class TableEmbedder(ConcatEmbedder):
    fx_col_position: EnumEmbedder  # distinguishes FX1 from FX2

    def __init__(self, out_dim, key, table_fx_embedder):
        PARAMS = {
            "env_volume": (1, 0, 0x0F),
            "env_duration": (1, 0, 0x0F),
            "transpose": (1, 0, 255),
        }

        keys = jr.split(key, 5)

        embedders = {
            name: GatedNormedEmbedder(
                out_dim,
                keys[idx],
                params[0],
                params[1],
                params[2],
            )
            for idx, (name, params) in enumerate(PARAMS.items())
        }

        # Shared FX embedder for both columns
        embedders["fx1"] = table_fx_embedder
        embedders["fx2"] = table_fx_embedder

        # Learned position embedding to distinguish FX1 (0) from FX2 (1)
        self.fx_col_position = EnumEmbedder(2, table_fx_embedder.out_dim, keys[3])

        super().__init__(keys[4], embedders, out_dim)

    def __call__(self, **kwargs):
        embs = []
        for k, e in self.embedders.items():
            emb = e(kwargs[k])
            if k == "fx1":
                emb = emb + self.fx_col_position(jnp.array(0))
            elif k == "fx2":
                emb = emb + self.fx_col_position(jnp.array(1))
            embs.append(emb)
        return self.projection(jnp.concatenate(embs))


class TableEntityEmbedder(EntityEmbedder):
    def __init__(
        self,
        out_dim: int,
        key: Key,
        tables: Array,
        table_fx_embedder: FXEmbedder
    ):
        self.entity_bank = tables
        self.embedder = TableEmbedder(
            out_dim,
            key,
            table_fx_embedder,
        )


class PhraseFXValueEmbedder(SumEmbedder):
    """FX value embedder for phrases: reuses table FX value weights + adds table entity lookup."""
    def __init__(
        self,
        table_fx_value_embedder: TableFXValueEmbedder,
        table_entity_embedder: TableEntityEmbedder,
    ):
        self.out_dim = table_fx_value_embedder.out_dim
        self.embedders = {
            **table_fx_value_embedder.embedders,
            "table_embedder": table_entity_embedder,
        }


class PhraseFXEmbedder(FXEmbedder):
    def __init__(
        self,
        key: Key,
        table_fx_value_embedder: TableFXValueEmbedder,
        table_entity_embedder: TableEntityEmbedder,
        out_dim: int=128,
        cmd_out_dim: int=32,
    ):
        fx_value_embedder = PhraseFXValueEmbedder(
            table_fx_value_embedder,
            table_entity_embedder,
        )
        super().__init__(
            key,
            fx_value_embedder,
            out_dim,
            cmd_out_dim,
        )
