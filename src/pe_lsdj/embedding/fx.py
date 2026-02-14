from pe_lsdj.constants import *
from pe_lsdj.embedding.base import EnumEmbedder, EntityEmbedder, GatedNormedEmbedder
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx


class GrooveEntityEmbedder(EntityEmbedder):
    def __init__(self, d_model, key, grooves):
        self.entity_bank = grooves
        self.embedder = GatedNormedEmbedder(
            d_model,
            key,
            STEPS_PER_GROOVE * 2,
            0, 
            255,
        )


class TableEmbedder(eqx.Module):
    # TODO
    # Tricky: This should include an FXEmbedder (see below)
    fx_embedder: "FXEmbedder"
    pass


class TableEntityEmbedder(EntityEmbedder):
    pass


# TODO subclass SumEmbedder for part of this?
class FXEmbedder(eqx.Module):
    table_embedder: TableEntityEmbedder
    groove_embedder: GrooveEntityEmbedder
    hop_embedder: GatedNormedEmbedder # Or Enum?
    pan_embedder: EnumEmbedder
    chord_embedder: GatedNormedEmbedder
    env_embedder: GatedNormedEmbedder
    retrig_embedder: GatedNormedEmbedder
    vibrato_embedder: GatedNormedEmbedder
    volume_embdder: GatedNormedEmbedder
    wave_embedder: EnumEmbedder
    random_embedder: GatedNormedEmbedder

    fx_cmd_embedder: EnumEmbedder
    continuous_embedder: GatedNormedEmbedder

    def __init__(
        self, 
        d_model,
        key,
        table_embedder,
        groove_embedder,
    ):
        
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
        pan_key, wave_key, fx_cmd_key, *keys = jr.split(key, 11)

        self.table_embedder = table_embedder # TODO
        self.groove_embedder = groove_embedder
        self.pan_embedder = EnumEmbedder(
            vocab_size=4,
            d_model=d_model,
            key=pan_key,
        )
        self.wave_embedder = EnumEmbedder(
            vocab_size=4,
            d_model=d_model,
            key=wave_key,
        )
        self.fx_cmd_embedder = EnumEmbedder(
            vocab_size=8, 
            d_model=d_model, 
            key=fx_cmd_key,
        )

        for idx, (attr_name, params) in enumerate(PARAMS.items()):
            setattr(self, f"{attr_name}_embedder", GatedNormedEmbedder(
                d_model,
                keys[idx],
                params[0],
                params[1],
                params[2],
            ))
    
    def __call__(self, x, fx_cmd_reduced):
        """
        Combine embeddings. 
        TODO: Decide: Use fx_cmd_reduced to determine which 
        continuous embedder to use (?)
        ...or, use a shared encoder (as here)

        If the former, should be fed in as one-hot to allow for
        "soft-hot" input        
        """

        embeddings = [
            self.table_embedder(x[0]),
            self.groove_embedder(x[1]),
            self.hop_embedder(x[2]),
            self.pan_embedder(x[3]),
            self.chord_embedder(x[4:6]),
            self.env_embedder(x[6:8]),
            self.retrig_embedder(x[8:10]),
            self.vibrato_embedder(x[10:12]),
            self.volume_embedder(x[12]),
            self.wave_embedder(x[13]),
            self.random_embedder(x[14:16]),
            self.fx_cmd_embedder(fx_cmd_reduced),
            self.continuous_embedder(x[16]),
        ]

        # or jnp.stack(embeddings).sum(axis=0)
        return jax.tree.reduce(jnp.add, embeddings)
