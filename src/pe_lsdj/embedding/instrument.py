from pe_lsdj.constants import *
from pe_lsdj.embedding import (
    BaseEmbedder,
    ConcatEmbedder,
    SumEmbedder,
    EnumEmbedder, 
    EntityEmbedder, 
    GatedNormedEmbedder,
    TableEmbedder,
)
from jaxtyping import Array, Key
import jax.random as jr
import equinox as eqx



class SoftsynthEmbedder(ConcatEmbedder):
    def __init__(
        self, 
        key, 
        enum_out_dim=32,
        sound_param_out_dim=16, 
        continuous_out_dim=16,
        out_dim=64,
    ):
        keys = jr.split(key, 9)

        param_embedders = [
            GatedNormedEmbedder(sound_param_out_dim, keys[0]), # volume
            GatedNormedEmbedder(sound_param_out_dim, keys[1]), # cutoff
            GatedNormedEmbedder(sound_param_out_dim, keys[2]), # phase
            GatedNormedEmbedder(sound_param_out_dim, keys[3]), # v.shift
        ]

        sound_param_embedder = ConcatEmbedder(
            keys[4],
            param_embedders,
            continuous_out_dim,
        )

        embedders = {
            "waveform": EnumEmbedder(
                4,
                enum_out_dim,
                keys[5]
            ),
            "filter_type": EnumEmbedder(
                5,
                enum_out_dim,
                keys[6]
            ),
            "distortion": EnumEmbedder(
                3,
                enum_out_dim,
                keys[7]
            ),
            "phase_type": EnumEmbedder(
                4,
                enum_out_dim,
                keys[8]
            ),
            "start_params": sound_param_embedder,
            "end_params": sound_param_embedder,
        }

        super().__init__(keys[9], embedders, out_dim)


class SoftsynthEntityEmbedder(EntityEmbedder):
    def __init__(self, key, softsynths: Array):
        self.entity_bank = softsynths
        self.embedder = SoftsynthEmbedder(key)


class WaveframeEmbedder(BaseEmbedder):
    # This could just be a continuous embedder...?
    # Waveframes will be part of the instrument definition,
    # but differ from other entity collections in that they're 
    # always defined, no alloc table
    linear: eqx.nn.Linear

    def __init__(self, key: Key, out_dim: int):
        self.linear = eqx.nn.Linear(
            in_features=WAVES_PER_SYNTH * FRAMES_PER_WAVE,
            out_features=out_dim,
            use_bias=False,
            key=key,

        )

    def __call__(self, x):
        return self.linear(x)
    

class WaveFrameEntityEmbedder(EntityEmbedder):
    # TODO
    pass


class InstrumentEmbedder(ConcatEmbedder):
    def __init__(
        self, 
        key,
        table_embedder: TableEmbedder,
        enum_out_dim=32,
        continuous_out_dim=16,
        out_dim=128,
    ):
        keys = jr.split(key, 9) #?

        sparse_feat_embedder = (
            embedders = 

        )

        embedders = {
            TYPE_ID: EnumEmbedder(
                5,
                enum_out_dim,
                keys[0]
            ),
            TABLE: table_embedder,
            TABLE_ON_OFF: EnumEmbedder(
                2,
                enum_out_dim,
                keys[1]
            ),
            TABLE_AUTOMATE: EnumEmbedder(
                2,
                enum_out_dim,
                keys[2]
            ),
            AUTOMATE_2: EnumEmbedder(
                2,
                enum_out_dim,
                keys[2]
            ),
            PAN: EnumEmbedder(
                5,
                enum_out_dim,
                keys[3]
            )
        }

        super().__init__(keys[9], embedders, out_dim)    


class InstrumentEntityEmbedder(EntityEmbedder):
    # TODO
    pass

