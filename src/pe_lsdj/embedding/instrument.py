from pe_lsdj.constants import *
from pe_lsdj.embedding.base import (
    BaseEmbedder,
    ConcatEmbedder,
    EnumEmbedder,
    EntityEmbedder,
    GatedNormedEmbedder,
)
from jaxtyping import Array, Key
import jax.numpy as jnp
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
        keys = jr.split(key, 11)

        sound_param_embedder = ConcatEmbedder(
            keys[4],
            [
                GatedNormedEmbedder(sound_param_out_dim, keys[0]),  # volume
                GatedNormedEmbedder(sound_param_out_dim, keys[1]),  # cutoff
                GatedNormedEmbedder(sound_param_out_dim, keys[2]),  # phase
                GatedNormedEmbedder(sound_param_out_dim, keys[3]),  # vshift
            ],
            continuous_out_dim,
        )

        embedders = [
            EnumEmbedder(4, enum_out_dim, keys[5]),    # waveform
            EnumEmbedder(5, enum_out_dim, keys[6]),    # filter_type
            GatedNormedEmbedder(enum_out_dim, keys[10]),  # filter_resonance
            EnumEmbedder(3, enum_out_dim, keys[7]),    # distortion
            EnumEmbedder(4, enum_out_dim, keys[8]),    # phase_type
            sound_param_embedder,                       # start_params
            sound_param_embedder,                       # end_params (shared)
        ]

        super().__init__(keys[9], embedders, out_dim)


class SoftsynthEntityEmbedder(EntityEmbedder):
    def __init__(
        self,
        key,
        softsynths: Array,
        out_dim: int=64,
        enum_out_dim=32,
        sound_param_out_dim=16,
        continuous_out_dim=16,
        **kwargs
    ):
        super().__init__(
            softsynths,
            SoftsynthEmbedder(
                key,
                enum_out_dim,
                sound_param_out_dim,
                continuous_out_dim,
                out_dim,
            ),
            **kwargs
        )


class WaveframeEmbedder(BaseEmbedder):
    linear: eqx.nn.Linear

    def __init__(self, key: Key, out_dim: int):
        self.in_dim = WAVES_PER_SYNTH * FRAMES_PER_WAVE
        self.out_dim = out_dim
        self.linear = eqx.nn.Linear(
            in_features=WAVES_PER_SYNTH * FRAMES_PER_WAVE,
            out_features=out_dim,
            use_bias=False,
            key=key,
        )

    def __call__(self, x):
        return self.linear(x)


class WaveFrameEntityEmbedder(EntityEmbedder):
    def __init__(self, key, waveframes: Array, out_dim: int, **kwargs):
        super().__init__(
            waveframes,
            WaveframeEmbedder(key, out_dim),
            null_entry=True,  # 0 = null (non-WAV instruments)
        )


class InstrumentEmbedder(ConcatEmbedder):
    """
    Type-specific fields (e.g. KIT-only) are zeroed for inapplicable 
    types, handled by GatedNormedEmbedder's gate and EnumEmbedder's 
    null position (index 0).

    Entity references (TABLE, SOFTSYNTH_ID) are passed in from outside
    since they depend on entity banks constructed at the song level.
    """
    def __init__(
        self,
        key: Key,
        table_entity_embedder: EntityEmbedder,
        softsynth_entity_embedder: SoftsynthEntityEmbedder,
        waveframe_entity_embedder: WaveFrameEntityEmbedder,
        enum_out_dim: int = 16,
        gated_out_dim: int = 16,
        out_dim: int = 128,
    ):
        # 33 embedders + 1 projection key = 34
        keys = jr.split(key, 34)
        ki = iter(range(34))

        def _enum(vocab_size):
            return EnumEmbedder(vocab_size, enum_out_dim, keys[next(ki)])

        def _gated(max_value=255):
            return GatedNormedEmbedder(gated_out_dim, keys[next(ki)],
                                       1, 0, max_value)

        embedders = [
            # --- Universal (all instrument types) ---
            _enum(5),                    # TYPE_ID: PU=1, WAV=2, KIT=3, NOI=4
            table_entity_embedder,       # TABLE
            _enum(2),                    # TABLE_ON_OFF: 0=off, 1=on
            _enum(2),                    # TABLE_AUTOMATE
            _enum(2),                    # AUTOMATE_2
            _enum(5),                    # PAN: 0=null, 1=off, 2=L, 3=R, 4=LR

            # --- All but Noise ---
            _enum(5),                    # VIBRATO_TYPE: 0=null, 1-4
            _enum(3),                    # VIBRATO_DIRECTION: 0=null, 1=down, 2=up

            # --- Pulse / Noise ---
            _gated(0x0F),               # ENV_VOLUME
            _gated(0x0F),               # ENV_FADE
            _gated(0x3F),               # LENGTH
            _enum(3),                    # LENGTH_LIMITED
            _gated(),                    # SWEEP

            # --- WAV / KIT ---
            _enum(5),                    # VOLUME: 0=null, 1-4

            # --- Pulse only ---
            _gated(),                    # PHASE_TRANSPOSE
            _enum(5),                    # WAVE: 0=null, 1-4: 12.5/25/50/75%
            _gated(0x0F),               # PHASE_FINETUNE

            # --- WAV only ---
            softsynth_entity_embedder,   # SOFTSYNTH_ID
            _gated(0x0F),               # REPEAT
            _enum(5),                    # PLAY_TYPE: 0=null, 1-4
            _gated(0x0F),               # WAVE_LENGTH
            _gated(0x0F),               # SPEED

            # --- KIT only ---
            _enum(3),                    # KEEP_ATTACK_1
            _enum(3),                    # KEEP_ATTACK_2
            _gated(0x3F),               # KIT_1_ID
            _gated(0x3F),               # KIT_2_ID
            _gated(),                    # LENGTH_KIT_1
            _gated(),                    # LENGTH_KIT_2
            _enum(3),                    # LOOP_KIT_1
            _enum(3),                    # LOOP_KIT_2
            _gated(),                    # OFFSET_KIT_1
            _gated(),                    # OFFSET_KIT_2
            _enum(3),                    # HALF_SPEED
            _gated(),                    # PITCH
            _enum(5),                    # DISTORTION_TYPE: 0=null, 1-4

            # Append waveframes for insrument ID (if WAV)
            waveframe_entity_embedder,   # WAVEFRAME_ID
        ]

        proj_key = keys[next(ki)]
        super().__init__(proj_key, embedders, out_dim)


class InstrumentEntityEmbedder(EntityEmbedder):
    # Column index of SOFTSYNTH_ID in the instrument token array.
    # Duplicated as the waveframe reference (last column) at construction time.
    SOFTSYNTH_COL = 17

    def __init__(
        self,
        key: Key,
        instruments: Array,
        table_entity_embedder: EntityEmbedder,
        softsynth_entity_embedder: SoftsynthEntityEmbedder,
        waveframe_entity_embedder: WaveFrameEntityEmbedder,
        out_dim: int = 128,
    ):
        # Append softsynth_id as waveframe reference column.
        # WAV instruments map to the correct waveframe bank;
        # non-WAV have softsynth_id=0 → null via WaveFrameEntityEmbedder.
        waveframe_ref = instruments[:, self.SOFTSYNTH_COL : self.SOFTSYNTH_COL + 1]
        instruments_aug = jnp.concatenate([instruments, waveframe_ref], axis=1)

        super().__init__(
            entity_bank = instruments_aug,
            embedder = InstrumentEmbedder(
                key,
                table_entity_embedder,
                softsynth_entity_embedder,
                waveframe_entity_embedder,
                out_dim=out_dim,
            )
        )
