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
        out_dim=128,
    ):
        keys = jr.split(key, 11)

        sound_param_embedder = ConcatEmbedder(
            keys[4],
            {
                'volume': GatedNormedEmbedder(sound_param_out_dim, keys[0]),
                'cutoff': GatedNormedEmbedder(sound_param_out_dim, keys[1]),
                'phase': GatedNormedEmbedder(sound_param_out_dim, keys[2]),
                'vshift': GatedNormedEmbedder(sound_param_out_dim, keys[3]),
            },
            continuous_out_dim,
        )

        embedders = {
            'waveform': EnumEmbedder(4, enum_out_dim, keys[5]),
            'filter_type': EnumEmbedder(5, enum_out_dim, keys[6]),
            'filter_resonance': GatedNormedEmbedder(enum_out_dim, keys[10]),
            'distortion': EnumEmbedder(3, enum_out_dim, keys[7]),
            'phase_type': EnumEmbedder(4, enum_out_dim, keys[8]),
            'start_params': sound_param_embedder,
            'end_params': sound_param_embedder,  # shared
        }

        super().__init__(keys[9], embedders, out_dim)


class SoftsynthEntityEmbedder(EntityEmbedder):
    def __init__(
        self,
        key,
        softsynths: Array,
        out_dim: int = 128,
        enum_out_dim=32,
        sound_param_out_dim=16,
        continuous_out_dim=16,
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
            null_entry=False,
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
    def __init__(self, key, waveframes: Array, out_dim: int):
        super().__init__(
            waveframes,
            WaveframeEmbedder(key, out_dim),
            null_entry=False,
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
        keys = jr.split(key, 34)
        ki = iter(range(34))

        def _enum(vocab_size):
            return EnumEmbedder(vocab_size, enum_out_dim, keys[next(ki)])

        def _gated(max_value=255):
            return GatedNormedEmbedder(gated_out_dim, keys[next(ki)],
                                       1, 0, max_value)

        embedders = {
            # --- Universal (all instrument types) ---
            'type_id': _enum(5),
            'table': table_entity_embedder,
            'table_on_off': _enum(2),
            'table_automate': _enum(2),
            'automate_2': _enum(2),
            'pan': _enum(5),
            # --- All but Noise ---
            'vibrato_type': _enum(5),
            'vibrato_direction': _enum(3),
            # --- Pulse / Noise ---
            'env_volume': _gated(0x0F),
            'env_fade': _gated(0x0F),
            'length': _gated(0x3F),
            'length_limited': _enum(3),
            'sweep': _gated(),
            # --- WAV / KIT ---
            'volume': _enum(5),
            # --- Pulse only ---
            'phase_transpose': _gated(),
            'wave': _enum(5),
            'phase_finetune': _gated(0x0F),
            # --- WAV only ---
            'softsynth': softsynth_entity_embedder,
            'repeat': _gated(0x0F),
            'play_type': _enum(5),
            'wave_length': _gated(0x0F),
            'speed': _gated(0x0F),
            # --- KIT only ---
            'keep_attack_1': _enum(3),
            'keep_attack_2': _enum(3),
            'kit_1_id': _gated(0x3F),
            'kit_2_id': _gated(0x3F),
            'length_kit_1': _gated(),
            'length_kit_2': _gated(),
            'loop_kit_1': _enum(3),
            'loop_kit_2': _enum(3),
            'offset_kit_1': _gated(),
            'offset_kit_2': _gated(),
            'half_speed': _enum(3),
            'pitch': _gated(),
            'distortion_type': _enum(5),
            # Append waveframes for instrument ID (if WAV)
            'waveframe': waveframe_entity_embedder,
        }

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
            ),
            null_entry=False,
        )
