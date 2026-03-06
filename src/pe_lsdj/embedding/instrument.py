from pe_lsdj.constants import *
from pe_lsdj.embedding.base import (
    BaseEmbedder,
    ConcatEmbedder,
    EnumEmbedder,
    EntityEmbedder,
    EntityType,
    GatedNormedEmbedder,
)
from jaxtyping import Array, Key
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class WaveframeEmbedder(BaseEmbedder):
    linear: eqx.nn.Linear

    def __init__(self, key: Key, out_dim: int):
        self.in_dim = WAVEFRAME_DIM
        self.out_dim = out_dim
        self.linear = eqx.nn.Linear(
            in_features=WAVEFRAME_DIM,
            out_features=out_dim,
            use_bias=False,
            key=key,
        )

    def __call__(self, x, _banks=None):
        return self.linear(x)


class SynthWavesEmbedder(ConcatEmbedder):
    """
    Embeds a synth_waves bank row (SOFTSYNTH_WIDTH + WAVEFRAME_DIM).

    Structured as a ConcatEmbedder whose sub-embedders cover the
    softsynth parameter fields followed by a WaveframeEmbedder.
    The alphabetical sort of 'waveframes' places it after 'waveform',
    so ConcatEmbedder offsets align naturally with the synth_waves layout:
      [softsynth_params (0..SOFTSYNTH_WIDTH-1) | waveframe_data (SOFTSYNTH_WIDTH..)]
    """
    def __init__(
        self,
        key,
        out_dim=64,
        enum_out_dim=32,
        sound_param_out_dim=16,
        continuous_out_dim=16,
        waveframe_out_dim=32,
    ):
        keys = jr.split(key, 13)
        ki = iter(range(13))

        sound_param_embedder = ConcatEmbedder(
            keys[next(ki)],
            {
                'volume': GatedNormedEmbedder(sound_param_out_dim, keys[next(ki)]),
                'cutoff': GatedNormedEmbedder(sound_param_out_dim, keys[next(ki)]),
                'phase':  GatedNormedEmbedder(sound_param_out_dim, keys[next(ki)]),
                'vshift': GatedNormedEmbedder(sound_param_out_dim, keys[next(ki)]),
            },
            continuous_out_dim,
        )

        embedders = {
            'waveform':         EnumEmbedder(4, enum_out_dim, keys[next(ki)]),
            'filter_type':      EnumEmbedder(5, enum_out_dim, keys[next(ki)]),
            'filter_resonance': GatedNormedEmbedder(enum_out_dim, keys[next(ki)]),
            'distortion':       EnumEmbedder(3, enum_out_dim, keys[next(ki)]),
            'phase_type':       EnumEmbedder(4, enum_out_dim, keys[next(ki)]),
            'start_params':     sound_param_embedder,
            'end_params':       sound_param_embedder,  # shared
            # 'waveframes' sorts after 'waveform' — offset lands at SOFTSYNTH_WIDTH
            'waveframes':       WaveframeEmbedder(keys[next(ki)], waveframe_out_dim),
        }

        super().__init__(keys[next(ki)], embedders, out_dim)


class SynthWavesEntityEmbedder(EntityEmbedder):
    def __init__(
        self,
        key: Key,
        out_dim: int = 64,
        enum_out_dim: int = 32,
        sound_param_out_dim: int = 16,
        continuous_out_dim: int = 16,
        waveframe_out_dim: int = 32,
    ):
        super().__init__(
            EntityType.SYNTH_WAVES,
            SynthWavesEmbedder(
                key, out_dim, enum_out_dim, sound_param_out_dim,
                continuous_out_dim, waveframe_out_dim,
            ),
        )


class InstrumentEmbedder(ConcatEmbedder):
    """
    Type-specific fields (e.g. KIT-only) are zeroed for inapplicable
    types, handled by GatedNormedEmbedder's gate and EnumEmbedder's
    null position (index 0).

    Entity references (TABLE, SYNTH_WAVE) look up their banks at call
    time from the SongBanks argument.
    """
    def __init__(
        self,
        key: Key,
        table_entity_embedder: EntityEmbedder,
        synth_waves_entity_embedder: SynthWavesEntityEmbedder,
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
            'type_id':        _enum(5),
            'table':          table_entity_embedder,
            'table_on_off':   _enum(2),
            'table_automate': _enum(2),
            'automate_2':     _enum(2),
            'pan':            _enum(5),
            # --- All but Noise ---
            'vibrato_type':      _enum(5),
            'vibrato_direction': _enum(3),
            # --- Pulse / Noise ---
            'env_volume':    _gated(0x0F),
            'env_fade':      _gated(0x0F),
            'length':        _gated(0x3F),
            'length_limited': _enum(3),
            'sweep':         _gated(),
            # --- WAV / KIT ---
            'volume': _enum(5),
            # --- Pulse only ---
            'phase_transpose': _gated(),
            'wave':            _enum(5),
            'phase_finetune':  _gated(0x0F),
            # --- WAV only: softsynth params + waveframes (unified lookup) ---
            'synth_wave': synth_waves_entity_embedder,
            # --- WAV only (remaining) ---
            'repeat':     _gated(0x0F),
            'play_type':  _enum(5),
            'wave_length': _gated(0x0F),
            'speed':      _gated(0x0F),
            # --- KIT only ---
            'keep_attack_1': _enum(3),
            'keep_attack_2': _enum(3),
            'kit_1_id':      _gated(0x3F),
            'kit_2_id':      _gated(0x3F),
            'length_kit_1':  _gated(),
            'length_kit_2':  _gated(),
            'loop_kit_1':    _enum(3),
            'loop_kit_2':    _enum(3),
            'offset_kit_1':  _gated(),
            'offset_kit_2':  _gated(),
            'half_speed':    _enum(3),
            'pitch':         _gated(),
            'distortion_type': _enum(5),
        }

        proj_key = keys[next(ki)]
        super().__init__(proj_key, embedders, out_dim)


class InstrumentEntityEmbedder(EntityEmbedder):
    def __init__(
        self,
        key: Key,
        table_entity_embedder: EntityEmbedder,
        synth_waves_entity_embedder: SynthWavesEntityEmbedder,
        out_dim: int = 128,
    ):
        super().__init__(
            EntityType.INSTRUMENTS,
            InstrumentEmbedder(
                key,
                table_entity_embedder,
                synth_waves_entity_embedder,
                out_dim=out_dim,
            ),
        )
