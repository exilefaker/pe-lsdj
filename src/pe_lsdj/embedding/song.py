import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from pe_lsdj.constants import *
from pe_lsdj.embedding.base import (
    DummyEmbedder,
    EntityEmbedder,
    GatedNormedEmbedder,
)
from pe_lsdj.embedding.fx import (
    GrooveEntityEmbedder,
    build_fx_value_embedders,
    FXValueEmbedder,
    FXEmbedder,
    TableEmbedder,
)
from pe_lsdj.embedding.instrument import (
    InstrumentEntityEmbedder,
    SoftsynthEntityEmbedder,
    WaveFrameEntityEmbedder,
)
from pe_lsdj.embedding.position import (
    SinusoidalPositionEncoding,
    PhrasePositionEmbedder,
    ChannelPositionEmbedder,
)
from jaxtyping import Array, Key


class SongStepEmbedder(eqx.Module):
    """
    Embedder for one step of an LSDJ track:
    [Pulse 1 | Pulse 2 | Wav | Noise]
        |
        |__> | Note | Instrument | FX | Transpose | (per channel)
                          |        |
                         ...      ...

    Channels share sub-embedders (note, instrument, FX, transpose).
    Each channel gets its own learned linear projection.

    FX value embedding uses three different table embedders:
      Base: DummyEmbedder for use inside TableEmbedder definition
            (no A commands possible here)
      Within-table: EntityEmbedder over traces bank for TABLE_FX
            (tables can be used, but not recursively)
      Phrase/instrument-level:  EntityEmbedder over tables bank
            (full table representation)

    Output: (4 * per_ch_dim,) = (out_dim,)
    """
    note_embedder: GatedNormedEmbedder
    instrument_embedder: InstrumentEntityEmbedder
    fx_embedder: FXEmbedder
    transpose_embedder: GatedNormedEmbedder

    # (4, per_ch_dim, concat_dim) — one projection per channel
    channel_projections: Array

    per_ch_dim: int
    out_dim: int

    def __init__(
        self,
        key: Key,
        instruments: Array,
        softsynths: Array,
        waveframes: Array,
        grooves: Array,
        tables: Array,
        traces: Array,
        out_dim: int = 512,
        note_dim: int = 128,
        instr_dim: int = 128,
        fx_dim: int = 128,
        table_dim: int = 64,
        transpose_dim: int = 16,
        value_out_dim: int = 64,
        softsynth_dim: int = 64,
        waveframe_dim: int = 32,
    ):
        assert out_dim % 4 == 0, f"out_dim must be divisible by 4, got {out_dim}"
        per_ch_dim = out_dim // 4
        self.per_ch_dim = per_ch_dim

        keys = jr.split(key, 14)

        # --- Note ---
        self.note_embedder = GatedNormedEmbedder(
            note_dim,
            keys[0],
            max_value=NUM_NOTES,
        )

        # --- Shared FX value sub-embedders (positions 1..11) ---
        groove_embedder = GrooveEntityEmbedder(
            value_out_dim,
            keys[1],
            grooves,
            null_entry=True,
        )
        shared_embedders = build_fx_value_embedders(value_out_dim, keys[2], groove_embedder)

        # --- Tier 0: base table embedder (DummyEmbedder for TABLE_FX) ---
        dummy_table_fx = DummyEmbedder(1, value_out_dim)
        fxv0 = FXValueEmbedder(dummy_table_fx, shared_embedders)
        fx0 = FXEmbedder(keys[3], fxv0, fx_dim)
        table_embedder_0 = TableEmbedder(table_dim, keys[4], fx0)

        # --- Tier 1: trace entity for TABLE_FX ---
        trace_embedder = EntityEmbedder(traces, table_embedder_0)
        fxv1 = FXValueEmbedder(trace_embedder, shared_embedders)
        fx1 = FXEmbedder(keys[5], fxv1, fx_dim, _projection=fx0.projection)
        table_table_embedder = TableEmbedder(table_dim, keys[6], fx1, _projection=table_embedder_0.projection)

        # --- Phrase/instr level: table entity for TABLE_FX ---
        table_embedder = EntityEmbedder(tables, table_table_embedder)
        fxv_phrase = FXValueEmbedder(table_embedder, shared_embedders)
        self.fx_embedder = FXEmbedder(
            keys[7], fxv_phrase, fx_dim, _projection=fx0.projection,
        )

        # --- Instrument ---
        softsynth_embedder = SoftsynthEntityEmbedder(
            keys[8],
            softsynths,
            softsynth_dim,
            null_entry=True,
        )

        waveframe_embedder = WaveFrameEntityEmbedder(
            keys[9],
            waveframes,
            waveframe_dim,
            null_entry=True,
        )

        self.instrument_embedder = InstrumentEntityEmbedder(
            keys[10],
            instruments,
            table_embedder,
            softsynth_embedder,
            waveframe_embedder,
            instr_dim,
        )

        # --- Transpose ---
        self.transpose_embedder = GatedNormedEmbedder(
            transpose_dim,
            keys[11],
        )

        # --- Per-channel projections (stacked for vmap) ---
        concat_dim = note_dim + instr_dim + fx_dim + transpose_dim

        proj_keys = jr.split(keys[12], 4)
        self.channel_projections = jnp.stack([
            jr.normal(k, (per_ch_dim, concat_dim)) / jnp.sqrt(concat_dim)
            for k in proj_keys
        ])  # (4, per_ch_dim, concat_dim)

        self.out_dim = out_dim

    def _embed_one_channel(self, note, instr_id, fx_data, transpose):
        """Shared sub-embedders → concatenated feature vector."""
        return jnp.concatenate([
            self.note_embedder(note),
            self.instrument_embedder(instr_id),
            self.fx_embedder(fx_data),
            self.transpose_embedder(transpose),
        ])  # (concat_dim,)

    def __call__(self, step):
        """
        step: (4, 21) — one timestep across all channels
        Returns: (4, per_ch_dim) — structured per-channel embeddings
        """
        # Embed each channel with shared weights → (4, concat_dim)
        channel_embs = jnp.stack([
            self._embed_one_channel(
                step[ch, 0:1],     # note
                step[ch, 1:2],     # instr_id
                step[ch, 2:20],    # fx_cmd + fx_vals (18 values)
                step[ch, 20:21],   # transpose
            )
            for ch in range(4)
        ])

        # Per-channel projection via vmap: (4, per_ch_dim)
        return jax.vmap(jnp.dot)(self.channel_projections, channel_embs)


class SequenceEmbedder(eqx.Module):
    """
    Full sequence embedding: content + positional encodings.

    Input:  song_tokens (S, 4, 21)
    Output: (S, 4, per_ch_dim)
    """
    step_embedder: SongStepEmbedder
    global_position: SinusoidalPositionEncoding
    phrase_position: PhrasePositionEmbedder
    channel_position: ChannelPositionEmbedder

    def __init__(self, step_embedder, key):
        k1, k2 = jr.split(key)
        self.step_embedder = step_embedder
        d = step_embedder.per_ch_dim
        self.global_position = SinusoidalPositionEncoding(d)
        self.phrase_position = PhrasePositionEmbedder(d, k1)
        self.channel_position = ChannelPositionEmbedder(d, k2)

    def __call__(self, song_tokens):
        S = song_tokens.shape[0]
        content = jax.vmap(self.step_embedder)(song_tokens)   # (S, 4, d)
        global_pos = self.global_position(jnp.arange(S))      # (S, d)
        phrase_pos = self.phrase_position(
            jnp.arange(S) % STEPS_PER_PHRASE
        )                                                     # (S, d)
        channel_pos = self.channel_position()                 # (4, d)
        return (
            content
            + global_pos[:, None, :]
            + phrase_pos[:, None, :]
            + channel_pos[None, :, :]
        )
