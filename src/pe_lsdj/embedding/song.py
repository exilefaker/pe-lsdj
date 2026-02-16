import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from pe_lsdj.constants import *
from pe_lsdj.embedding.base import BaseEmbedder, GatedNormedEmbedder
from pe_lsdj.embedding.fx import (
    GrooveEntityEmbedder,
    PhraseFXEmbedder,
    TableFXValueEmbedder,
    FXEmbedder,
    TableEntityEmbedder,
)
from pe_lsdj.embedding.instrument import (
    InstrumentEntityEmbedder,
    SoftsynthEntityEmbedder,
    WaveFrameEntityEmbedder,
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
    Each channel gets its own learned linear projection via a stacked
    weight array and vmap'd matmul.

    Output: (4 * per_ch_dim,) = (out_dim,)
    """
    note_embedder: GatedNormedEmbedder
    instrument_embedder: InstrumentEntityEmbedder
    fx_embedder: PhraseFXEmbedder
    transpose_embedder: GatedNormedEmbedder

    # (4, per_ch_dim, concat_dim) — one projection per channel
    channel_projections: Array

    out_dim: int

    def __init__(
        self,
        key: Key,
        instruments: Array,
        softsynths: Array,
        waveframes: Array,
        grooves: Array,
        tables: Array,
        out_dim: int = 512,
        note_dim: int = 128,
        instr_dim: int = 128,
        fx_dim: int = 128,
        table_dim: int = 64,
        transpose_dim: int = 16,
        groove_dim: int = 64,
        softsynth_dim: int = 64,
        waveframe_dim: int = 32,
    ):
        assert out_dim % 4 == 0, f"out_dim must be divisible by 4, got {out_dim}"
        per_ch_dim = out_dim // 4

        keys = jr.split(key, 11)

        # --- Shared sub-embedders ---
        self.note_embedder = GatedNormedEmbedder(
            note_dim,
            keys[0],
            max_value=NUM_NOTES,
        )

        groove_embedder = GrooveEntityEmbedder(
            groove_dim,
            keys[1],
            grooves,
        )

        table_fx_value_embedder = TableFXValueEmbedder(
            64,
            keys[2],
            groove_embedder,
        )

        table_fx_embedder = FXEmbedder(
            keys[3],
            table_fx_value_embedder,
            fx_dim,
        )

        table_embedder = TableEntityEmbedder(
            table_dim,
            keys[4],
            tables,
            table_fx_embedder=table_fx_embedder,
        )

        softsynth_embedder = SoftsynthEntityEmbedder(
            keys[5],
            softsynths,
            softsynth_dim,
        )

        waveframe_embedder = WaveFrameEntityEmbedder(
            keys[6],
            waveframes,
            waveframe_dim,
        )

        self.instrument_embedder = InstrumentEntityEmbedder(
            keys[7],
            instruments,
            table_embedder,
            softsynth_embedder,
            waveframe_embedder,
            instr_dim,
        )

        self.fx_embedder = PhraseFXEmbedder(
            keys[8],
            table_fx_value_embedder,
            table_embedder,
            fx_dim,
        )

        self.transpose_embedder = GatedNormedEmbedder(
            transpose_dim,
            keys[9],
        )

        # --- Per-channel projections (stacked for vmap) ---
        concat_dim = note_dim + instr_dim + fx_dim + transpose_dim

        proj_keys = jr.split(keys[10], 4)
        self.channel_projections = jnp.stack([
            jr.normal(k, (per_ch_dim, concat_dim)) / jnp.sqrt(concat_dim)
            for k in proj_keys
        ])  # (4, per_ch_dim, concat_dim)

        self.out_dim = out_dim

    def _embed_one_channel(self, note, instr_id, fx_cmd, fx_vals, transpose):
        """Shared sub-embedders → concatenated feature vector."""
        return jnp.concatenate([
            self.note_embedder(note),
            self.instrument_embedder(instr_id),
            self.fx_embedder(fx_cmd=fx_cmd, fx_value=fx_vals),
            self.transpose_embedder(transpose),
        ])  # (concat_dim,)

    def __call__(self, step):
        """
        step: (4, 21) — one timestep across all channels
        Returns: (out_dim,) where out_dim = 4 * per_ch_dim
        """
        # Embed each channel with shared weights → (4, concat_dim)
        channel_embs = jnp.stack([
            self._embed_one_channel(
                step[ch, 0],       # note
                step[ch, 1],       # instr_id
                step[ch, 2],       # fx_cmd
                step[ch, 3:20],    # fx_vals (17 values)
                step[ch, 20],      # transpose
            )
            for ch in range(4)
        ])

        # Per-channel projection via vmap: (4, per_ch_dim)
        projected = jax.vmap(jnp.dot)(self.channel_projections, channel_embs)

        return projected.reshape(-1)  # (4 * per_ch_dim,) = (out_dim,)
