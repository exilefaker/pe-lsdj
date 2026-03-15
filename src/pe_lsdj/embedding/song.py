import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import NamedTuple

from pe_lsdj import SongFile
from pe_lsdj.constants import *
from pe_lsdj.embedding.base import (
    DummyEmbedder,
    EntityEmbedder,
    EntityType,
    GatedNormedEmbedder,
    HelixEmbedder,
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
    SynthWavesEntityEmbedder,
)
from pe_lsdj.embedding.position import (
    PhrasePositionEmbedder,
    ChannelPositionEmbedder,
)
from jaxtyping import Array, Key


class SongBanks(NamedTuple):
    """
    All per-song entity banks needed by the embedding pipeline.

    All banks are null-prepended: index 0 is a zero sentinel row so that
    token IDs (0=NULL, 1..N=entity) can index directly without an offset.
    This is consistent across all entity types and simplifies both the
    embedders and the entity parameter loss.
    """
    instruments: Array   # (NUM_INSTRUMENTS + 1, INSTR_WIDTH)
    grooves: Array       # (NUM_GROOVES + 1, STEPS_PER_GROOVE * 2)
    tables: Array        # (NUM_TABLES + 1, TABLE_WIDTH)
    traces: Array        # (NUM_TABLES + 1, TABLE_WIDTH)
    synth_waves: Array   # (NUM_SYNTHS + 1, SOFTSYNTH_WIDTH + WAVES_PER_SYNTH * FRAMES_PER_WAVE)
    instrs_occupied: Array   # (NUM_INSTRUMENTS + 1,) bool — index 0 always False (null sentinel)
    grooves_occupied: Array  # (NUM_GROOVES + 1,) bool
    tables_occupied: Array   # (NUM_TABLES + 1,) bool
    synths_occupied: Array   # (NUM_SYNTHS + 1,) bool

    @classmethod
    def default(cls):
        """Zero-filled banks (null rows included). All slots unoccupied."""
        def _z(n, w):
            return jnp.zeros((n + 1, w), dtype=jnp.uint16)
        def _occ(n):
            return jnp.zeros(n + 1, dtype=jnp.bool_)
        return cls(
            instruments=_z(NUM_INSTRUMENTS, INSTR_WIDTH),
            grooves=_z(NUM_GROOVES, STEPS_PER_GROOVE * 2),
            tables=_z(NUM_TABLES, TABLE_WIDTH),
            traces=_z(NUM_TABLES, TABLE_WIDTH),
            synth_waves=_z(NUM_SYNTHS, SOFTSYNTH_WIDTH + WAVES_PER_SYNTH * FRAMES_PER_WAVE),
            instrs_occupied=_occ(NUM_INSTRUMENTS),
            grooves_occupied=_occ(NUM_GROOVES),
            tables_occupied=_occ(NUM_TABLES),
            synths_occupied=_occ(NUM_SYNTHS),
        )

    @classmethod
    def from_songfile(cls, songfile: SongFile):
        """
        Load banks from a song file.

        Instruments and tables: occupancy from LSDJ's native alloc tables.
        Grooves and synths: no LSDJ alloc table exists; derive from non-zero rows
        (empty slots have all-zero defaults in LSDJ's save format).
        """
        def _prepend(arr):
            return jnp.concatenate(
                [jnp.zeros((1, arr.shape[1]), arr.dtype), arr], axis=0
            )
        def _prepend_bool(arr_1d):
            return jnp.concatenate([jnp.array([False]), arr_1d])

        return cls(
            instruments=_prepend(songfile.instruments_array),
            grooves=_prepend(songfile.grooves_array),
            tables=_prepend(songfile.tables_array),
            traces=_prepend(songfile.traces_array),
            synth_waves=_prepend(
                jnp.concatenate([
                    songfile.softsynths_array,
                    songfile.waveframes_array,
                ], axis=-1)
            ),
            instrs_occupied=_prepend_bool(songfile.instr_alloc),
            tables_occupied=_prepend_bool(songfile.table_alloc),
            grooves_occupied=_prepend_bool(
                jnp.any(songfile.grooves_array != 0, axis=-1)
            ),
            synths_occupied=_prepend_bool(
                jnp.any(songfile.softsynths_array != 0, axis=-1)
                | jnp.any(songfile.waveframes_array != 0, axis=-1)
            ),
        )


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
      Phrase/instrument-level: EntityEmbedder over tables bank
            (full table representation)

    Banks are passed at call time (not stored in the model tree),
    enabling correct per-item batching across songs.

    Output: (4 * per_ch_dim,) = (out_dim,)
    """
    note_embedder: HelixEmbedder
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
        *,
        out_dim: int = 1024,
        note_dim: int = 128,
        instr_dim: int = 128,
        fx_dim: int = 128,
        transpose_dim: int = 16,
        value_out_dim: int = 64,
        synth_waves_dim: int = 64,
    ):
        assert out_dim % 4 == 0, f"out_dim must be divisible by 4, got {out_dim}"
        per_ch_dim = out_dim // 4
        self.per_ch_dim = per_ch_dim
        # NOTE that value_out_dim (the dimension of FX value embeddings before being combined with FX commands)
        # governs the dimensionality of the table and groove representations (since tables and grooves are possible FX values).

        keys = jr.split(key, 12)

        # --- Note ---
        self.note_embedder = HelixEmbedder(
            note_dim,
            keys[0],
            period=12,
            num_values=NUM_NOTES,
        )

        # --- Shared FX value sub-embedders (positions 1..11) ---
        groove_embedder = GrooveEntityEmbedder(value_out_dim, keys[1])
        shared_embedders = build_fx_value_embedders(value_out_dim, keys[2], groove_embedder)

        # --- Tier 0: base table embedder (DummyEmbedder for TABLE_FX) ---
        dummy_table_fx = DummyEmbedder(1, value_out_dim)
        fxv0 = FXValueEmbedder(dummy_table_fx, shared_embedders)
        fx0 = FXEmbedder(keys[3], fxv0, fx_dim)
        table_embedder_0 = TableEmbedder(value_out_dim, keys[4], fx0)

        # --- Tier 1: trace entity for TABLE_FX ---
        trace_embedder = EntityEmbedder(EntityType.TRACES, table_embedder_0)
        fxv1 = FXValueEmbedder(trace_embedder, shared_embedders)
        fx1 = FXEmbedder(keys[5], fxv1, fx_dim, _projection=fx0.projection)
        table_table_embedder = TableEmbedder(value_out_dim, keys[6], fx1, _projection=table_embedder_0.projection)

        # --- Phrase/instr level: table entity for TABLE_FX ---
        table_embedder = EntityEmbedder(EntityType.TABLES, table_table_embedder)
        fxv_phrase = FXValueEmbedder(table_embedder, shared_embedders)
        self.fx_embedder = FXEmbedder(
            keys[7], fxv_phrase, fx_dim, _projection=fx0.projection,
        )

        # --- Instrument ---
        synth_waves_embedder = SynthWavesEntityEmbedder(keys[8], synth_waves_dim)
        self.instrument_embedder = InstrumentEntityEmbedder(
            keys[9],
            table_embedder,
            synth_waves_embedder,
            instr_dim,
        )

        # --- Transpose ---
        self.transpose_embedder = GatedNormedEmbedder(transpose_dim, keys[10])

        # --- Per-channel projections (stacked for vmap) ---
        concat_dim = note_dim + instr_dim + fx_dim + transpose_dim

        proj_keys = jr.split(keys[11], 4)
        self.channel_projections = jnp.stack([
            jr.normal(k, (per_ch_dim, concat_dim)) / jnp.sqrt(concat_dim)
            for k in proj_keys
        ])  # (4, per_ch_dim, concat_dim)

        self.out_dim = out_dim

    def _embed_one_channel(self, note, instr_id, fx_data, transpose, banks):
        """Shared sub-embedders → concatenated feature vector."""
        return jnp.concatenate([
            self.note_embedder(note),
            self.instrument_embedder(instr_id, banks),
            self.fx_embedder(fx_data, banks),
            self.transpose_embedder(transpose),
        ])  # (concat_dim,)

    def __call__(self, step, banks):
        """
        step:  (4, 21) — one timestep across all channels
        banks: SongBanks for the current song
        Returns: (4, per_ch_dim)
        """
        channel_embs = jnp.stack([
            self._embed_one_channel(
                step[ch, 0:1],     # note
                step[ch, 1:2],     # instr_id
                step[ch, 2:20],    # fx_cmd + fx_vals (18 values)
                step[ch, 20:21],   # transpose
                banks,
            )
            for ch in range(4)
        ])

        # Per-channel projection via vmap: (4, per_ch_dim)
        return jax.vmap(jnp.dot)(self.channel_projections, channel_embs)


class SequenceEmbedder(eqx.Module):
    """
    Full sequence embedding: content + positional encodings.

    Input:  song_tokens (S, 4, 21), banks: SongBanks
    Output: (S, 4, per_ch_dim)

    song_length: total length of the source song (not the crop).  Used to
    compute a normalised song-progress signal positions / song_length ∈ [0, 1).
    Defaults to the crop length S when not provided (crop-relative fallback).
    During training, pass the actual song length per item so that the signal
    is consistent across songs and precisely reflects global position.
    """
    step_embedder: SongStepEmbedder
    phrase_position: PhrasePositionEmbedder
    channel_position: ChannelPositionEmbedder
    progress_proj: eqx.nn.Linear

    def __init__(self, step_embedder, key):
        k1, k2, k3 = jr.split(key, 3)
        self.step_embedder = step_embedder
        d = step_embedder.per_ch_dim
        self.phrase_position = PhrasePositionEmbedder(d, k1)
        self.channel_position = ChannelPositionEmbedder(d, k2)
        self.progress_proj = eqx.nn.Linear(1, d, key=k3)

    @classmethod
    def create(cls, key: Key, **step_kwargs):
        k1, k2 = jr.split(key)
        step = SongStepEmbedder(k1, **step_kwargs)
        return cls(step, k2)

    def __call__(self, song_tokens, banks, positions=None, song_length=None):
        S = song_tokens.shape[0]
        if positions is None:
            positions = jnp.arange(S)
        _song_length = (
            jnp.float32(S) if song_length is None
            else jnp.asarray(song_length, dtype=jnp.float32)
        )
        content = jax.vmap(lambda step: self.step_embedder(step, banks))(song_tokens)  # (S, 4, d)
        phrase_pos = self.phrase_position(positions % STEPS_PER_PHRASE)  # (S, d)
        channel_pos = self.channel_position()                             # (4, d)
        progress = positions.astype(jnp.float32) / _song_length          # (S,) in [0, 1)
        progress_emb = jax.vmap(lambda p: self.progress_proj(p[None]))(progress)  # (S, d)
        return (
            content
            + phrase_pos[:, None, :]
            + channel_pos[None, :, :]
            + progress_emb[:, None, :]
        )
