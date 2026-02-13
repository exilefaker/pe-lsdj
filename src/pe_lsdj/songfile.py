import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from pe_lsdj.tokenizer import (
    parse_grooves,
    parse_instruments,
    parse_notes,
    parse_fx_commands,
    parse_fx_values,
    parse_softsynths,
    parse_tables,
)
from pylsdj import load_lsdsng
from pe_lsdj.constants import *


def step_format(data: Array) -> Array:
    """
    (phrases, channels, steps_per_phrase) -> (total_steps, channels)
    Merges phrases and steps into a single time axis, preserving channels.
    """
    return jnp.transpose(data, (0, 2, 1)).reshape(-1, data.shape[1])


def step_format_nd(data: Array) -> Array:
    """
    (phrases, channels, steps_per_phrase, feat_dim) -> (total_steps, channels, feat_dim)
    Like step_format but for multi-dimensional features.
    """
    return jnp.transpose(data, (0, 2, 1, 3)).reshape(
        -1, data.shape[1], data.shape[3]
    )


def first_contiguous_block(arr_1d):
    """Return (start, end) of first contiguous run of non-255 values."""
    active = arr_1d != 255
    if not jnp.any(active):
        return 0, 0
    
    start = int(jnp.argmax(active))
    rest = active[start:]
    if jnp.all(rest):
        end = len(arr_1d)
    else:
        end = start + int(jnp.argmin(rest))

    return start, end


def _inline_slot(fx_vals_flat, slot_idx, lookup_padded):
    """Replace a scalar ID column in sparse FX values with looked-up vectors.

    fx_vals_flat: (N, D) — flattened FX value array
    slot_idx: column index of the ID to replace
    lookup_padded: (max_id+1, vec_dim) with zero row at index 0 for null
    Returns: (N, D - 1 + vec_dim)
    """
    vecs = lookup_padded[fx_vals_flat[:, slot_idx]]
    return jnp.concatenate([
        fx_vals_flat[:, :slot_idx],
        vecs,
        fx_vals_flat[:, slot_idx + 1:],
    ], axis=1)


def _with_sentinel(arr):
    return jnp.concatenate([
        arr, jnp.zeros((1,) + arr.shape[1:], dtype=arr.dtype)
    ])


class SongTokenizer(eqx.Module):
    name: str
    tempo: jnp.int32
    song_tokens: Array

    # TODO probably remove
    song_notes: Array
    song_fx_values: Array
    song_instruments: Array
    song_transposes: Array

    def __init__(self, filename: str):
        self._load_data(filename)
    
    def _load_data(self, filename: str):
        """
        Parse the (decompressed) raw bytes of a LSDJ v3.9.2 track
        into embedding indices (tokens)

        Map (song_chains, chain_phrases), 
        (
          phrase_notes, 
          phrase_instruments, 
          phrase_fx, 
          phrase_fx_val, 
          chain_transpose
        ) ->
        NUM_SONG_STEPS x (
            song_notes, 
            song_instruments, 
            song_fx_val, 
            chain_transpose
        ) ... concatenated across channels!

        """
        # Decompress using pylsdj's load function
        pylsdj_project = load_lsdsng(filename)
        raw_data = jnp.array(pylsdj_project._raw_bytes, dtype=jnp.uint8)

        self.name = pylsdj_project.name
        self.tempo = raw_data[TEMPO_ADDR][0]
        
        # ===== Create 2D (flat_sequence x feature_dim) representation =====

        # Don't tokenize these, as they're intermediate steps
        song_chains = raw_data[SONG_CHAINS_ADDR].reshape(
            ((NUM_SONG_CHAINS, NUM_CHANNELS))
        ).astype(jnp.uint8)

        chain_phrases = raw_data[CHAIN_PHRASES_ADDR].reshape(
            (NUM_CHAINS, PHRASES_PER_CHAIN)
        ).astype(jnp.uint8)

        chain_transposes = raw_data[CHAIN_TRANSPOSES_ADDR].reshape(
             (NUM_CHAINS, PHRASES_PER_CHAIN)
        ).astype(jnp.uint8)

        # Instruments per phrase
        phrase_instrument_ids = raw_data[PHRASE_INSTR_ADDR].reshape(
            (NUM_PHRASES, STEPS_PER_PHRASE)
        ).astype(jnp.uint8)

        # ------- Preprocess to get "effective" sequence --------

        # Chain level: per channel, find the played chain block
        active_song_chains = []
        for ch in range(NUM_CHANNELS):
            ch_chains = song_chains[:, ch]
            first_phrases = chain_phrases[ch_chains, 0]
            c_start, c_end = first_contiguous_block(first_phrases)
            active_song_chains.append(ch_chains[c_start:c_end])

        # Phrase level: per active chain, extract the played phrase block
        # and its transpose, then concatenate per channel
        active_song_phrases = []
        active_song_transposes = []
        for ch in range(NUM_CHANNELS):
            ph_ids = []
            tr_vals = []
            for cid in active_song_chains[ch]:
                cp = chain_phrases[cid]
                ct = chain_transposes[cid]
                p_start, p_end = first_contiguous_block(cp)
                if p_end > p_start:
                    ph_ids.append(cp[p_start:p_end])
                    tr_vals.append(ct[p_start:p_end])
            if ph_ids:
                active_song_phrases.append(jnp.concatenate(ph_ids))
                active_song_transposes.append(jnp.concatenate(tr_vals))
            else:
                active_song_phrases.append(jnp.array([], dtype=jnp.uint8))
                active_song_transposes.append(jnp.array([], dtype=jnp.uint8))

        # ------- Pack ragged lists into (max_phrases, NUM_CHANNELS) --------
        # Pad shorter channels with 255 (sentinel = empty phrase).
        max_phrases = max(len(cp) for cp in active_song_phrases)
        song_phrases = jnp.full(
            (max_phrases, NUM_CHANNELS), 255, dtype=jnp.uint8
        )
        song_transposes_raw = jnp.zeros(
            (max_phrases, NUM_CHANNELS), dtype=jnp.uint8
        )
        for ch in range(NUM_CHANNELS):
            n = len(active_song_phrases[ch])
            song_phrases = song_phrases.at[:n, ch].set(active_song_phrases[ch])
            song_transposes_raw = song_transposes_raw.at[:n, ch].set(
                active_song_transposes[ch]
            )
        num_active_phrases = max_phrases
        num_active_steps = num_active_phrases * STEPS_PER_PHRASE

        # =========== Parse tokens from raw data ===========
        # Each phrase-level array gets a zero sentinel row appended at
        # index 255, so padded (empty) phrase slots produce null data.

        phrase_notes = _with_sentinel(
            parse_notes(raw_data[PHRASE_NOTES_ADDR])
        )
        phrase_instrument_ids = _with_sentinel(phrase_instrument_ids)

        phrase_fx_raw = parse_fx_commands(raw_data[PHRASE_FX_ADDR])
        grooves = parse_grooves(raw_data[GROOVES_ADDR])

        # Parse FX values and inline groove vectors
        fx_vals_dict = parse_fx_values(raw_data[PHRASE_FX_VAL_ADDR], phrase_fx_raw)
        fx_vals_flat = jnp.column_stack(list(fx_vals_dict.values()))

        # Replace scalar groove IDs with full groove data.
        # Groove IDs have +1 null offset: 0 = no groove, k = groove k-1.
        # Prepend a zero row so index 0 maps to zeros.
        groove_idx = list(fx_vals_dict.keys()).index(GROOVE_FX)
        grooves_flat = grooves.reshape(NUM_GROOVES, -1)
        grooves_padded = jnp.concatenate([
            jnp.zeros((1, grooves_flat.shape[-1]), dtype=jnp.uint8),
            grooves_flat,
        ])
        # Groove-inline phrase FX values (flat; table inlining applied below)
        fx_vals_grooved = _inline_slot(fx_vals_flat, groove_idx, grooves_padded)

        tables_vecs, trace_vecs = parse_tables(raw_data)

        # --- Table-level inlining (inside-out) ---

        # 1. Inline grooves in trace FX values
        table_slot = FX_VALUE_KEYS.index(TABLE_FX)
        for key in (TABLE_FX_VALUE_1, TABLE_FX_VALUE_2):
            fxv = trace_vecs[key].reshape(-1, trace_vecs[key].shape[-1])
            trace_vecs[key] = _inline_slot(fxv, groove_idx, grooves_padded) \
                .reshape(NUM_TABLES, STEPS_PER_TABLE, -1)

        # 2. Flatten groove-inlined traces into per-table lookup vectors.
        #    Drop FX command fields (TABLE_FX_1/2) — the active command is
        #    implicit in the sparse FX value structure.
        trace_keys = [k for k in trace_vecs if k not in (TABLE_FX_1, TABLE_FX_2)]
        trace_flat = jnp.concatenate([
            trace_vecs[k].reshape(NUM_TABLES, -1) for k in trace_keys
        ], axis=1)
        trace_padded = jnp.concatenate([
            jnp.zeros((1, trace_flat.shape[1]), dtype=trace_flat.dtype),
            trace_flat,
        ])

        # 3. Build groove-inlined raw table vectors for instrument inlining.
        #    Uses raw tables (A-command patterns preserved) before step 4
        #    mutates tables_vecs with trace inlining.
        raw_tbl_keys = [k for k in tables_vecs if k not in (TABLE_FX_1, TABLE_FX_2)]
        raw_tbl_parts = []
        for k in raw_tbl_keys:
            v = tables_vecs[k]
            if k in (TABLE_FX_VALUE_1, TABLE_FX_VALUE_2):
                flat = v.reshape(-1, v.shape[-1])
                v = _inline_slot(flat, groove_idx, grooves_padded) \
                    .reshape(NUM_TABLES, STEPS_PER_TABLE, -1)
            raw_tbl_parts.append(v.reshape(NUM_TABLES, -1))
        raw_table_flat = jnp.concatenate(raw_tbl_parts, axis=1)
        raw_table_padded = jnp.concatenate([
            jnp.zeros((1, raw_table_flat.shape[1]), dtype=raw_table_flat.dtype),
            raw_table_flat,
        ])

        # 4. Inline grooves then traces in raw table FX values.
        #    Groove slot (index 1) first, then table slot (index 0) —
        #    groove inlining doesn't shift index 0.
        for key in (TABLE_FX_VALUE_1, TABLE_FX_VALUE_2):
            fxv = tables_vecs[key].reshape(-1, tables_vecs[key].shape[-1])
            fxv = _inline_slot(fxv, groove_idx, grooves_padded)
            fxv = _inline_slot(fxv, table_slot, trace_padded)
            tables_vecs[key] = fxv.reshape(NUM_TABLES, STEPS_PER_TABLE, -1)

        # 5. Inline table traces in phrase FX values.
        #    Table slot is still at index 0 (groove inlining was at index 1).
        fx_vals_inlined = _inline_slot(fx_vals_grooved, table_slot, trace_padded)
        phrase_fx_values = _with_sentinel(
            fx_vals_inlined.reshape(NUM_PHRASES, STEPS_PER_PHRASE, -1)
        )

        # 6. Inline table and softsynth data into instruments.
        instruments_dict = parse_instruments(raw_data[INSTRUMENTS_ADDR])
        instr_keys = list(instruments_dict.keys())
        instruments_flat = jnp.column_stack(instruments_dict.values())

        #    TABLE field → groove-inlined raw table vector (1584 dims)
        table_col = instr_keys.index(TABLE)
        instruments_flat = _inline_slot(instruments_flat, table_col, raw_table_padded)

        #    SOFTSYNTH_ID field → full softsynth vector (13 dims)
        #    Column index shifted by table expansion (1584 - 1 = +1583).
        softsynths_dict = parse_softsynths(raw_data[SOFTSYNTH_PARAMS_ADDR])
        synths_flat = jnp.column_stack(softsynths_dict.values())
        synths_padded = jnp.concatenate([
            jnp.zeros((1, synths_flat.shape[1]), dtype=jnp.uint8),
            synths_flat,
        ])
        synth_col_orig = instr_keys.index(SOFTSYNTH_ID)
        synth_col = synth_col_orig + (raw_table_padded.shape[1] - 1) * (synth_col_orig > table_col)
        instruments = _inline_slot(instruments_flat, synth_col, synths_padded)

        # ======= Build song-level step sequences =======
        # Lookups produce (max_phrases, NUM_CHANNELS, STEPS_PER_PHRASE[, feat])
        # step_format merges phrase+step dims: -> (num_active_steps, NUM_CHANNELS[, feat])
        song_notes = step_format(phrase_notes[song_phrases])
        song_instrument_IDs = step_format(phrase_instrument_ids[song_phrases])
        song_instruments = instruments[song_instrument_IDs]
        song_fx_values = step_format_nd(phrase_fx_values[song_phrases])

        # Broadcast per-phrase transposes to step level
        song_transposes = jnp.repeat(
            song_transposes_raw, repeats=STEPS_PER_PHRASE, axis=0
        )

        # Final shape: (num_active_steps, NUM_CHANNELS, per_channel_feat_dim)
        # note(1) + instrument(N) + fx_values(48) + transpose(1)
        self.song_tokens = jnp.concatenate(
            [
                song_notes[:, :, None],
                song_instruments,
                song_fx_values,
                song_transposes[:, :, None],
            ],
            axis=-1
        )
        self.song_notes = song_notes
        self.song_instruments = song_instruments
        self.song_fx_values = song_fx_values
        self.song_transposes = song_transposes
