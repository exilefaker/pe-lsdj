import equinox as eqx
import jax.numpy as jnp
import bread
from typing import NamedTuple
from pylsdj.blockutils import BlockWriter, BlockFactory
from jaxtyping import Array
from pe_lsdj.tokenizer import (
    _reduced_fx_cmd,
    parse_grooves,
    parse_instruments,
    parse_notes,
    parse_fx_commands,
    parse_fx_values,
    parse_softsynths,
    parse_tables,
    parse_waveframes,
)
from pe_lsdj.detokenizer import repack_song
from pylsdj import load_lsdsng, filepack, bread_spec as spec
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
    active = arr_1d != EMPTY
    if not jnp.any(active):
        return 0, 0

    start = int(jnp.argmax(active))
    rest = active[start:]
    if jnp.all(rest):
        end = len(arr_1d)
    else:
        end = start + int(jnp.argmin(rest))

    return start, end


def _with_sentinel(arr):
    return jnp.concatenate([
        arr, jnp.zeros((1,) + arr.shape[1:], dtype=arr.dtype)
    ])


class Settings(NamedTuple):
    """
    Placeholder for more comprehensive storage of LSDJ project settings.
    For now, we can use this to treat these values as pass-throughs.
    """
    settings_bytes: Array


class SongFile(eqx.Module):
    name: str
    tempo: jnp.int32
    song_tokens: Array # Song structure
    settings: Settings

    # Entity tensors (inlined during generation)
    instruments: dict
    tables: dict
    traces: dict
    softsynths: dict
    grooves: Array
    waveframes: Array

    def __init__(self, filename: str = None, *, raw_bytes=None, name=""):
        if raw_bytes is not None:
            self._load_from_raw(raw_bytes, name)
        elif filename is not None:
            pylsdj_project = load_lsdsng(filename)
            self._load_from_raw(pylsdj_project._raw_bytes, pylsdj_project.name)
        else:
            raise ValueError("Must provide either filename or raw_bytes")

    def _load_from_raw(self, raw_bytes, name: str):
        """
        Parse the (decompressed) raw bytes of a LSDJ v3.9.2 track.

        Produces:
        - song_tokens: (S, 4, 21) per-step sequence with scalar entity IDs
        - Entity tensors stored separately (instruments, tables, traces,
          grooves, softsynths) for model lookup by ID
        """
        raw_data = jnp.array(raw_bytes, dtype=jnp.uint8)

        self.name = name
        self.tempo = raw_data[TEMPO_ADDR][0]

        # ===== Parse entity tensors (stored directly, no inlining) =====

        self.grooves = parse_grooves(raw_data[GROOVES_ADDR])
        self.softsynths = parse_softsynths(raw_data[SOFTSYNTH_PARAMS_ADDR])
        self.instruments = parse_instruments(raw_data[INSTRUMENTS_ADDR])
        self.tables, self.traces = parse_tables(raw_data)
        self.waveframes = parse_waveframes(raw_data[WAVE_FRAMES_ADDR])

        # ===== Extract active song structure =====

        song_chains = raw_data[SONG_CHAINS_ADDR].reshape(
            ((NUM_SONG_CHAINS, NUM_CHANNELS))
        ).astype(jnp.uint8)

        chain_phrases = raw_data[CHAIN_PHRASES_ADDR].reshape(
            (NUM_CHAINS, PHRASES_PER_CHAIN)
        ).astype(jnp.uint8)

        chain_transposes = raw_data[CHAIN_TRANSPOSES_ADDR].reshape(
             (NUM_CHAINS, PHRASES_PER_CHAIN)
        ).astype(jnp.uint8)

        phrase_instrument_ids = raw_data[PHRASE_INSTR_ADDR].reshape(
            (NUM_PHRASES, STEPS_PER_PHRASE)
        ).astype(jnp.uint8)

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

        # Pack ragged lists into (max_phrases, NUM_CHANNELS)
        # Pad shorter channels with EMPTY=255 (sentinel = empty phrase).
        max_phrases = max(len(cp) for cp in active_song_phrases)
        song_phrases = jnp.full(
            (max_phrases, NUM_CHANNELS), EMPTY, dtype=jnp.uint8
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

        # ===== Parse phrase-level token arrays =====
        # Each gets a zero sentinel row at index 255 for padded phrase slots.

        phrase_notes = _with_sentinel(
            parse_notes(raw_data[PHRASE_NOTES_ADDR])
        )
        phrase_instrument_ids = _with_sentinel(phrase_instrument_ids)

        phrase_fx_raw = parse_fx_commands(raw_data[PHRASE_FX_ADDR])
        phrase_reduced_fx = _with_sentinel(
            _reduced_fx_cmd(phrase_fx_raw).reshape(NUM_PHRASES, STEPS_PER_PHRASE)
        )

        fx_vals_dict = parse_fx_values(raw_data[PHRASE_FX_VAL_ADDR], phrase_fx_raw)
        phrase_fx_vals = _with_sentinel(
            jnp.column_stack(list(fx_vals_dict.values()))
            .reshape(NUM_PHRASES, STEPS_PER_PHRASE, -1)
        )

        # ===== Build song-level step sequences =====

        song_notes = step_format(phrase_notes[song_phrases])
        song_instr_ids = step_format(phrase_instrument_ids[song_phrases])
        song_reduced_fx = step_format(phrase_reduced_fx[song_phrases])
        song_fx_vals = step_format_nd(phrase_fx_vals[song_phrases])

        song_transposes = jnp.repeat(
            song_transposes_raw, repeats=STEPS_PER_PHRASE, axis=0
        )

        # song_tokens: (steps, NUM_CHANNELS, f_dim)
        # f_dim = note(1) + instr_id(1) + reduced_fx_cmd(1) 
        # + sparse_fx_vals(17) + transpose(1)
        self.song_tokens = jnp.concatenate(
            [
                song_notes[:, :, None],
                song_instr_ids[:, :, None],
                song_reduced_fx[:, :, None],
                song_fx_vals,
                song_transposes[:, :, None],
            ],
            axis=-1
        )

        # Save settings, mainly to copy over defaults
        self.settings = Settings(
            raw_bytes[SETTINGS_ADDR]
        )
    
    def repack(self):
        return repack_song(
            self.song_tokens,
            self.instruments,
            self.tables,
            self.grooves,
            self.softsynths,
            self.waveframes,
            self.tempo,
            self.settings.settings_bytes,
        )

    def to_lsdsng(self, output_filename=None, name="", version=25):
        """
        Compresses raw memory bytes and writes an .lsdsng file directly.
        
        Args:
            output_filename (str): Where to save the result.
            name (str): Name to embed in the file header (max 8 chars usually).
            version (int): LSDj version byte (e.g. 12 for recent versions).
        """
        
        # 1. Setup the Preamble (Metadata)
        # We use a dummy buffer just to initialize the bread structure
        preamble_dummy = bytearray([0] * 9) 
        preamble = bread.parse(preamble_dummy, spec.lsdsng_preamble)
        
        # Populate metadata
        preamble.name = name or self.name or "PROJECT"
        preamble.version = version
        
        # Serialize Preamble
        preamble_data = bread.write(preamble)

        # 2. Compress the Raw Data (In-Memory)
        # This replaces 'self.get_raw_data()'
        # filepack.compress takes bytes and returns compressed bytes
        compressed_data = filepack.compress(self.repack())

        # 3. Block Chunking
        # lsdsng format wraps the compressed data in blocks
        writer = BlockWriter()
        factory = BlockFactory()
        writer.write(compressed_data, factory)

        # 4. Write to Disk
        output_filename = output_filename or f"{name}.lsdsng"
        with open(output_filename, 'wb') as fp:
            # A. Write Header
            fp.write(preamble_data)
            
            # B. Write Blocks
            # (The factory stores the chunks in memory)
            for key in sorted(factory.blocks.keys()):
                fp.write(bytearray(factory.blocks[key].data))
                
        print(f"Save to {output_filename} complete.")