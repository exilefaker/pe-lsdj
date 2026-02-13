import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from pe_lsdj.tokenizer import (
    parse_grooves,
    parse_envelopes,
    parse_instruments,
    parse_notes,
    parse_fx_commands,
    parse_fx_values,
    parse_tables,
)
from pylsdj import load_lsdsng
from pe_lsdj.constants import *


def flat_format(data: Array) -> Array:
    """
    Expects input of shape (NUM_SONG_PHRASES, NUM_CHANNELS, STEPS_PER_PHRASE)
    Collects the channel dimension upfront and flattens -
    Returns (NUM_CHANNELS * NUM_SONG_PHRASES * STEPS_PER_PHRASE)
    """
    return jnp.transpose(
        data,
        (1, 0, 2) # (channel, song_phrase, step)
    ).reshape((NUM_CHANNELS * NUM_SONG_STEPS))

class SongTokenizer(eqx.Module):
    name: str
    tokens: dict[str, Array]

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
            song_fx, 
            song_fx_val, 
            chain_transpose
        )

        For each input stream:
            batch_dim = (channels * song_steps) [flattened]
            embedding_dim = (batch dim, feature_dim)
        """
        # Decompress using pylsdj's load function
        pylsdj_project = load_lsdsng(filename)
        raw_data = jnp.array(pylsdj_project._raw_bytes, dtype=jnp.uint8)

        self.name = pylsdj_project.name
        
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

        song_phrases = chain_phrases[
            song_chains
        ].reshape((NUM_SONG_PHRASES, -1))

        phrase_instruments = raw_data[PHRASE_INSTR_ADDR].reshape(
            (NUM_PHRASES, STEPS_PER_PHRASE)
        ).astype(jnp.uint8)

        # =========== Parse tokens from raw data ===========

        # NOTES: tokens per phrase, to be rearranged
        phrase_notes = parse_notes(raw_data[PHRASE_NOTES_ADDR])
        
        instruments_dict = parse_instruments(raw_data[INSTRUMENTS_ADDR])
        # INSTRUMENTS: (NUM_INSTRUMENTS, concatenated_feature_tokens_dim)
        instruments = jnp.column_stack(instruments_dict.values())

        phrase_fx = parse_fx_commands(raw_data[PHRASE_FX_ADDR])
        phrase_fx_values_dict = parse_fx_values(
            raw_data[PHRASE_FX_VAL_ADDR], 
            phrase_fx
        )
        # PHRASE FX VALUES: (NUM_PHRASES, concatenated_FX_val_features_dim)
        phrase_fx_values = jnp.column_stack(
            phrase_fx_values_dict.values()
        )

        # This pulls data from several places in the array
        # TABLES: (NUM_TABLES, STEPS_PER_TABLE, concatenate_table_feature_tokens_dim)
        tables = parse_tables(raw_data)

        # ======= Slot tokens into global structure =========

        # Reshape to (channel * song_phrases * steps_per_phrase, feature_dim)
        song_notes = flat_format(phrase_notes[song_phrases])
        song_instrument_IDs = flat_format(phrase_instruments[song_phrases])
        song_instruments = instruments[song_instrument_IDs]
        song_fx = flat_format(phrase_fx.reshape((NUM_PHRASES, STEPS_PER_PHRASE))[song_phrases])

        song_fx_values = jnp.transpose(
            phrase_fx_values.reshape(
                (NUM_PHRASES, STEPS_PER_PHRASE, FX_VALUES_FEATURE_DIM)
            )[song_phrases],
            (1, 0, 2, 3)
        ).reshape((NUM_CHANNELS * NUM_SONG_STEPS, FX_VALUES_FEATURE_DIM))

        # Broadcast per-phrase chain transposes across steps in each phrase
        song_phrase_transposes = jnp.repeat(
            jnp.transpose(
                chain_transposes[song_chains],
                (1,0,2)
            ).reshape(NUM_CHANNELS * NUM_SONG_CHAINS, PHRASES_PER_CHAIN),
            repeats=STEPS_PER_PHRASE,
            axis=-1
        ).ravel()

        print("song notes", song_notes.shape)
        print("song instruments", song_instruments.shape)
        print("song fx", song_fx.shape)
        print("song fx val", song_fx_values.shape)
        print("song transpose", song_phrase_transposes.shape)





        self.tokens = dict()
        # tokens[GROOVES] = parse_grooves(raw_data[GROOVES_ADDR])



        # self.table_envelopes = parse_envelopes(
        #      raw_data[TABLE_ENVELOPES_ADDR]
        # )


        

        # self.table_transposes = raw_data[TABLE_TRANSPOSES_ADDR].reshape(
        #     NUM_TABLES, STEPS_PER_TABLE
        # ) + 1

        # self.table_fx_1 = parse_table_fx(raw_data[TABLE_FX_ADDR])




        # table_command = [
        #     ("fx", b.array(NUM_TABLES,
        #                 b.array(STEPS_PER_TABLE, b.enum(8, FX_COMMANDS)))),
        #     ("val", b.array(NUM_TABLES, b.array(STEPS_PER_TABLE, b.byte)))
        # ]



        # #  512 3680-387F: tables-> fx
        # TABLE_FX_ADDR = slice(0x3680,0x3880)
        # #  (FX enum, value) - 1 byte each
        # #  I believe pylsdj first interleaves the FX and FX-val per command, 
        # #  in the decompression stage?
        # #  512 3880-3A7F: tables-> fx val
        # TABLE_FX_VAL_ADDR = slice(0x3880,0x3A80)
        # #  512 3A80-3C7F: tables-> fx 2
        # TABLE_FX_2_ADDR = slice(0x3A80,0x3C80)
        # #  512 3C80-3E7F: tables-> fx 2 val
        # TABLE_FX_2_VAL_ADDR = slice(0x3C80,0x3E80)
        # #    2 3E80-3E81: mem initialized flag (set to “rb” on init)
        # MEM_INIT_FLAG2_ADDR = slice(0x3E80,0x3E82)
        # #   32 3E82-3EA1: phrase allocation table (bit per phrase, 1 if alloced, 0 else)
        # # 255 phrases; technically I think the last bit is padding
        # PHRASE_ALLOC_TABLE_ADDR = slice(0x3E82,0x3EA2)
        # #   16 3EA2-3EB1: chain allocation table (bit per chain, 1 if alloced, 0 else)
        # CHAIN_ALLOC_TABLE_ADDR = slice(0x3EA2,0x3EB2)
        # #  256 3EB2-3FB1: softsynth params (16 parameter blocks of 16 bytes each)
        # SOFTSYNTH_PARAMS_ADDR = slice(0x3EB2,0x3FB2)
        # #    1 3FB2-3FB2: clock, hours
        # CLOCK_HOURS_LEN = 1 # Ignore for now
        # #    1 3FB3-3FB3: clock, minutes
        # CLOCK_MINS_LEN = 1 # Ignore for now
        # #    1 3FB4-3FB4: tempo
        # TEMPO_ADDR = slice(0x3FB4,0x3FB5)
        # # Ignore most of these for now...
        # #    1 3FB5-3FB5: tune setting
        # #    1 3FB6-3FB6: total clock, days
        # #    1 3FB7-3FB7: total clock, hours
        # #    1 3FB8-3FB8: total clock, minutes
        # #    1 3FB9-3FB9: total clock, checksum (days+hours+minutes)
        # #    1 3FBA-3FBA: key delay
        # #    1 3FBB-3FBB: key repeat
        # #    1 3FBC-3FBC: font
        # #    1 3FBD-3FBD: sync setting
        # #    1 3FBE-3FBE: colorset
        # #    1 3FBF-3FBF: empty
        # #    1 3FC0-3FC0: clone (0=deep, 1=slim)
        # #    1 3FC1-3FC1: file changed?
        # #    1 3FC2-3FC2: power save
        # #    1 3FC3-3FC3: prelisten
        # #    2 3FC4-3FC5: wave synth overwrite locks
        # #   58 3FC6-3FFF: empty

        # # Bank 2:
        # # 4080 4000-4FEF: phrases->fx
        # PHRASE_FX_ADDR = slice(0x4000,0x4FF0)
        # # 4080 4FF0-5FDF: phrases->fx val
        # PHRASE_FX_VAL_ADDR = slice(0x4FF0,0x5FE0)
        # #   32 5FE0-5FFF: empty

        # # Bank 3:
        # # 4096 6000-6FFF: 256 wave frames (each frame has 32 4-bit samples)
        # WAVE_FRAMES_ADDR = slice(0x6000,0x7000)
        # # 4080 7000-7FEF: phrases->instr
        # PHRASE_INSTR_ADDR = slice(0x7000,0x7FF0)
        # #    2 7FF0-7FF1: mem initialized flag (set to “rb” on init)
        # MEM_INIT_FLAG3_ADDR = slice(0x7FF0,0x7FF2)
        # #   13 7FF2-7FFE: empty
        # #    1 7FFF-7FFF: version byte
        # VERSION_ADDR = slice(0x7FFF,0x8000)

