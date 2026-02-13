"""
Tools to parse specific datatypes from LSDJ into tokens
(Assumes v3.9.2)
"""
import jax.numpy as jnp
from jaxtyping import Array
from pylsdj import NUM_INSTRUMENTS, NUM_TABLES, STEPS_PER_TABLE
from pe_lsdj.constants import *

# NOTES

# First, generate "palette" fields via dedicated embedders:
#   - Tables (32 of them)    [ [concat features -> proj per step] -> proj per table ]
#   - Instruments (64 of them) [concat features -> proj per instrument]
#   - Softsynths (16 of them)  [concat features -> proj per synth]

# Then, use transformer to generate main sequence
# use embeddings in place of the IDs for the above palette fields
# Concatenate; project to global emb_dim

def parse_notes(data_bytes: Array) -> Array:
    raw_data = data_bytes.reshape(
        (NUM_PHRASES, STEPS_PER_PHRASE)
    ).astype(jnp.uint8) + 1 # Reserve 0 for NULL across entire system

    return raw_data * ~(raw_data > 158) # Set invalid notes to NULL

def parse_grooves(data_bytes: Array) -> Array:
    return data_bytes.reshape(
        (NUM_GROOVES, STEPS_PER_GROOVE)
    ).astype(jnp.uint8) + 1

def parse_fx(data_bytes: Array) -> Array:
    raw_data = data_bytes.reshape(
        (NUM_PHRASES, STEPS_PER_PHRASE)
    ).astype(jnp.uint8)

    return raw_data * ~(raw_data > 19) # Set invalid FX commands to NULL

def parse_fx_values(data_bytes: Array) -> Array:
    return data_bytes.reshape(
        (NUM_PHRASES, STEPS_PER_PHRASE)
    ).astype(jnp.uint8)

# def parse_song_chains(data_bytes: Array) -> Array:
#     # Don't offset by 1, because this is only used to construct
#     # tokens[song_phrases], and doesn't appear in the final embedding
#     return data_bytes.reshape(
#         ((NUM_SONG_CHAINS, NUM_CHANNELS))
#     ).astype(jnp.uint8)

#     # NOTE: This makes the "null value" 255 wrap around to 0,
#     # which works with our semantics

# def parse_chain_phrases(data_bytes: Array) -> Array:
#     # Don't offset by 1, because this is only used to construct
#     # tokens[song_phrases], and doesn't appear in the final embedding
#     return data_bytes.reshape(
#         ((NUM_CHAINS, PHRASES_PER_CHAIN))
#     ).astype(jnp.uint8) + 1
#     # NOTE: This makes the "null value" 255 wrap around to 0,
#     # which works with our semantics

def parse_envelopes(data_bytes: Array) -> Array:
    """
    Parse LSDJ envelope bytes (represented as a list of ints)

    Format: 
    - First digit (0-F) represents initial volume
    - Second digit represents fade:
        - 0: No fade
        - 1-7: Fade out over x steps
        - 8: No fade
        - A-F: Fade in over x steps
    
    We'll just use two small "vocabularies" of 16 tokens to represent
    each, and let the model sort out the semantics
    
    divmod(x, 16) performs x >> 4 on high bit and x & 0x0F on low bit
    """
    return jnp.column_stack(
        jnp.divmod(
            data_bytes.reshape((NUM_TABLES, STEPS_PER_TABLE, 2)), 
            16
        )
    ).astype(jnp.uint8) + 1

def _nibble_split(bytes: list[int]) -> Array:
    return jnp.column_stack(jnp.divmod(bytes, 16))

def _get_bit(bytes: list[int], bit_no: int) -> Array:
    return ((bytes >> bit_no) & 0x01)

def parse_instruments(data_bytes: list[int]) -> dict[str, Array]:
    """
    Parse raw (decompressed) bytes into tokens representing instruments.

    Field          | Semantics              | Mem. Addr.      | Used by
    ====================================================================
    Type ID        | [PU, WAV, KIT, NOI]    | byte 0 (0-3)    | All
    --------------------------------------------------------------------
    Envelope       | (vol, fade)            | byte 1          | [PU,NOI]
    Volume         | [0, 1, 2, 3]           | byte 1 bits 6-5 | [WAV,KIT]
    --------------------------------------------------------------------
    Phase transpose| number                 | byte 2          | [PU]
    Softsynth ID   | 4-bit ID               | byte 2 bits 7-4 | [WAV]
    Repeat         | 4-bit int (hex)        | byte 2 bits 3-0 | [WAV]
    Keep attack 1  | Boolean                | byte 2 bit 7    | [KIT]
    Half-speed     | Boolean                | byte 2 bit 6    | [KIT]
    Kit 1 ID       | 6-bit int              | byte 2 bits 5-0 | [KIT]
    --------------------------------------------------------------------
    Length         | 6-bit int              | byte 3 bits 5-0 | [PU,NOI]
    Length limited | Boolean                | byte 3 bit 6    | [PU,NOI]
    Length kit 1   | number (0 = auto)      | byte 3          | [KIT]
    --------------------------------------------------------------------
    Sweep / shape  | number                 | byte 4          | [PU,NOI]
    --------------------------------------------------------------------
    Table automate | Boolean                | byte 5 bit 3    | All
    Automate 2     | Boolean                | byte 5 bit 4    | All
    Vibrato type   | [HF, saw, sine, square]| byte 5 bit 2-1  | [PU,WAV,KIT]
    Vib. direction | [down, up]             | byte 5 bit 0    | [PU,WAV,KIT]
    Loop kit 1, 2  | Boolean x 2            | byte 5 bit 6-5  | [KIT]
    --------------------------------------------------------------------
    Table          | 5-bit ID               | byte 6 bit 4-0  | All
    Table on/off   | Boolean                | byte 6 bit 5    | All
    --------------------------------------------------------------------
    Wave           | [12.5, 25, 50, 75]%    | byte 7 bits 7-6 | [PU]
    Phase finetune | 4-bit int (hex)        | byte 7 bits 5-2 | [PU]
    Pan            | [Off, L, R, LR]        | byte 7 bits 1-0 | All
    --------------------------------------------------------------------
    Pitch          | number                 | byte 8          | [KIT]
    --------------------------------------------------------------------
    Play type      | [once,loop,pp,manual]  | byte 9 bits 1-0 | [WAV]
    Keep attack 2  | Boolean                | byte 9 bit 7    | [KIT]
    Kit 2 ID       | 6-bit int              | byte 9 bits 5-0 | [KIT]
    --------------------------------------------------------------------
    Wave length    | 4-bit int (hex)        | byte 10 bits 7-4| [WAV]
    Speed          | 4-bit int (hex)        | byte 10 bits 3-0| [WAV]
    Distortion type|[clip,shape,shape2,wrap]| byte 10 (D0-D3) | [KIT]
    --------------------------------------------------------------------
    Length kit 2   | number (0 = auto)      | byte 11         | [KIT]
    --------------------------------------------------------------------
    Offset kit 1   | number                 | byte 12         | [KIT]
    --------------------------------------------------------------------
    Offset kit 2   | number                 | byte 13         | [KIT]
    """

    raw_instruments = data_bytes.reshape(NUM_INSTRUMENTS, INSTRUMENT_SIZE)

    # Byte 0
    type_IDs = raw_instruments[:,0] + 1

    # Byte 1
    byte1 = raw_instruments[:,1]
    envelopes = (
        (_nibble_split(byte1) + 1) 
        * ((type_IDs == PU) | (type_IDs == NOI))[:,None]
    ).astype(jnp.uint8)

    volumes = (
        ((byte1 >> 5) & 0x03) + 1
        * (type_IDs == WAV) | (type_IDs == KIT)
    ).astype(jnp.uint8)

    # Byte 2
    byte2 = raw_instruments[:,2]
    phase_transposes = ((byte2 + 1) * (type_IDs == PU)).astype(jnp.uint8)

    softsynths_and_repeats = (
        (_nibble_split(byte2) + 1) 
        * (type_IDs == WAV)[:,None]
    ).astype(jnp.uint8)

    keep_attacks = (
        (_get_bit(byte2, 7) + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    half_speeds = (
        (_get_bit(byte2, 6) + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    kit_1_IDs = (
        ((byte2 & 0x3F) + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)
    
    # Byte 3
    byte3 = raw_instruments[:,3]

    lengths = (
        ((byte3 & 0x3F) + 1)
        * ((type_IDs == PU) | (type_IDs == NOI))
    ).astype(jnp.uint8)

    length_limited = (
        (_get_bit(byte3, 6) + 1)
        * ((type_IDs == PU) | (type_IDs == NOI))
    ).astype(jnp.uint8)

    length_kit_1 = (
        (byte3 + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    # Byte 4
    sweeps = (
        (raw_instruments[:,4] + 1)
        * ((type_IDs == PU) | (type_IDs == NOI))
    ).astype(jnp.uint8)

    # Byte 5
    byte5 = raw_instruments[:,5]
    table_automates = _get_bit(byte5, 3) + 1
    automate_2s = _get_bit(byte5, 4) + 1

    vibrato_types = (
        (((byte5 >> 1) & 0x03) + 1)
        * ~(type_IDs == NOI)
    ).astype(jnp.uint8)

    vibrato_direction = (
        (_get_bit(byte5, 0) + 1)
        * ~(type_IDs == NOI)
    ).astype(jnp.uint8)

    loop_kit_1 = (
        (_get_bit(byte5, 5) + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    loop_kit_2 = (
        (_get_bit(byte5, 6) + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    # Byte 6
    byte6 = raw_instruments[:,6]
    tables = (byte6 & 0x1F) + 1
    table_toggles = _get_bit(byte6, 5) + 1

    # Byte 7
    byte7 = raw_instruments[:,7]
    waves = (
        (_get_bit(byte5, 6) + 1)
        * (type_IDs == PU)
    ).astype(jnp.uint8)

    phase_finetunes = (
        (((byte7 >> 2) & 0x0F) + 1)
        * (type_IDs == PU)
    ).astype(jnp.uint8)

    pans = (byte7 & 0x03) + 1

    # Byte 8
    pitches = (
        (raw_instruments[:,8] + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    # Byte 9
    byte9 = raw_instruments[:,9]
    play_types = (
        ((byte9 & 0x03) + 1)
        * (type_IDs == WAV)
    ).astype(jnp.uint8)

    keep_attacks_2 = (
        (_get_bit(byte9, 7) + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    kit_2_IDs = (
        ((byte9 & 0x40) + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    # Byte 10
    byte10 = raw_instruments[:,10]
    wave_lengths_and_speeds = (
        (_nibble_split(byte10) + 1)
        * ((type_IDs == WAV)[:,None])
    ).astype(jnp.uint8)

    distortion_types = (
        ((byte10 - 0xD0) + 1)
        * ((type_IDs == KIT) & (byte10 >= 0xD0) & (byte10 <= 0xD3))
    ).astype(jnp.uint8)

    # Bytes 11-13
    length_kit_2 = (
        (raw_instruments[:,11] + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    offset_kit_1 = (
        (raw_instruments[:,12] + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    offset_kit_2 = (
        (raw_instruments[:,13] + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    instruments = {        
         # All instruments
        TYPE_ID: type_IDs,
        TABLE: tables,
        TABLE_ON_OFF: table_toggles,
        TABLE_AUTOMATE: table_automates,
        AUTOMATE_2: automate_2s,
        PAN: pans,
        VIBRATO_TYPE: vibrato_types,
        # All but Noise 
        VIBRATO_DIRECTION: vibrato_direction,
        ENV_VOLUME: envelopes[:,0],
        ENV_FADE: envelopes[:,1],
        # Pulse / Noise 
        LENGTH: lengths,
        LENGTH_LIMITED: length_limited,
        SWEEP: sweeps,
        VOLUME: volumes, 
        # Wave / Kit
        PHASE_TRANSPOSE: phase_transposes, 
        # Pulse
        WAVE: waves,
        PHASE_FINETUNE: phase_finetunes,
        # Wave
        SOFTSYNTH_ID: softsynths_and_repeats[:,0],
        REPEAT: softsynths_and_repeats[:,1],
        PLAY_TYPE: play_types,
        WAVE_LENGTH: wave_lengths_and_speeds[:,0],
        SPEED: wave_lengths_and_speeds[:,1],
        # Kit
        KEEP_ATTACK_1: keep_attacks,
        KEEP_ATTACK_2: keep_attacks_2,
        KIT_1_ID: kit_1_IDs,
        KIT_2_ID: kit_2_IDs,
        LENGTH_KIT_1: length_kit_1,
        LENGTH_KIT_2: length_kit_2,
        LOOP_KIT_1: loop_kit_1,
        LOOP_KIT_2: loop_kit_2,
        OFFSET_KIT_1: offset_kit_1,
        OFFSET_KIT_2: offset_kit_2,
        HALF_SPEED: half_speeds,
        PITCH: pitches,
        DISTORTION_TYPE: distortion_types,
    }

    return instruments

# NOTE: Unclear we need these. Alloc tables can (and probably should)
# be derived algorithmically from generated data.
def parse_alloc_table(data: list[int]):
    """
    Parse an allocation table into Booleans using little-endian bit order
    Input `data` should be a list of ints representing bytes
    """
    alloc_table = []
    for byte in data:
        alloc_table.extend([bool((byte >> i) & 1) for i in range(8)])
    
    return alloc_table

def parse_phrase_alloc_table(data: list[int]):
    """
    The phrase allocation table uses only 255 bits, so discard the last bit
    """
    alloc_table = []
    for idx, byte in enumerate(data):
        bits = 8 - idx == len(data) - 1
        alloc_table.extend([bool((byte >> i) & 1) for i in range(8)])
    
    return alloc_table

def parse_table_fx(data: Array) -> Array:
    raw_bytes = data.reshape((NUM_TABLES, STEPS_PER_TABLE))

    # Set values outside the valid enum range to NULL
    return (raw_bytes + 1) * ~(raw_bytes > 19)






