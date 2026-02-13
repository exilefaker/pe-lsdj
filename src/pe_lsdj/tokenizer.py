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
#   - Grooves (32)

# Then, use transformer to generate main sequence
# use embeddings in place of the IDs for the above palette fields
# Concatenate; project to global emb_dim


def parse_notes(data_bytes: Array) -> Array:
    raw_data = data_bytes.reshape(
        (NUM_PHRASES, STEPS_PER_PHRASE)
    ).astype(jnp.uint8) + 1 # Reserve 0 for NULL across entire system

    return raw_data * ~(raw_data > 158) # Set invalid notes to NULL


def parse_fx_commands(data_bytes: Array) -> Array:
    # Set invalid FX commands to NULL
    return (data_bytes * ~(data_bytes > 18)).astype(jnp.uint8) 


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

def _with_null(data: Array) -> Array:
    return data.astype(jnp.uint8) + 1

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

    raw_pans = byte7 & 0x03
    pans = (raw_pans + 1) * (raw_pans <= 3)

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


# Parse FX command values

def parse_5bit_IDs(data_bytes: Array) -> Array:
    # Add NULL token and use for invalid entries
    with_null = data_bytes + 1
    return (with_null * ~(with_null > 32)).astype(jnp.uint8)

def parse_3bit_enum(data_bytes: Array) -> Array:
    return (data_bytes + 1) * (data_bytes <= 3)

def parse_fx_values(data_bytes: Array, fx_command_IDs: Array) -> dict[str, Array]:
    """
    Parse raw (decompressed) bytes representing FX into appropriate
    structures, conditional on FX command IDs.

    FX VALUE semantics by command:

    COMMAND       | Semantics                | Parse                |  Group
    ===========================================================================
    - (NULL)      | NULL                     | Set to 0             |    -
    A (table)     | 32 IDs                   | TableEncoder         | Table
    C (chord)     | (semitone 1, semitone 2) | High and low nibbles | Chord
    D (delay)     | Onset delay in ticks     | Discretized value    | Continuous
    E (env)       | (volume, fade)           | High and low nibbles | Env
    F (finetune)  | <Instrument-specific>    | Discretized value    | Continuous
    G (groove)    | 32 IDs                   | GrooveEncoder        | Groove
    H (hop)       | Location                 | Discretized value    | Hop
    K (kill)      | Kill delay in ticks      | Discretized value    | Continuous
    L (slide)     | Slide speed              | Discretized value    | Continuous
    M (master vol)| <Complex semantics>      | Discretized value    | Volume
                    (strictly speaking this could be split into nibbles)
    O (pan)       | [Off, L, R, LR]          | Embed 4-value enum   | Pan
    P (pitch)     | Intensity                | Discretized value    | Continuous
    R (retrigger) | (fade, rate)             | High and low nibbles | Retrig
    S (sweep)     | Intensity                | Discretized value    | Continuous
    T (tempo)     | Tempo                    | Discretized value    | Continuous
    V (vibrato)   | (speed, depth)           | High and low nibbles | Vibrato
    W (wave)      | [12.5, 25, 50, 75]%      | Embed 4-value enum   | Wave
                    (has dual use for WAV chan but rare)
    Z (random)    | Ranges (L digit, R digit)| High and low nibbles | Random
    """
    byte_parse = data_bytes + 1
    nibble_parse = (_nibble_split(data_bytes) + 1)
    ID_parse = parse_5bit_IDs(data_bytes)
    enum_parse = parse_3bit_enum(data_bytes)
    chord_FX = nibble_parse * (fx_command_IDs == CMD_C)[:,None]
    env_FX = nibble_parse * (fx_command_IDs == CMD_E)[:,None]
    retrig_FX = nibble_parse * (fx_command_IDs == CMD_R)[:,None]
    vibrato_FX = nibble_parse * (fx_command_IDs == CMD_V)[:,None]
    random_FX = nibble_parse * (fx_command_IDs == CMD_Z)[:,None]
    is_continuous = (
        (fx_command_IDs == CMD_D) 
        | (fx_command_IDs == CMD_F) 
        | (fx_command_IDs == CMD_K) 
        | (fx_command_IDs == CMD_L) 
        | (fx_command_IDs == CMD_P)
        | (fx_command_IDs == CMD_S)
        | (fx_command_IDs == CMD_T)
    )

    parsed_fx_values = {
        TABLE_FX: ID_parse * (fx_command_IDs == CMD_A),
        GROOVE_FX: ID_parse * (fx_command_IDs == CMD_G),
        HOP_FX: byte_parse * (fx_command_IDs == CMD_H),
        PAN_FX: enum_parse * (fx_command_IDs == CMD_O),
        CHORD_FX_1: chord_FX[:,0],
        CHORD_FX_2: chord_FX[:,1],
        ENV_FX_VOL: env_FX[:,0],
        ENV_FX_FADE: env_FX[:,1],
        RETRIG_FX_FADE: retrig_FX[:,0],
        RETRIG_FX_RATE: retrig_FX[:,1],
        VIBRATO_FX_SPEED: vibrato_FX[:,0],
        VIBRATO_FX_DEPTH: vibrato_FX[:,1],
        VOLUME_FX: byte_parse * (fx_command_IDs == CMD_M),
        WAVE_FX: enum_parse * (fx_command_IDs == CMD_W),
        RANDOM_FX_L: random_FX[:,0],
        RANDOM_FX_R: random_FX[:,1], 
        CONTINUOUS_FX: byte_parse * is_continuous,
    }

    return parsed_fx_values


def parse_softsynths(data: Array) -> dict[str, Array]:
    """
    Parse raw (decompressed) bytes into Softsynth params
    (WAV channel instrument configs).

    Field              | Semantics                          | Byte
    ====================================================================
    Waveform           | [sawtooth, square, sine]           | 0
    Filter type        | [lowpass, highpass, bandpass, all]  | 1
    Filter resonance   | continuous (0-255)                 | 2
    Distortion         | [clip, wrap]                       | 3
    Phase type         | [normal, resync, resync2]          | 4
    Start volume       | continuous (0-255)                 | 5
    Start filter cutoff| continuous (0-255)                 | 6
    Start phase amount | continuous (0-255)                 | 7
    Start vert. shift  | continuous (0-255)                 | 8
    End volume         | continuous (0-255)                 | 9
    End filter cutoff  | continuous (0-255)                 | 10
    End phase amount   | continuous (0-255)                 | 11
    End vert. shift    | continuous (0-255)                 | 12
    (padding)          |                                    | 13-15
    """
    raw = data.reshape(NUM_SYNTHS, SYNTH_SIZE)

    # Byte 0: waveform enum (0=sawtooth, 1=square, 2=sine)
    waveforms = ((raw[:, 0] + 1) * (raw[:, 0] <= 2)).astype(jnp.uint8)

    # Byte 1: filter_type enum (0=lowpass, 1=highpass, 2=bandpass, 3=allpass)
    filter_types = ((raw[:, 1] + 1) * (raw[:, 1] <= 3)).astype(jnp.uint8)

    # Byte 2: filter_resonance (continuous)
    filter_resonances = _with_null(raw[:, 2])

    # Byte 3: distortion enum (0=clip, 1=wrap)
    distortions = ((raw[:, 3] + 1) * (raw[:, 3] <= 1)).astype(jnp.uint8)

    # Byte 4: phase_type enum (0=normal, 1=resync, 2=resync2)
    phase_types = ((raw[:, 4] + 1) * (raw[:, 4] <= 2)).astype(jnp.uint8)

    # Bytes 5-8: start params
    start_volumes = _with_null(raw[:, 5])
    start_filter_cutoffs = _with_null(raw[:, 6])
    start_phase_amounts = _with_null(raw[:, 7])
    start_vertical_shifts = _with_null(raw[:, 8])

    # Bytes 9-12: end params
    end_volumes = _with_null(raw[:, 9])
    end_filter_cutoffs = _with_null(raw[:, 10])
    end_phase_amounts = _with_null(raw[:, 11])
    end_vertical_shifts = _with_null(raw[:, 12])

    # Bytes 13-15: padding (ignored)

    return {
        SOFTSYNTH_WAVEFORM: waveforms,
        SOFTSYNTH_FILTER_TYPE: filter_types,
        SOFTSYNTH_FILTER_RESONANCE: filter_resonances,
        SOFTSYNTH_DISTORTION: distortions,
        SOFTSYNTH_PHASE_TYPE: phase_types,
        SOFTSYNTH_START_VOLUME: start_volumes,
        SOFTSYNTH_START_FILTER_CUTOFF: start_filter_cutoffs,
        SOFTSYNTH_START_PHASE_AMOUNT: start_phase_amounts,
        SOFTSYNTH_START_VERTICAL_SHIFT: start_vertical_shifts,
        SOFTSYNTH_END_VOLUME: end_volumes,
        SOFTSYNTH_END_FILTER_CUTOFF: end_filter_cutoffs,
        SOFTSYNTH_END_PHASE_AMOUNT: end_phase_amounts,
        SOFTSYNTH_END_VERTICAL_SHIFT: end_vertical_shifts,
    }


# Parse tables

def parse_tables(data: Array) -> dict[str, Array]:
    """
    Parse raw (decompressed) bytes into tables.

    Field        | Semantics          | Byte
    ====================================================================
    Envelope     | (volume, fade)     | ...
    Transpose    | continuous (0-255) | ...
    FX command 1 | FX command enum    | ...
    FX value 1   | FX value tokens    | ...
    FX command 2 ...
    FX value 2 ...
    """
    shape = (NUM_TABLES, STEPS_PER_TABLE)

    # Envelopes: 1 byte per step, nibble-split into (volume, fade)
    env_nibbles = _nibble_split(data[TABLE_ENVELOPES_ADDR]) + 1  # (512, 2)
    env_volume = env_nibbles[:, 0].reshape(shape).astype(jnp.uint8)
    env_fade = env_nibbles[:, 1].reshape(shape).astype(jnp.uint8)

    # Transposes: continuous 0-255
    transposes = _with_null(data[TABLE_TRANSPOSES_ADDR]).reshape(shape)

    # FX slot 1
    fx_cmd_1_flat = parse_fx_commands(data[TABLE_FX_ADDR])
    fx_val_1_dict = parse_fx_values(data[TABLE_FX_VAL_ADDR], fx_cmd_1_flat)
    fx_val_1 = jnp.column_stack(
        [fx_val_1_dict[k] for k in FX_VALUE_KEYS]
    ).reshape((*shape, FX_VALUES_FEATURE_DIM)).astype(jnp.uint8)

    # FX slot 2
    fx_cmd_2_flat = parse_fx_commands(data[TABLE_FX_2_ADDR])
    fx_val_2_dict = parse_fx_values(data[TABLE_FX_2_VAL_ADDR], fx_cmd_2_flat)
    fx_val_2 = jnp.column_stack(
        [fx_val_2_dict[k] for k in FX_VALUE_KEYS]
    ).reshape((*shape, FX_VALUES_FEATURE_DIM)).astype(jnp.uint8)

    return {
        TABLE_ENV_VOLUME: env_volume,
        TABLE_ENV_FADE: env_fade,
        TABLE_TRANSPOSE: transposes,
        TABLE_FX_1: fx_cmd_1_flat.reshape(shape),
        TABLE_FX_VALUE_1: fx_val_1,
        TABLE_FX_2: fx_cmd_2_flat.reshape(shape),
        TABLE_FX_VALUE_2: fx_val_2,
    }


# Parse Grooves

def parse_grooves(data: Array) -> Array:
    """
    Parse raw (decompressed) bytes into a Groove representation.
    The representation for one Groove is:
        (even_step_ticks, odd_step_ticks) * NUM_GROOVE_STEPS

    Output: (NUM_GROOVES, STEPS_PER_GROOVE, 2)
    """
    nibbles = _nibble_split(data.ravel()) + 1  # (NUM_GROOVES * STEPS_PER_GROOVE, 2)
    return nibbles.reshape(NUM_GROOVES, STEPS_PER_GROOVE, 2).astype(jnp.uint8)