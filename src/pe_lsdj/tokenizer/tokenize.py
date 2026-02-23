"""
Tools to parse specific datatypes from LSDJ into tokens
(Assumes v3.9.2)
"""
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array
from pe_lsdj.constants import *


def parse_notes(data_bytes: Array) -> Array:
    # Raw 0 = "---" (no note); 1..NUM_NOTES-1 = playable notes. No +1 offset needed.
    raw_data = data_bytes.reshape(
        (NUM_PHRASES, STEPS_PER_PHRASE)
    ).astype(jnp.uint8)
    return raw_data * ~(raw_data >= NUM_NOTES)  # Set invalid bytes to 0


# TODO: Decide how best to integrate
def parse_notes_normed(data_bytes: Array) -> Array:
    is_note = (data_bytes > 0) & ~(data_bytes > NUM_NOTES)

    return jnp.column_stack(
        [is_note, (data_bytes * is_note) / NUM_NOTES]
    ).reshape((NUM_PHRASES, STEPS_PER_PHRASE)).astype(jnp.float32)


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


def _nibble_split(bytes: Array) -> Array:
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
    Distortion type|[clip,shape,shape2,wrap]| byte 10 (D0-D3) | [KIT]
    --------------------------------------------------------------------
    Length kit 2   | number (0 = auto)      | byte 11         | [KIT]
    --------------------------------------------------------------------
    Offset kit 1   | number                 | byte 12         | [KIT]
    --------------------------------------------------------------------
    Offset kit 2   | number                 | byte 13         | [KIT]
    --------------------------------------------------------------------
    Wave length    | 4-bit int (hex)        | byte 14 bits 7-4| [WAV]
    Speed          | 4-bit int (hex)        | byte 14 bits 3-0| [WAV]
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
        (((byte1 >> 5) & 0x03) + 1)
        * ((type_IDs == WAV) | (type_IDs == KIT))
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
    table_automates = _get_bit(byte5, 4)  # universal boolean, no +1
    automate_2s = _get_bit(byte5, 3)      # universal boolean, no +1

    vibrato_types = (
        (((byte5 >> 1) & 0x03) + 1)
        * ~(type_IDs == NOI)
    ).astype(jnp.uint8)

    vibrato_direction = (
        (_get_bit(byte5, 0) + 1)
        * ~(type_IDs == NOI)
    ).astype(jnp.uint8)

    loop_kit_1 = (
        (_get_bit(byte5, 6) + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    loop_kit_2 = (
        (_get_bit(byte5, 5) + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    # Byte 6
    byte6 = raw_instruments[:,6]
    tables = (byte6 & 0x1F) + 1
    table_toggles = _get_bit(byte6, 5)  # universal boolean, no +1

    # Byte 7
    byte7 = raw_instruments[:,7]
    waves = (
        (((byte7 >> 6) & 0x03) + 1)
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
        ((byte9 & 0x3F) + 1)
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    # Byte 10 (KIT only: distortion type)
    byte10 = raw_instruments[:,10]
    distortion_types = (
        ((byte10 - 0xD0) + 1)
        * ((type_IDs == KIT) & (byte10 >= 0xD0) & (byte10 <= 0xD3))
    ).astype(jnp.uint8)

    # Byte 14 (WAV only: steps / speed)
    byte14 = raw_instruments[:,14]
    wave_lengths_and_speeds = (
        (_nibble_split(byte14) + 1)
        * ((type_IDs == WAV)[:,None])
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

def parse_fx_values(
    data_bytes: Array,
    fx_command_IDs: Array,
) -> dict[str, Array]:
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


# Synths

def parse_softsynths(data: Array) -> dict[str, Array]:
    """
    Parse raw (decompressed) bytes into Softsynth params
    (WAV channel instrument configs).

    Field              | Semantics                          | Byte
    ====================================================================
    Waveform           | [sawtooth, square, sine]           | 0
    Filter type        | [lowpass, highpass, bandpass, all] | 1
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


def parse_waveframes(data: Array) -> Array:
    return _nibble_split(data).reshape((
        NUM_SYNTHS,
        WAVES_PER_SYNTH,
        FRAMES_PER_WAVE
    )) + 1


# Parse tables

def get_resolve_maps(fx_cmd_1, fx_val_1_raw, fx_cmd_2, fx_val_2_raw):
    """
    Helper function to build resolution maps for Left (CMD1/Trans) and 
    Right (CMD2) columns in table ata.

    Used to create "trace" representations of tables, which can be used 
    inside table definitions to avoid recursion.

    Uses parsed commands (masked, no +1 offset) and raw value bytes
    (no +1 offset) so values can be used directly as indices.

    fx_cmd_1, fx_cmd_2: (N,) - output of parse_fx_commands
    fx_val_1_raw, fx_val_2_raw: (N,) - raw FX value bytes from save data
    
    Rules:
    1. 'A' Command: Affects both columns. Priority: CMD1 > CMD2.
    2. 'H' Command: Affects only its own column.
    3. 'H' Count: Ignored (Infinite Loop approximation).
    """
    flat_dim = len(fx_cmd_1)
    
    # Initialize both as identity maps (pointing to themselves)
    resolve_map_L = np.arange(flat_dim)
    resolve_map_R = np.arange(flat_dim)
    
    for idx in range(flat_dim):
        cmd1 = int(fx_cmd_1[idx])
        val1 = int(fx_val_1_raw[idx])
        
        cmd2 = int(fx_cmd_2[idx])
        val2 = int(fx_val_2_raw[idx])
        
        table_start = (idx // STEPS_PER_TABLE) * STEPS_PER_TABLE
        
        # --- A (table) commands ---
        target_table_idx = -1
        
        if cmd1 == CMD_A:
            target_table_idx = val1 * STEPS_PER_TABLE
        elif cmd2 == CMD_A:
            target_table_idx = val2 * STEPS_PER_TABLE
            
        if target_table_idx != -1:
            # If A exists, it overrides everything.
            # Both cursors jump to the START of the target table.
            resolve_map_L[idx] = target_table_idx
            resolve_map_R[idx] = target_table_idx
            continue 

        # --- H (hop) commands ---
        
        # Left Column Logic
        if cmd1 == CMD_H:
            # Mask high nibble (count), keep low nibble (step index)
            hop_target = val1 & 0x0F 
            resolve_map_L[idx] = table_start + hop_target
            
        # Right Column Logic
        if cmd2 == CMD_H:
            hop_target = val2 & 0x0F
            resolve_map_R[idx] = table_start + hop_target
            
    # We must compress both maps to handle chains like:
    # Table 1 (A->2) -> Table 2 (A->3) -> Data
    for _ in range(4):
        resolve_map_L = resolve_map_L[resolve_map_L]
        resolve_map_R = resolve_map_R[resolve_map_R]
        
    return resolve_map_L, resolve_map_R


def get_traces(resolve_map_L, resolve_map_R, flat_tables):
    """
    Algorithm to construct table "execution trace" representations that
    eliminate any nested table commands. This allows us to compute meaningful
    table embeddings without potentially endless recursion.

    For example, given the following:

    Table 1              Table 2
    =============||=============
    Env|CMD1|CMD2||Env|CMD1|CMD2
    -------------||-------------
    A3 |C03 | -  ||82 |C35 | E88
    82 | -  |OL  ||71 | -  | E67
    72 |H00 |OR  ||00 |P12 | E55
    62 | -  |A02 ||00 |P37 | E23
    51 | -  | -  ||00 | -  | E11
         ...           ...

    The encoding for Table 1 would walk through the steps following hop (H)
    and table (A) commands:

    trace(Table 1) = Env|CMD1|CMD2
                     -------------
                     A3 |C03 | -
                     82 | -  |OL
                   * 72 |C03 |OR         * Execution trace follows H command
                     62 |C35 |E88 +        (left column only)
                     51 | -  |E67
                         ...             + Execution trace follows A command
                                           (switch to Table 2)

    - Note that hops for the Transpose column (not shown here) are
    controlled by CMD1.

    - The Env column runs on its own timer which depends on duration values,
    so aligning it with the CMD columns is difficult. As a simple heuristic
    we just use the Env from the first (root) table.

    - Edge case: two A commands on same line: column 1 overrides column 2

    The algorithm works by initializing a "cursor" at the start of each table
    in parallel, and walking through a pre-computed next-step map.

    Returns:

    Trace representations for all 32 tables, shaped (NUM_TABLES, STEPS_PER_TABLE, ...).
    """
    
    cursor_L = jnp.arange(NUM_TABLES) * STEPS_PER_TABLE
    cursor_R = jnp.arange(NUM_TABLES) * STEPS_PER_TABLE
    
    trace_L = []
    trace_R = []

    transpose_flat = flat_tables[TABLE_TRANSPOSE]
    cmd1_flat = flat_tables[TABLE_FX_1]
    cmd_val1 = flat_tables[TABLE_FX_VALUE_1]

    # data_left columns: [0: transpose, 1: cmd1, 2..2+FX_dim: cmd_val1]
    data_left = jnp.column_stack([transpose_flat, cmd1_flat, cmd_val1])

    cmd2_flat = flat_tables[TABLE_FX_2]
    cmd_val2 = flat_tables[TABLE_FX_VALUE_2]

    # data_right columns: [0: cmd2, 1..1+FX_dim: cmd_val2]
    data_right = jnp.column_stack([cmd2_flat, cmd_val2])

    for _ in range(STEPS_PER_TABLE):

        real_L = jnp.take(resolve_map_L, cursor_L, axis=0)
        real_R = jnp.take(resolve_map_R, cursor_R, axis=0)

        # Sync A-command jumps: if either column resolved to a
        # different table, force BOTH to the same target (CMD1 priority)
        a_jumped_L = (real_L // STEPS_PER_TABLE) != (cursor_L // STEPS_PER_TABLE)
        a_jumped_R = (real_R // STEPS_PER_TABLE) != (cursor_R // STEPS_PER_TABLE)
        any_jump = a_jumped_L | a_jumped_R
        a_target = jnp.where(a_jumped_L, real_L, real_R)
        real_L = jnp.where(any_jump, a_target, real_L)
        real_R = jnp.where(any_jump, a_target, real_R)

        trace_L.append(jnp.take(data_left, real_L, axis=0))
        trace_R.append(jnp.take(data_right, real_R, axis=0))

        # Advance from (potentially synced) resolved positions
        step_L = real_L % STEPS_PER_TABLE
        step_R = real_R % STEPS_PER_TABLE
        table_start_L = real_L - step_L
        table_start_R = real_R - step_R
        next_step_L = jnp.where(step_L == 15, 0, step_L + 1)
        next_step_R = jnp.where(step_R == 15, 0, step_R + 1)
        cursor_L = table_start_L + next_step_L
        cursor_R = table_start_R + next_step_R

    # Stack: (STEPS_PER_TABLE, NUM_TABLES, cols) -> (NUM_TABLES, STEPS_PER_TABLE, cols)
    traces_L = jnp.stack(trace_L).transpose(1, 0, 2)
    traces_R = jnp.stack(trace_R).transpose(1, 0, 2)

    return {
        TABLE_ENV_VOLUME: flat_tables[TABLE_ENV_VOLUME].reshape(NUM_TABLES, STEPS_PER_TABLE),
        TABLE_ENV_DURATION: flat_tables[TABLE_ENV_DURATION].reshape(NUM_TABLES, STEPS_PER_TABLE),
        TABLE_TRANSPOSE: traces_L[:, :, 0],
        TABLE_FX_1: traces_L[:, :, 1],
        TABLE_FX_VALUE_1: traces_L[:, :, 2:],
        TABLE_FX_2: traces_R[:, :, 0],
        TABLE_FX_VALUE_2: traces_R[:, :, 1:],
    }


def parse_tables(data: Array) -> tuple[dict[str, Array], dict[str, Array]]:
    """
    Parse raw (decompressed) bytes into tables.

    Returns (raw_tables, traces):
    - raw_tables: original table data with A commands intact (for phrase/
      instrument level embedding via cosine similarity lookup)
    - traces: resolved table data with A/H commands followed through
      (for nested table references, avoiding recursion)

    Field        | Semantics          | Bytes
    ====================================================================
    Envelope     | (volume, duration) | 0x1690:0x1890
    Transpose    | continuous (0-255) | 0x3480:0x3680
    FX command 1 | FX command enum    | 0x3680:0x3880
    FX value 1   | FX value tokens    | 0x3880:0x3A80
    FX command 2 | FX command enum    | 0x3A80:0x3C80
    FX value 2   | FX value tokens    | 0x3C80:0x3E80
    """
    shape = (NUM_TABLES, STEPS_PER_TABLE)
    fx_dim = len(FX_VALUE_KEYS)

    # Envelopes: 1 byte per step, nibble-split into (volume, fade)
    env_nibbles = _nibble_split(data[TABLE_ENVELOPES_ADDR]) + 1  # (512, 2)
    env_volume = env_nibbles[:, 0].reshape(shape).astype(jnp.uint8)
    env_duration = env_nibbles[:, 1].reshape(shape).astype(jnp.uint8)

    # Transposes: continuous 0-255
    transposes = _with_null(data[TABLE_TRANSPOSES_ADDR]).reshape(shape)

    # FX slot 1
    fx_cmd_1_flat = parse_fx_commands(data[TABLE_FX_ADDR])
    fx_val_1_dict = parse_fx_values(data[TABLE_FX_VAL_ADDR], fx_cmd_1_flat)
    fx_val_1 = jnp.column_stack(
        [fx_val_1_dict[k] for k in FX_VALUE_KEYS]
    ).reshape((*shape, fx_dim)).astype(jnp.uint8)

    # FX slot 2
    fx_cmd_2_flat = parse_fx_commands(data[TABLE_FX_2_ADDR])
    fx_val_2_dict = parse_fx_values(data[TABLE_FX_2_VAL_ADDR], fx_cmd_2_flat)
    fx_val_2 = jnp.column_stack(
        [fx_val_2_dict[k] for k in FX_VALUE_KEYS]
    ).reshape((*shape, fx_dim)).astype(jnp.uint8)

    raw_tables = {
        TABLE_ENV_VOLUME: env_volume,
        TABLE_ENV_DURATION: env_duration,
        TABLE_TRANSPOSE: transposes,
        TABLE_FX_1: fx_cmd_1_flat.reshape(shape),
        TABLE_FX_VALUE_1: fx_val_1,
        TABLE_FX_2: fx_cmd_2_flat.reshape(shape),
        TABLE_FX_VALUE_2: fx_val_2,
    }

    # Build resolve maps and traces
    flat_table_data = {
        TABLE_ENV_VOLUME: env_volume.ravel(),
        TABLE_ENV_DURATION: env_duration.ravel(),
        TABLE_TRANSPOSE: transposes.ravel(),
        TABLE_FX_1: fx_cmd_1_flat.ravel(),
        TABLE_FX_VALUE_1: fx_val_1.reshape(-1, fx_dim),
        TABLE_FX_2: fx_cmd_2_flat.ravel(),
        TABLE_FX_VALUE_2: fx_val_2.reshape(-1, fx_dim),
    }

    resolve_L, resolve_R = get_resolve_maps(
        fx_cmd_1_flat, data[TABLE_FX_VAL_ADDR],
        fx_cmd_2_flat, data[TABLE_FX_2_VAL_ADDR],
    )

    traces = get_traces(resolve_L, resolve_R, flat_table_data)

    return raw_tables, traces


# NOTE: Currently unused. Maybe usable at embedding time? Revisit.
def replace_table_ids(fx_commands, fx_values, table_traces):
    """
    Given FX command and value arrays, replace table ID references (CMD_A)
    with pre-computed trace embeddings to avoid recursion.

    fx_commands: (N,) - parsed FX command IDs (0-18)
    fx_values: (N, FX_dim) - parsed FX values (2D flat)
    table_traces: dict from get_table_traces, values shaped (NUM_TABLES, STEPS_PER_TABLE, ...)
    """
    # Column 0 of fx_values is TABLE_FX (table IDs with +1 null offset)
    table_ids = (fx_values[:, 0] - 1).astype(jnp.int32)

    # Flatten and concatenate all trace fields into (NUM_TABLES, trace_dim)
    trace_keys = (TABLE_TRANSPOSE, TABLE_FX_1, TABLE_FX_VALUE_1, TABLE_FX_2, TABLE_FX_VALUE_2)
    full_traces = jnp.concatenate(
        [table_traces[k].reshape(NUM_TABLES, -1) for k in trace_keys], axis=-1
    )

    # Look up trace for each step's referenced table ID
    trace_embeddings = full_traces[table_ids]  # (N, trace_dim)

    # Zero out non-CMD_A steps
    is_table_cmd = (fx_commands == CMD_A)
    trace_embeddings = trace_embeddings * is_table_cmd[:, None]

    # Replace column 0 (TABLE_FX ID) with trace embedding, keep remaining columns
    return jnp.column_stack([trace_embeddings, fx_values[:, 1:]])


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
