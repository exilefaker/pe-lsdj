import jax.numpy as jnp
from jaxtyping import Array
from pe_lsdj.constants import *

# ----------- De-tokenizers (map back to raw bytes) -------------

def _nibble_merge(high: Array, low: Array) -> Array:
    return ((high & 0x0F) << 4) | (low & 0x0F)

# TODO: Do we want all these values to be in tokens_dict or are some
# just middle men?
def repack_notes(tokens_dict: dict[str, Array]) -> Array:
    return (tokens_dict[PHRASE_NOTES] - 1).ravel().tolist()

def repack_grooves(groove_tokens: Array) -> list:
    """Reverse of parse_grooves. Input shape: (NUM_GROOVES, STEPS_PER_GROOVE, 2)"""
    flat = groove_tokens.reshape(-1, 2)
    return _nibble_merge(flat[:, 0] - 1, flat[:, 1] - 1).astype(jnp.uint8).ravel().tolist()

def repack_instruments(tokens_dict: dict[str, Array]) -> Array:
    repacked_bytes = jnp.zeros((NUM_INSTRUMENTS, 16), dtype=jnp.uint8)

    # Byte 0
    type_IDs = tokens_dict[TYPE_ID].astype(jnp.uint8)
    repacked_bytes = repacked_bytes.at[:,0].set(type_IDs - 1)

    # Byte 1
    env_byte = (
        ((tokens_dict[ENV_VOLUME] - 1) & 0x0F) << 4 | 
        ((tokens_dict[ENV_FADE]  - 1) & 0x0F)
    )
    vol_byte = ((tokens_dict[VOLUME] - 1) & 0x03) << 5

    byte1 = jnp.where(
        (type_IDs == PU) | (type_IDs == NOI), # PU/NOI
        env_byte,
        vol_byte # WAV/KIT
    ).astype(jnp.uint8)
    repacked_bytes = repacked_bytes.at[:, 1].set(byte1)

    # Byte 2
    wave_byte2 = (
        ((tokens_dict[SOFTSYNTH_ID] - 1) & 0x0F) << 4 |
        ((tokens_dict[REPEAT]    - 1) & 0x0F)
    )

    kit_attack1_bit = ((tokens_dict[KEEP_ATTACK_1] - 1) & 0x01) << 7
    kit_halfspeed_bit = ((tokens_dict[HALF_SPEED]  - 1) & 0x01) << 6
    kit_1_ID_bits = ((tokens_dict[KIT_1_ID] - 1) & 0x3F)

    kit_byte2 = kit_attack1_bit | kit_halfspeed_bit | kit_1_ID_bits

    pulse_byte2 = tokens_dict[PHASE_TRANSPOSE] - 1

    repacked_bytes = repacked_bytes.at[:, 2].set(
        jnp.select(
            [type_IDs == PU, type_IDs == WAV, type_IDs == KIT], 
            [pulse_byte2, wave_byte2, kit_byte2], 
            default=0
        )
    ).astype(jnp.uint8)

    # Byte 3
    length_bits = ((tokens_dict[LENGTH] - 1) & 0x3F)
    length_limited_bits = ((tokens_dict[LENGTH_LIMITED] - 1) & 0x01) << 6

    length_kit_1_byte = tokens_dict[LENGTH_KIT_1] - 1

    pu_noi_byte2 = length_bits | length_limited_bits
    
    repacked_bytes = repacked_bytes.at[:,3].set(
        jnp.select(
            [(type_IDs == PU) | (type_IDs == NOI), type_IDs == KIT],
            [pu_noi_byte2, length_kit_1_byte],
            default = 0
        )
    ).astype(jnp.uint8)

    # Byte 4
    sweep_byte = tokens_dict[SWEEP] - 1

    repacked_bytes = repacked_bytes.at[:,4].set(
        sweep_byte * ((type_IDs == PU) | (type_IDs == NOI))
    ).astype(jnp.uint8)

    # Byte 5
    table_automate_bit = ((tokens_dict[TABLE_AUTOMATE] - 1) & 0x01) << 3
    automate_2_bit = ((tokens_dict[AUTOMATE_2] - 1) & 0x01) << 4
    vibrato_type_bits = ((tokens_dict[VIBRATO_TYPE] - 1) & 0x03) << 1
    vibrato_direction_bits = (tokens_dict[VIBRATO_DIRECTION] - 1) & 0x01
    loop_kit_bit1 = ((tokens_dict[LOOP_KIT_1] - 1) & 0x01) << 5
    loop_kit_bit2 = ((tokens_dict[LOOP_KIT_2] - 1) & 0x01) << 6

    pu_wav_kit_bits = vibrato_type_bits | vibrato_direction_bits
    kit_bits = loop_kit_bit1 | loop_kit_bit2

    pu_wav_or_kit = (type_IDs == PU) | (type_IDs == WAV) | (type_IDs == KIT)

    byte5 = (
        table_automate_bit
        | automate_2_bit
        | pu_wav_kit_bits * pu_wav_or_kit 
        | kit_bits * (type_IDs == KIT)
    ).astype(jnp.uint8)
    
    repacked_bytes = repacked_bytes.at[:,5].set(byte5)

    # Byte 6
    table_bits = (tokens_dict[TABLE] - 1) & 0x1F
    table_on_off_bits = ((tokens_dict[TABLE_ON_OFF] - 1) << 5)
    table_byte = (table_bits | table_on_off_bits) & 0xFF

    repacked_bytes = repacked_bytes.at[:,6].set(table_byte)

    # Byte 7
    wave_bits = ((tokens_dict[WAVE] - 1) << 6)
    phase_finetune_bits = ((tokens_dict[PHASE_FINETUNE] - 1) & 0x0F) << 2
    pan_bits = (tokens_dict[PAN] - 1) & 0x0F
    pu_bits = wave_bits | phase_finetune_bits
    byte7 = pan_bits | pu_bits * (type_IDs == PU) 
    
    repacked_bytes = repacked_bytes.at[:,7].set(byte7)

    # Byte 8
    pitch_byte = tokens_dict[PITCH] - 1

    repacked_bytes = repacked_bytes.at[:,8].set(
        pitch_byte * (type_IDs == KIT)
    ).astype(jnp.uint8)

    # Byte 9
    play_type_bits = (tokens_dict[PLAY_TYPE] - 1) & 0x03
    keep_attack_2_bit = ((tokens_dict[KEEP_ATTACK_2] - 1) & 0x01) << 7
    kit_2_id_bit = (tokens_dict[KIT_2_ID] - 1) & 0x3F

    byte9 = (
        play_type_bits * (type_IDs == WAV)
        | keep_attack_2_bit * (type_IDs == KIT)
        | kit_2_id_bit * (type_IDs == KIT)
    ).astype(jnp.uint8)

    repacked_bytes = repacked_bytes.at[:,9].set(byte9)

    # Byte 10

    wave_length_bits = ((tokens_dict[WAVE_LENGTH] - 1) & 0x0F) << 4
    speed_bits = (tokens_dict[SPEED] - 1) & 0x0F
    distortion_type_bits = (tokens_dict[DISTORTION_TYPE] - 1) + 0xD0

    byte10 = (
        wave_length_bits * (type_IDs == WAV)
        | speed_bits * (type_IDs == WAV)
        | distortion_type_bits * (type_IDs == KIT)
    ).astype(jnp.uint8)

    repacked_bytes = repacked_bytes.at[:,10].set(byte10)

    # Byte 11
    byte11 = (
        (tokens_dict[LENGTH_KIT_2] - 1) 
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    repacked_bytes = repacked_bytes.at[:,11].set(byte11)

    # Byte 12
    byte12 = (
        (tokens_dict[OFFSET_KIT_1] - 1) 
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    repacked_bytes = repacked_bytes.at[:,12].set(byte12)

    # Byte 13
    byte13 = (
        (tokens_dict[OFFSET_KIT_2] - 1) 
        * (type_IDs == KIT)
    ).astype(jnp.uint8)

    repacked_bytes = repacked_bytes.at[:,13].set(byte13)


    return repacked_bytes.ravel().tolist()


def repack_softsynths(tokens_dict: dict[str, Array]) -> Array:
    repacked_bytes = jnp.zeros((NUM_SYNTHS, SYNTH_SIZE), dtype=jnp.uint8)

    # Byte 0: waveform
    repacked_bytes = repacked_bytes.at[:, 0].set(
        (tokens_dict[SOFTSYNTH_WAVEFORM] - 1).astype(jnp.uint8)
    )

    # Byte 1: filter_type
    repacked_bytes = repacked_bytes.at[:, 1].set(
        (tokens_dict[SOFTSYNTH_FILTER_TYPE] - 1).astype(jnp.uint8)
    )

    # Byte 2: filter_resonance
    repacked_bytes = repacked_bytes.at[:, 2].set(
        (tokens_dict[SOFTSYNTH_FILTER_RESONANCE] - 1).astype(jnp.uint8)
    )

    # Byte 3: distortion
    repacked_bytes = repacked_bytes.at[:, 3].set(
        (tokens_dict[SOFTSYNTH_DISTORTION] - 1).astype(jnp.uint8)
    )

    # Byte 4: phase_type
    repacked_bytes = repacked_bytes.at[:, 4].set(
        (tokens_dict[SOFTSYNTH_PHASE_TYPE] - 1).astype(jnp.uint8)
    )

    # Bytes 5-8: start params
    repacked_bytes = repacked_bytes.at[:, 5].set(
        (tokens_dict[SOFTSYNTH_START_VOLUME] - 1).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 6].set(
        (tokens_dict[SOFTSYNTH_START_FILTER_CUTOFF] - 1).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 7].set(
        (tokens_dict[SOFTSYNTH_START_PHASE_AMOUNT] - 1).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 8].set(
        (tokens_dict[SOFTSYNTH_START_VERTICAL_SHIFT] - 1).astype(jnp.uint8)
    )

    # Bytes 9-12: end params
    repacked_bytes = repacked_bytes.at[:, 9].set(
        (tokens_dict[SOFTSYNTH_END_VOLUME] - 1).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 10].set(
        (tokens_dict[SOFTSYNTH_END_FILTER_CUTOFF] - 1).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 11].set(
        (tokens_dict[SOFTSYNTH_END_PHASE_AMOUNT] - 1).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 12].set(
        (tokens_dict[SOFTSYNTH_END_VERTICAL_SHIFT] - 1).astype(jnp.uint8)
    )

    # Bytes 13-15: padding (zeros, already initialized)

    return repacked_bytes.ravel().tolist()


def repack_fx_values(tokens_dict: dict[str, Array], fx_command_IDs: Array) -> list:
    """
    Reverse of parse_fx_values: reconstruct raw FX value bytes from tokens,
    conditional on FX command IDs (0-18, where 0 = NULL).
    """
    # Nibble-pair commands: recombine high and low nibbles
    chord_byte = _nibble_merge(
        tokens_dict[CHORD_FX_1] - 1, tokens_dict[CHORD_FX_2] - 1
    )
    env_byte = _nibble_merge(
        tokens_dict[ENV_FX_VOL] - 1, tokens_dict[ENV_FX_FADE] - 1
    )
    retrig_byte = _nibble_merge(
        tokens_dict[RETRIG_FX_FADE] - 1, tokens_dict[RETRIG_FX_RATE] - 1
    )
    vibrato_byte = _nibble_merge(
        tokens_dict[VIBRATO_FX_SPEED] - 1, tokens_dict[VIBRATO_FX_DEPTH] - 1
    )
    random_byte = _nibble_merge(
        tokens_dict[RANDOM_FX_L] - 1, tokens_dict[RANDOM_FX_R] - 1
    )

    # ID/enum/byte commands: subtract the null offset
    table_byte = tokens_dict[TABLE_FX] - 1
    groove_byte = tokens_dict[GROOVE_FX] - 1
    hop_byte = tokens_dict[HOP_FX] - 1
    pan_byte = tokens_dict[PAN_FX] - 1
    volume_byte = tokens_dict[VOLUME_FX] - 1
    wave_byte = tokens_dict[WAVE_FX] - 1
    continuous_byte = tokens_dict[CONTINUOUS_FX] - 1

    is_continuous = (
        (fx_command_IDs == CMD_D)
        | (fx_command_IDs == CMD_F)
        | (fx_command_IDs == CMD_K)
        | (fx_command_IDs == CMD_L)
        | (fx_command_IDs == CMD_P)
        | (fx_command_IDs == CMD_S)
        | (fx_command_IDs == CMD_T)
    )

    data_bytes = jnp.select(
        [
            fx_command_IDs == CMD_A,
            fx_command_IDs == CMD_C,
            fx_command_IDs == CMD_E,
            fx_command_IDs == CMD_G,
            fx_command_IDs == CMD_H,
            fx_command_IDs == CMD_M,
            fx_command_IDs == CMD_O,
            fx_command_IDs == CMD_R,
            fx_command_IDs == CMD_V,
            fx_command_IDs == CMD_W,
            fx_command_IDs == CMD_Z,
            is_continuous,
        ],
        [
            table_byte,
            chord_byte,
            env_byte,
            groove_byte,
            hop_byte,
            volume_byte,
            pan_byte,
            retrig_byte,
            vibrato_byte,
            wave_byte,
            random_byte,
            continuous_byte,
        ],
        default=0,
    ).astype(jnp.uint8)

    return data_bytes.tolist()


def repack_tables(tokens_dict: dict[str, Array]) -> dict[str, list]:
    """
    Reverse of parse_tables. Returns dict of byte lists, one per memory region:
        "envelopes"  → TABLE_ENVELOPES_ADDR
        "transposes" → TABLE_TRANSPOSES_ADDR
        "fx_cmd_1"   → TABLE_FX_ADDR
        "fx_val_1"   → TABLE_FX_VAL_ADDR
        "fx_cmd_2"   → TABLE_FX_2_ADDR
        "fx_val_2"   → TABLE_FX_2_VAL_ADDR
    """
    # Envelopes: merge volume/fade nibbles back into bytes
    env_bytes = _nibble_merge(
        tokens_dict[TABLE_ENV_VOLUME] - 1,
        tokens_dict[TABLE_ENV_FADE] - 1,
    ).ravel().astype(jnp.uint8).tolist()

    # Transposes: remove null offset
    transpose_bytes = (
        tokens_dict[TABLE_TRANSPOSE] - 1
    ).ravel().astype(jnp.uint8).tolist()

    # FX commands: parse_fx_commands doesn't add +1, raw 0-18 values
    fx_cmd_1 = tokens_dict[TABLE_FX_1].ravel()
    fx_cmd_2 = tokens_dict[TABLE_FX_2].ravel()

    # FX values: unstack (32, 16, 17) → dict of (512,), then repack
    fx_val_1_flat = tokens_dict[TABLE_FX_VALUE_1].reshape(
        -1, FX_VALUES_FEATURE_DIM
    )
    fx_val_1_dict = {
        k: fx_val_1_flat[:, i] for i, k in enumerate(FX_VALUE_KEYS)
    }

    fx_val_2_flat = tokens_dict[TABLE_FX_VALUE_2].reshape(
        -1, FX_VALUES_FEATURE_DIM
    )
    fx_val_2_dict = {
        k: fx_val_2_flat[:, i] for i, k in enumerate(FX_VALUE_KEYS)
    }

    return {
        "envelopes": env_bytes,
        "transposes": transpose_bytes,
        "fx_cmd_1": fx_cmd_1.astype(jnp.uint8).tolist(),
        "fx_val_1": repack_fx_values(fx_val_1_dict, fx_cmd_1),
        "fx_cmd_2": fx_cmd_2.astype(jnp.uint8).tolist(),
        "fx_val_2": repack_fx_values(fx_val_2_dict, fx_cmd_2),
    }
