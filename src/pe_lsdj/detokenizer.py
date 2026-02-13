import jax.numpy as jnp
from jaxtyping import Array
from pe_lsdj.constants import *

# ----------- De-tokenizers (map back to raw bytes) -------------

# TODO: Do we want all these values to be in tokens_dict or are some
# just middle men?
def repack_notes(tokens_dict: dict[str, Array]) -> Array:
    return (tokens_dict[PHRASE_NOTES] - 1).ravel().tolist()

def repack_grooves(tokens_dict: dict[str, Array]) -> Array:
    return (tokens_dict[GROOVES] - 1).ravel().tolist()

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
