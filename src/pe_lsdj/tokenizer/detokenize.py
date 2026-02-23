import numpy as np
import jax.numpy as jnp
from jaxtyping import Array
from pe_lsdj.constants import *

# ----------- De-tokenizers (map back to raw bytes) -------------

def _nibble_merge(high: Array, low: Array) -> Array:
    return ((high & 0x0F) << 4) | (low & 0x0F)

# TODO: Do we want to preserve the `tokens_dict` thing?
def repack_notes(tokens_dict: dict[str, Array]) -> Array:
    return tokens_dict[PHRASE_NOTES].ravel().tolist()

def repack_grooves(groove_tokens: Array) -> list[int]:
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
    table_automate_bit = (tokens_dict[TABLE_AUTOMATE] & 0x01) << 4
    automate_2_bit = (tokens_dict[AUTOMATE_2] & 0x01) << 3
    vibrato_type_bits = ((tokens_dict[VIBRATO_TYPE] - 1) & 0x03) << 1
    vibrato_direction_bits = (tokens_dict[VIBRATO_DIRECTION] - 1) & 0x01
    loop_kit_bit1 = ((tokens_dict[LOOP_KIT_1] - 1) & 0x01) << 6
    loop_kit_bit2 = ((tokens_dict[LOOP_KIT_2] - 1) & 0x01) << 5

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
    table_on_off_bits = (tokens_dict[TABLE_ON_OFF] << 5)
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

    # Byte 10 (KIT only: distortion type)
    distortion_type_bits = (tokens_dict[DISTORTION_TYPE] - 1) + 0xD0
    byte10 = (
        distortion_type_bits * (type_IDs == KIT)
    ).astype(jnp.uint8)
    repacked_bytes = repacked_bytes.at[:,10].set(byte10)

    # Byte 14 (WAV only: steps / speed)
    wave_length_bits = ((tokens_dict[WAVE_LENGTH] - 1) & 0x0F) << 4
    speed_bits = (tokens_dict[SPEED] - 1) & 0x0F
    byte14 = (
        (wave_length_bits | speed_bits) * (type_IDs == WAV)
    ).astype(jnp.uint8)
    repacked_bytes = repacked_bytes.at[:,14].set(byte14)

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


def repack_waveframes(waveframe_tokens: Array) -> list[int]:
    flat = waveframe_tokens.reshape(-1, 2) - 1
    return _nibble_merge(flat[:, 0], flat[:, 1]).astype(jnp.uint8).ravel().tolist()


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
        tokens_dict[TABLE_ENV_DURATION] - 1,
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


# --- FX command recovery (reduced → full enum) ---

# --- Song-level reconstruction ---

def repack_song(
    song_tokens: Array,
    instrument_tokens: Array,
    table_tokens: Array,
    groove_tokens: Array,
    softsynth_tokens: Array,
    waveframe_tokens: Array,
    tempo_token: Array | int,
    settings: Array,
    max_phrases_per_chain: int = PHRASES_PER_CHAIN,
) -> list[int]:
    """Reconstruct raw LSDJ bytes (0x8000) from tokens.

    Reverses the full tokenize pipeline: splits song_tokens back into
    phrase/chain/song arrays, deduplicates phrases and chains, repacks
    entities, and assembles the 32KB byte array.
    """
    tokens = np.array(song_tokens, dtype=np.uint8)
    S = tokens.shape[0]
    num_phrase_blocks = S // STEPS_PER_PHRASE

    # 1. Split song_tokens columns
    notes = tokens[:, :, 0]
    instr_ids = tokens[:, :, 1]
    fx = tokens[:, :, 2]
    fx_vals = tokens[:, :, 3:20]
    transposes = tokens[:, :, 20]

    # 2. Reverse step_format: (S, C) → (P, 16, C)
    notes_by_phrase = notes.reshape(num_phrase_blocks, STEPS_PER_PHRASE, NUM_CHANNELS)
    instr_by_phrase = instr_ids.reshape(num_phrase_blocks, STEPS_PER_PHRASE, NUM_CHANNELS)
    fx_cmd_by_phrase = fx.reshape(num_phrase_blocks, STEPS_PER_PHRASE, NUM_CHANNELS)
    fx_val_by_phrase = fx_vals.reshape(
        num_phrase_blocks, STEPS_PER_PHRASE, NUM_CHANNELS, FX_VALUES_FEATURE_DIM
    )
    tr_by_phrase = transposes.reshape(num_phrase_blocks, STEPS_PER_PHRASE, NUM_CHANNELS)

    # 4. Deduplicate phrases → allocate phrase IDs
    # Full-size output arrays (255 phrases max, indexed 0-254)
    phrase_notes_out = np.full((NUM_PHRASES, STEPS_PER_PHRASE), EMPTY, dtype=np.uint8)
    phrase_instr_out = np.full((NUM_PHRASES, STEPS_PER_PHRASE), EMPTY, dtype=np.uint8)
    phrase_fx_cmd_out = np.zeros((NUM_PHRASES, STEPS_PER_PHRASE), dtype=np.uint8)
    phrase_fx_val_out = np.zeros(
        (NUM_PHRASES, STEPS_PER_PHRASE, FX_VALUES_FEATURE_DIM), dtype=np.uint8
    )

    next_phrase_id = 0
    # phrase_map: fingerprint → phrase_id
    phrase_map = {}
    # Per-channel list of (phrase_id, transpose) in song order
    phrase_ids_per_channel = [[] for _ in range(NUM_CHANNELS)]

    # For each channel, find active_len: one past the last phrase block that has
    # any non-zero token. Phrase blocks beyond this are trailing sentinel padding
    # (shorter channels padded to the song length). Blocks with all-zero tokens
    # BEFORE active_len are real silence (e.g. all "---" notes) and are included.
    def _active_len(ch):
        for p in range(num_phrase_blocks - 1, -1, -1):
            block = np.concatenate([
                notes_by_phrase[p, :, ch],
                instr_by_phrase[p, :, ch],
                fx_cmd_by_phrase[p, :, ch],
                fx_val_by_phrase[p, :, ch, :].ravel(),
                tr_by_phrase[p, :, ch],
            ])
            if np.any(block != 0):
                return p + 1
        return 0

    active_lens = [_active_len(ch) for ch in range(NUM_CHANNELS)]

    for ch in range(NUM_CHANNELS):
        for p in range(active_lens[ch]):
            p_notes = notes_by_phrase[p, :, ch]
            p_instr = instr_by_phrase[p, :, ch]
            p_fx_cmd = fx_cmd_by_phrase[p, :, ch]
            p_fx_val = fx_val_by_phrase[p, :, ch, :]

            # Fingerprint: tuple of all phrase data
            fp = (
                p_notes.tobytes() + p_instr.tobytes()
                + p_fx_cmd.tobytes() + p_fx_val.tobytes()
            )

            if fp in phrase_map:
                pid = phrase_map[fp]
            else:
                pid = next_phrase_id
                phrase_map[fp] = pid
                next_phrase_id += 1
                phrase_notes_out[pid] = p_notes
                phrase_instr_out[pid] = p_instr
                phrase_fx_cmd_out[pid] = p_fx_cmd
                phrase_fx_val_out[pid] = p_fx_val

            # Transpose is constant across the 16 steps; take first
            tr = int(tr_by_phrase[p, 0, ch])
            phrase_ids_per_channel[ch].append((pid, tr))

    # 5. Group phrases into chains + extract transposes
    chain_phrases_out = np.full(
        (NUM_CHAINS, PHRASES_PER_CHAIN), EMPTY, dtype=np.uint8
    )
    chain_transposes_out = np.zeros(
        (NUM_CHAINS, PHRASES_PER_CHAIN), dtype=np.uint8
    )

    next_chain_id = 0
    chain_map = {}
    song_chain_ids_per_channel = [[] for _ in range(NUM_CHANNELS)]

    for ch in range(NUM_CHANNELS):
        entries = phrase_ids_per_channel[ch]
        # Split into chunks of max_phrases_per_chain 
        # (default PHRASES_PER_CHAIN)
        for i in range(0, len(entries), max_phrases_per_chain):
            chunk = entries[i:i + max_phrases_per_chain]
            pids = [e[0] for e in chunk]
            trs = [e[1] for e in chunk]

            # Fingerprint for chain dedup
            fp = (tuple(pids), tuple(trs))
            if fp in chain_map:
                cid = chain_map[fp]
            else:
                cid = next_chain_id
                chain_map[fp] = cid
                next_chain_id += 1
                for j, (pid, tr) in enumerate(chunk):
                    chain_phrases_out[cid, j] = pid
                    chain_transposes_out[cid, j] = tr

            song_chain_ids_per_channel[ch].append(cid)

    # 6. Build song_chains array
    song_chains_out = np.full(
        (NUM_SONG_CHAINS, NUM_CHANNELS), EMPTY, dtype=np.uint8
    )
    for ch in range(NUM_CHANNELS):
        for i, cid in enumerate(song_chain_ids_per_channel[ch]):
            song_chains_out[i, ch] = cid

    # 7. Repack entities using existing functions
    # Notes: no offset — token 0 = "---", tokens 1..NUM_NOTES-1 = playable notes
    notes_bytes = repack_notes({PHRASE_NOTES: jnp.array(phrase_notes_out)})

    # Instruments (from SongFile entity tensors)
    instr_bytes = repack_instruments(instrument_tokens)

    # Grooves
    groove_bytes = repack_grooves(groove_tokens)

    # Softsynths
    synth_bytes = repack_softsynths(softsynth_tokens)

    # Wave Frames
    waveframe_bytes = repack_waveframes(waveframe_tokens)

    # Tables
    table_region = repack_tables(table_tokens)

    # Phrase FX: need flat arrays for repack_fx_values
    fx_cmd_flat = jnp.array(phrase_fx_cmd_out.ravel())
    fx_val_flat = phrase_fx_val_out.reshape(-1, FX_VALUES_FEATURE_DIM)
    fx_val_dict = {
        k: jnp.array(fx_val_flat[:, i]) for i, k in enumerate(FX_VALUE_KEYS)
    }
    fx_val_bytes = repack_fx_values(fx_val_dict, fx_cmd_flat)

    # Phrase FX commands: repack to raw bytes
    fx_cmd_bytes = fx_cmd_flat.astype(jnp.uint8).tolist()

    # Phrase instruments: raw byte values - 1 for offset
    phrase_instr_bytes = (phrase_instr_out - 1).ravel().tolist()

    # 8. Set allocation tables
    phrase_alloc = np.zeros(32, dtype=np.uint8)
    for pid in range(min(next_phrase_id, NUM_PHRASES)):
        byte_idx = pid // 8
        bit_idx = pid % 8
        phrase_alloc[byte_idx] |= (1 << bit_idx)

    chain_alloc = np.zeros(16, dtype=np.uint8)
    for cid in range(min(next_chain_id, NUM_CHAINS)):
        byte_idx = cid // 8
        bit_idx = cid % 8
        chain_alloc[byte_idx] |= (1 << bit_idx)

    # Instrument alloc: byte per instrument, 1 if used
    used_instr = set()
    for ch in range(NUM_CHANNELS):
        for p in range(num_phrase_blocks):
            for s in range(STEPS_PER_PHRASE):
                iid = int(instr_by_phrase[p, s, ch])
                if iid > 0:  # non-NULL (has +1 offset)
                    used_instr.add(iid - 1)
    instr_alloc = np.zeros(NUM_INSTRUMENTS, dtype=np.uint8)
    for iid in used_instr:
        if 0 <= iid < NUM_INSTRUMENTS:
            instr_alloc[iid] = 1

    # Table alloc: byte per table, 1 if used
    table_alloc = np.zeros(NUM_TABLES, dtype=np.uint8)
    for iid in used_instr:
        if 0 <= iid < NUM_INSTRUMENTS:
            tbl_id = int(instrument_tokens[TABLE][iid]) - 1
            if 0 <= tbl_id < NUM_TABLES:
                table_alloc[tbl_id] = 1

    # 9. Assemble raw_data (0x8000 bytes)
    raw = np.full(0x8000, EMPTY, dtype=np.uint8)

    # Phrase notes
    raw[PHRASE_NOTES_ADDR] = np.array(notes_bytes, dtype=np.uint8)
    # Grooves
    raw[GROOVES_ADDR] = np.array(groove_bytes, dtype=np.uint8)
    # Song chains
    raw[SONG_CHAINS_ADDR] = song_chains_out.ravel()
    # Chain phrases
    raw[CHAIN_PHRASES_ADDR] = chain_phrases_out.ravel()
    # Chain transposes
    raw[CHAIN_TRANSPOSES_ADDR] = chain_transposes_out.ravel()
    # Instruments
    raw[INSTRUMENTS_ADDR] = np.array(instr_bytes, dtype=np.uint8)
    # Tables
    raw[TABLE_ENVELOPES_ADDR] = np.array(table_region["envelopes"], dtype=np.uint8)
    raw[TABLE_TRANSPOSES_ADDR] = np.array(table_region["transposes"], dtype=np.uint8)
    raw[TABLE_FX_ADDR] = np.array(table_region["fx_cmd_1"], dtype=np.uint8)
    raw[TABLE_FX_VAL_ADDR] = np.array(table_region["fx_val_1"], dtype=np.uint8)
    raw[TABLE_FX_2_ADDR] = np.array(table_region["fx_cmd_2"], dtype=np.uint8)
    raw[TABLE_FX_2_VAL_ADDR] = np.array(table_region["fx_val_2"], dtype=np.uint8)
    # Softsynth params
    raw[SOFTSYNTH_PARAMS_ADDR] = np.array(synth_bytes, dtype=np.uint8)
    # Wave Frames
    raw[WAVE_FRAMES_ADDR] = np.array(waveframe_bytes, dtype=np.uint8)
    # Phrase FX commands
    raw[PHRASE_FX_ADDR] = np.array(fx_cmd_bytes, dtype=np.uint8)
    # Phrase FX values
    raw[PHRASE_FX_VAL_ADDR] = np.array(fx_val_bytes, dtype=np.uint8)
    # Phrase instruments
    raw[PHRASE_INSTR_ADDR] = np.array(phrase_instr_bytes, dtype=np.uint8)

    # Allocation tables
    raw[PHRASE_ALLOC_TABLE_ADDR] = phrase_alloc
    raw[CHAIN_ALLOC_TABLE_ADDR] = chain_alloc
    raw[INSTR_ALLOC_TABLE_ADDR] = instr_alloc
    raw[TABLE_ALLOC_TABLE_ADDR] = table_alloc

    # Mem init flags
    raw[MEM_INIT_FLAG_ADDR] = [ord('r'), ord('b')]
    raw[MEM_INIT_FLAG2_ADDR] = [ord('r'), ord('b')]
    raw[MEM_INIT_FLAG3_ADDR] = [ord('r'), ord('b')]

    # Tempo
    raw[TEMPO_ADDR] = [int(tempo_token)]

    # Bookmarks and other regions: zero out (not EMPTY)
    raw[BOOKMARKS_ADDR] = 0

    # Repack (default) settings
    raw[SETTINGS_ADDR] = settings

    return raw.tolist()
