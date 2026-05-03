import warnings
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array
from pe_lsdj.constants import *


def _nearest_phrase(p_notes, p_instr, p_fx_cmd,
                    phrase_notes_out, phrase_instr_out, phrase_fx_cmd_out,
                    n_allocated):
    """Return the index of the most similar already-allocated phrase.

    Uses weighted Hamming distance: note mismatches count 2×, instrument 1×,
    fx_cmd 1×.  Only considers the first n_allocated rows (allocated so far).
    Called only when the phrase budget (NUM_PHRASES) is exhausted.
    """
    note_diff  = np.sum(phrase_notes_out[:n_allocated]  != p_notes,  axis=1)
    instr_diff = np.sum(phrase_instr_out[:n_allocated]  != p_instr,  axis=1)
    fx_diff    = np.sum(phrase_fx_cmd_out[:n_allocated] != p_fx_cmd, axis=1)
    dist = 2 * note_diff + instr_diff + fx_diff
    return int(np.argmin(dist))

# ----------- De-tokenizers (map back to raw bytes) -------------

def _nibble_merge(high: Array, low: Array) -> Array:
    return ((high & 0x0F) << 4) | (low & 0x0F)

def _safe_dec(t: Array) -> Array:
    """Invert the +1 null offset. Token 0 (NULL) → 0; token n → n-1."""
    return jnp.where(t > 0, t - 1, 0)

# TODO: Do we want to preserve the `tokens_dict` thing?
def repack_notes(tokens_dict: dict[str, Array]) -> Array:
    return tokens_dict[PHRASE_NOTES].ravel().tolist()

def repack_grooves(groove_tokens: Array) -> list[int]:
    """Reverse of parse_grooves. Input shape: (NUM_GROOVES, STEPS_PER_GROOVE, 2)"""
    flat = groove_tokens.reshape(-1, 2)
    return _nibble_merge(_safe_dec(flat[:, 0]), _safe_dec(flat[:, 1])).astype(jnp.uint8).ravel().tolist()

def repack_instruments(tokens_dict: dict[str, Array]) -> Array:
    repacked_bytes = jnp.zeros((NUM_INSTRUMENTS, 16), dtype=jnp.uint16)

    # Byte 0
    type_IDs = tokens_dict[TYPE_ID]
    repacked_bytes = repacked_bytes.at[:,0].set(_safe_dec(type_IDs))

    # Byte 1
    env_byte = (
        (_safe_dec(tokens_dict[ENV_VOLUME]) & 0x0F) << 4 |
        (_safe_dec(tokens_dict[ENV_FADE])   & 0x0F)
    )
    vol_byte = (_safe_dec(tokens_dict[VOLUME]) & 0x03) << 5

    byte1 = jnp.where(
        (type_IDs == PU) | (type_IDs == NOI), # PU/NOI
        env_byte,
        vol_byte # WAV/KIT
    ).astype(jnp.uint8)
    repacked_bytes = repacked_bytes.at[:, 1].set(byte1)

    # Byte 2
    wave_byte2 = (
        (_safe_dec(tokens_dict[SOFTSYNTH_ID]) & 0x0F) << 4 |
        (_safe_dec(tokens_dict[REPEAT])       & 0x0F)
    )

    kit_attack1_bit = (_safe_dec(tokens_dict[KEEP_ATTACK_1]) & 0x01) << 7
    kit_halfspeed_bit = (_safe_dec(tokens_dict[HALF_SPEED])  & 0x01) << 6
    kit_1_ID_bits = (_safe_dec(tokens_dict[KIT_1_ID]) & 0x3F)

    kit_byte2 = kit_attack1_bit | kit_halfspeed_bit | kit_1_ID_bits

    pulse_byte2 = _safe_dec(tokens_dict[PHASE_TRANSPOSE])

    repacked_bytes = repacked_bytes.at[:, 2].set(
        jnp.select(
            [type_IDs == PU, type_IDs == WAV, type_IDs == KIT],
            [pulse_byte2, wave_byte2, kit_byte2],
            default=0
        )
    )

    # Byte 3
    length_bits = (_safe_dec(tokens_dict[LENGTH]) & 0x3F)
    length_limited_bits = (_safe_dec(tokens_dict[LENGTH_LIMITED]) & 0x01) << 6

    length_kit_1_byte = _safe_dec(tokens_dict[LENGTH_KIT_1])

    pu_noi_byte2 = length_bits | length_limited_bits
    
    repacked_bytes = repacked_bytes.at[:,3].set(
        jnp.select(
            [(type_IDs == PU) | (type_IDs == NOI), type_IDs == KIT],
            [pu_noi_byte2, length_kit_1_byte],
            default=0
        )
    )

    # Byte 4 (PU/NOI only: sweep; written for all types)
    sweep_byte = _safe_dec(tokens_dict[SWEEP]).astype(jnp.uint8)
    repacked_bytes = repacked_bytes.at[:,4].set(sweep_byte)

    # Byte 5
    table_automate_bit = (tokens_dict[TABLE_AUTOMATE] & 0x01) << 4
    automate_2_bit = (tokens_dict[AUTOMATE_2] & 0x01) << 3
    vibrato_type_bits = (_safe_dec(tokens_dict[VIBRATO_TYPE]) & 0x03) << 1
    vibrato_direction_bits = _safe_dec(tokens_dict[VIBRATO_DIRECTION]) & 0x01
    loop_kit_bit1 = (_safe_dec(tokens_dict[LOOP_KIT_1]) & 0x01) << 6
    loop_kit_bit2 = (_safe_dec(tokens_dict[LOOP_KIT_2]) & 0x01) << 5

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
    table_bits = _safe_dec(tokens_dict[TABLE]) & 0x1F
    table_on_off_bits = (tokens_dict[TABLE_ON_OFF] << 5)
    table_byte = (table_bits | table_on_off_bits) & 0xFF

    repacked_bytes = repacked_bytes.at[:,6].set(table_byte)

    # Byte 7
    wave_bits = (_safe_dec(tokens_dict[WAVE]) << 6)
    phase_finetune_bits = (_safe_dec(tokens_dict[PHASE_FINETUNE]) & 0x0F) << 2
    pan_bits = _safe_dec(tokens_dict[PAN]) & 0x0F
    pu_bits = wave_bits | phase_finetune_bits
    byte7 = pan_bits | pu_bits * (type_IDs == PU) 
    
    repacked_bytes = repacked_bytes.at[:,7].set(byte7)

    # Byte 8 (KIT only: pitch; written for all types)
    pitch_byte = _safe_dec(tokens_dict[PITCH]).astype(jnp.uint8)
    repacked_bytes = repacked_bytes.at[:,8].set(pitch_byte)

    # Byte 9
    play_type_bits = _safe_dec(tokens_dict[PLAY_TYPE]) & 0x03
    keep_attack_2_bit = (_safe_dec(tokens_dict[KEEP_ATTACK_2]) & 0x01) << 7
    kit_2_id_bit = _safe_dec(tokens_dict[KIT_2_ID]) & 0x3F

    byte9 = (
        play_type_bits * (type_IDs == WAV)
        | keep_attack_2_bit * (type_IDs == KIT)
        | kit_2_id_bit * (type_IDs == KIT)
    ).astype(jnp.uint8)

    repacked_bytes = repacked_bytes.at[:,9].set(byte9)

    # Byte 10 (KIT only: distortion type)
    distortion_type_bits = _safe_dec(tokens_dict[DISTORTION_TYPE]) + 0xD0
    byte10 = (
        distortion_type_bits * (type_IDs == KIT)
    ).astype(jnp.uint8)
    repacked_bytes = repacked_bytes.at[:,10].set(byte10)

    # Byte 14 (WAV only: steps / speed; written for all types)
    wave_length_bits = (_safe_dec(tokens_dict[WAVE_LENGTH]) & 0x0F) << 4
    speed_bits = _safe_dec(tokens_dict[SPEED]) & 0x0F
    byte14 = (wave_length_bits | speed_bits).astype(jnp.uint8)
    repacked_bytes = repacked_bytes.at[:,14].set(byte14)

    # Bytes 11-13 (KIT only; written for all types)
    repacked_bytes = repacked_bytes.at[:,11].set(
        _safe_dec(tokens_dict[LENGTH_KIT_2]).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:,12].set(
        _safe_dec(tokens_dict[OFFSET_KIT_1]).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:,13].set(
        _safe_dec(tokens_dict[OFFSET_KIT_2]).astype(jnp.uint8)
    )


    return repacked_bytes.astype(jnp.uint8).ravel().tolist()


def repack_softsynths(tokens_dict: dict[str, Array]) -> Array:
    repacked_bytes = jnp.zeros((NUM_SYNTHS, SYNTH_SIZE), dtype=jnp.uint16)

    # Byte 0: waveform
    repacked_bytes = repacked_bytes.at[:, 0].set(
        _safe_dec(tokens_dict[SOFTSYNTH_WAVEFORM]).astype(jnp.uint8)
    )

    # Byte 1: filter_type
    repacked_bytes = repacked_bytes.at[:, 1].set(
        _safe_dec(tokens_dict[SOFTSYNTH_FILTER_TYPE]).astype(jnp.uint8)
    )

    # Byte 2: filter_resonance
    repacked_bytes = repacked_bytes.at[:, 2].set(
        _safe_dec(tokens_dict[SOFTSYNTH_FILTER_RESONANCE]).astype(jnp.uint8)
    )

    # Byte 3: distortion
    repacked_bytes = repacked_bytes.at[:, 3].set(
        _safe_dec(tokens_dict[SOFTSYNTH_DISTORTION]).astype(jnp.uint8)
    )

    # Byte 4: phase_type
    repacked_bytes = repacked_bytes.at[:, 4].set(
        _safe_dec(tokens_dict[SOFTSYNTH_PHASE_TYPE]).astype(jnp.uint8)
    )

    # Bytes 5-8: start params
    repacked_bytes = repacked_bytes.at[:, 5].set(
        _safe_dec(tokens_dict[SOFTSYNTH_START_VOLUME]).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 6].set(
        _safe_dec(tokens_dict[SOFTSYNTH_START_FILTER_CUTOFF]).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 7].set(
        _safe_dec(tokens_dict[SOFTSYNTH_START_PHASE_AMOUNT]).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 8].set(
        _safe_dec(tokens_dict[SOFTSYNTH_START_VERTICAL_SHIFT]).astype(jnp.uint8)
    )

    # Bytes 9-12: end params
    repacked_bytes = repacked_bytes.at[:, 9].set(
        _safe_dec(tokens_dict[SOFTSYNTH_END_VOLUME]).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 10].set(
        _safe_dec(tokens_dict[SOFTSYNTH_END_FILTER_CUTOFF]).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 11].set(
        _safe_dec(tokens_dict[SOFTSYNTH_END_PHASE_AMOUNT]).astype(jnp.uint8)
    )
    repacked_bytes = repacked_bytes.at[:, 12].set(
        _safe_dec(tokens_dict[SOFTSYNTH_END_VERTICAL_SHIFT]).astype(jnp.uint8)
    )

    # Bytes 13-15: padding (zeros, already initialized)

    return repacked_bytes.astype(jnp.uint8).ravel().tolist()


def repack_waveframes(waveframe_tokens: Array) -> list[int]:
    flat = _safe_dec(waveframe_tokens.reshape(-1, 2))
    return _nibble_merge(flat[:, 0], flat[:, 1]).astype(jnp.uint8).ravel().tolist()


def repack_fx_values(tokens_dict: dict[str, Array], fx_command_IDs: Array) -> list:
    """
    Reverse of parse_fx_values: reconstruct raw FX value bytes from tokens,
    conditional on FX command IDs (0-18, where 0 = NULL).
    """
    # Nibble-pair commands: recombine high and low nibbles
    chord_byte = _nibble_merge(
        _safe_dec(tokens_dict[CHORD_FX_1]), _safe_dec(tokens_dict[CHORD_FX_2])
    )
    env_byte = _nibble_merge(
        _safe_dec(tokens_dict[ENV_FX_VOL]), _safe_dec(tokens_dict[ENV_FX_FADE])
    )
    retrig_byte = _nibble_merge(
        _safe_dec(tokens_dict[RETRIG_FX_FADE]), _safe_dec(tokens_dict[RETRIG_FX_RATE])
    )
    vibrato_byte = _nibble_merge(
        _safe_dec(tokens_dict[VIBRATO_FX_SPEED]), _safe_dec(tokens_dict[VIBRATO_FX_DEPTH])
    )
    random_byte = _nibble_merge(
        _safe_dec(tokens_dict[RANDOM_FX_L]), _safe_dec(tokens_dict[RANDOM_FX_R])
    )

    # ID/enum/byte commands: invert the null offset
    table_byte = _safe_dec(tokens_dict[TABLE_FX])
    groove_byte = _safe_dec(tokens_dict[GROOVE_FX])
    hop_byte = _safe_dec(tokens_dict[HOP_FX])
    pan_byte = _safe_dec(tokens_dict[PAN_FX])
    volume_byte = _safe_dec(tokens_dict[VOLUME_FX])
    wave_byte = _safe_dec(tokens_dict[WAVE_FX])
    continuous_byte = _safe_dec(tokens_dict[CONTINUOUS_FX])

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


def phrase_step_bytes(ch_tokens) -> tuple[int, int, int, int]:
    """
    Convert a single channel's 21-element token vector to raw LSDJ bytes.

    Returns (note_raw, instr_raw, fx_cmd_raw, fxval_raw) — all ints ready
    to write directly to SRAM.  Pure Python; no JAX required.

    Token layout: note(0) instr_id(1) fx_cmd(2) fx_vals[17](3..19) transpose(20)

    Encoding rules (mirrors parse_fx_values / parse_notes / songfile.py):
      note    — raw 0 = '---'; token IS the raw byte (no +1 offset)
      instr   — token 0 → EMPTY (0xFF); token k → raw k-1
      fx_cmd  — native LSDJ value; no offset
      fx_vals — all fields use +1 offset; decode = max(0, token-1)
                 nibble-packed commands (C,E,R,V,Z) reconstruct from two tokens

    FX_VALUE_KEYS indices used below:
      0=TABLE(A)  1=GROOVE(G)  2=HOP(H)   3=PAN(O)   4/5=CHORD(C)
      6/7=ENV(E)  8/9=RETRIG(R)  10/11=VIB(V)
      12=VOLUME(M)  13=WAVE(W)  14/15=RANDOM(Z)  16=CONTINUOUS(D,F,K,L,P,S,T)
    """
    note   = int(ch_tokens[0])
    instr  = int(ch_tokens[1])
    fx_cmd = int(ch_tokens[2])
    fx     = ch_tokens[3:3 + FX_VALUES_FEATURE_DIM]

    instr_raw = EMPTY if instr == 0 else instr - 1

    def dec(i):        return max(0, int(fx[i]) - 1)
    def nib(hi, lo):   return (dec(hi) & 0xF) << 4 | (dec(lo) & 0xF)

    if   fx_cmd == CMD_NULL: fxval_raw = 0
    elif fx_cmd == CMD_A:    fxval_raw = dec(0)
    elif fx_cmd == CMD_C:    fxval_raw = nib(4, 5)
    elif fx_cmd == CMD_D:    fxval_raw = dec(16)
    elif fx_cmd == CMD_E:    fxval_raw = nib(6, 7)
    elif fx_cmd == CMD_F:    fxval_raw = dec(16)
    elif fx_cmd == CMD_G:    fxval_raw = dec(1)
    elif fx_cmd == CMD_H:    fxval_raw = dec(2)
    elif fx_cmd == CMD_K:    fxval_raw = dec(16)
    elif fx_cmd == CMD_L:    fxval_raw = dec(16)
    elif fx_cmd == CMD_M:    fxval_raw = dec(12)
    elif fx_cmd == CMD_O:    fxval_raw = dec(3)
    elif fx_cmd == CMD_P:    fxval_raw = dec(16)
    elif fx_cmd == CMD_R:    fxval_raw = nib(8, 9)
    elif fx_cmd == CMD_S:    fxval_raw = dec(16)
    elif fx_cmd == CMD_T:    fxval_raw = dec(16)
    elif fx_cmd == CMD_V:    fxval_raw = nib(10, 11)
    elif fx_cmd == CMD_W:    fxval_raw = dec(13)
    elif fx_cmd == CMD_Z:    fxval_raw = nib(14, 15)
    else:                    fxval_raw = 0

    return note, instr_raw, fx_cmd, fxval_raw


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
        _safe_dec(tokens_dict[TABLE_ENV_VOLUME]),
        _safe_dec(tokens_dict[TABLE_ENV_DURATION]),
    ).ravel().astype(jnp.uint8).tolist()

    # Transposes: remove null offset
    transpose_bytes = (
        _safe_dec(tokens_dict[TABLE_TRANSPOSE])
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
    tokens = np.array(song_tokens, dtype=np.uint16)
    S = tokens.shape[0]
    pad = (STEPS_PER_PHRASE - S % STEPS_PER_PHRASE) % STEPS_PER_PHRASE
    if pad:
        tokens = np.pad(tokens, ((0, pad), (0, 0), (0, 0)))
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
    phrase_notes_out = np.full((NUM_PHRASES, STEPS_PER_PHRASE), EMPTY, dtype=np.uint16)
    phrase_instr_out = np.full((NUM_PHRASES, STEPS_PER_PHRASE), EMPTY, dtype=np.uint16)
    phrase_fx_cmd_out = np.zeros((NUM_PHRASES, STEPS_PER_PHRASE), dtype=np.uint16)
    phrase_fx_val_out = np.zeros(
        (NUM_PHRASES, STEPS_PER_PHRASE, FX_VALUES_FEATURE_DIM), dtype=np.uint16
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
            elif next_phrase_id >= NUM_PHRASES:
                # Phrase budget exhausted — fall back to nearest allocated phrase.
                pid = _nearest_phrase(
                    p_notes, p_instr, p_fx_cmd,
                    phrase_notes_out, phrase_instr_out, phrase_fx_cmd_out,
                    next_phrase_id,
                )
                warnings.warn(
                    f"Phrase budget exhausted (>{NUM_PHRASES} unique phrases). "
                    f"ch={ch}, block={p} mapped to nearest existing phrase {pid}."
                )
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

    # Phrase instruments: NULL token (0) → EMPTY (0xFF); token n → raw n-1.
    phrase_instr_bytes = jnp.where(
        phrase_instr_out == 0, EMPTY, phrase_instr_out - 1
    ).ravel().tolist()

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
