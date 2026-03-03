import jax.numpy as jnp
import jax.random as jr
import jax
import equinox as eqx
from functools import partial
from jaxtyping import Array, Key
from pe_lsdj.constants import (
    NUM_CHANNELS,
    NUM_INSTRUMENTS,
    NUM_SYNTHS,
    NUM_TABLES,
    NUM_GROOVES,
    INSTR_TABLE_COL,
    INSTR_SOFTSYNTH_COL,
    INSTR_WIDTH,
    TABLE_WIDTH,
    STEPS_PER_TABLE,
    CMD_A,
    CMD_G,
    CMD_H,
    CMD_NULL,
    SOFTSYNTH_WIDTH,
    WAVEFRAME_DIM,
    WAV,
)
from pe_lsdj.models.transformer import (
    TOKEN_HEADS,
    ENTITY_HEADS,
    _GROOVE_FX_COLS_ARRAY,
    _TABLE_FX_COLS_ARRAY,
    _TABLE_SCALAR_CAT_GROUPS,
    _TABLE_SCALAR_CONT_COLS_ARRAY,
    _TABLE_SCALAR_CONT_MAX_VALUES,
    _INSTR_SCALAR_CAT_GROUPS,
    _INSTR_SCALAR_CONT_COLS_ARRAY,
    _INSTR_SCALAR_CONT_MAX_VALUES,
    _SOFTSYNTH_CAT_GROUPS,
    _SOFTSYNTH_CONT_COLS_ARRAY,
    _SOFTSYNTH_CONT_MAX_VALUES,
    _TABLE_CAT_TRACE_MASK,
    _GROOVE_CONT_MAX,
    N_GROOVE_SLOTS,
    N_TABLE_SLOTS,
    instr_loss,
    groove_loss,
    table_loss,
    softsynth_loss,
    score_one_trace,
)
from pe_lsdj.embedding.song import SongBanks


def _next_free_idx(occupied_table: Array):
    return jnp.argmin(occupied_table[1:]) + 1


def _score_table_grooves(banks, heads, table_h):

    def score_groove(slot_idx):                                                                                                                           
        pred = heads.groove_decoder(table_h, slot_idx)                                                                                                     
        return jax.vmap(lambda row: groove_loss(pred, row))(banks.grooves[1:])
            
    groove_fit = jax.vmap(score_groove)(jnp.arange(N_GROOVE_SLOTS))  # (N_GROOVE_SLOTS, N_GROOVES)
    groove_ids = banks.tables[1:][:, _GROOVE_FX_COLS_ARRAY]          # (N_TABLES, N_GROOVE_SLOTS)
    active_slots = groove_ids > 0
    gathered = groove_fit[jnp.arange(N_GROOVE_SLOTS)[None, :],
        jnp.clip(groove_ids - 1, 0, None)]                           # (N_TABLES, N_GROOVE_SLOTS)
    return jnp.sum(jnp.where(active_slots, gathered, 0.0), axis=1)   # (N_TABLES,)


def _score_table_traces(banks, heads, table_h):

    all_trace_rows = banks.traces[1:]                     # (N_TRACES, TABLE_WIDTH)
    all_tgr_rows   = banks.grooves[
        all_trace_rows[:, _GROOVE_FX_COLS_ARRAY]          # (N_TRACES, N_GROOVE_SLOTS)
    ]                                                     # (N_TRACES, N_GROOVE_SLOTS, GROOVE_CONT_N)

    def trace_scores_for_slot(sidx):
        return jax.vmap(
            lambda tr, tgr: score_one_trace(heads, table_h, sidx, tr, tgr)
        )(all_trace_rows, all_tgr_rows)                   # (N_TRACES,)

    trace_fit = jax.vmap(trace_scores_for_slot)(
        jnp.arange(N_TABLE_SLOTS)
    )                                                     # (N_TABLE_SLOTS, N_TRACES)

    # Reduce to (N_TABLES,) via gather
    trace_ids = banks.tables[1:][:, _TABLE_FX_COLS_ARRAY] # (N_TABLES, N_TABLE_SLOTS)
    active    = trace_ids > 0
    gathered  = trace_fit[
        jnp.arange(N_TABLE_SLOTS)[None, :],
        jnp.clip(trace_ids - 1, 0, None)
    ]                                                     # (N_TABLES, N_TABLE_SLOTS)
    return jnp.sum(
        jnp.where(active, gathered, 0.0), axis=1
    )      


def _build_table(logits, key) -> Array:
    table = jnp.zeros(TABLE_WIDTH)
    categorical, continuous = logits['cat'], logits['cont']
    keys = jr.split(key, len(_TABLE_SCALAR_CAT_GROUPS))

    for i, (vocab_size, starts, cols) in enumerate(_TABLE_SCALAR_CAT_GROUPS):
        gather_idxs = starts[:, None] + jnp.arange(vocab_size)[None, :]
        field_logits = categorical[gather_idxs]
        cat_values = jr.categorical(
            keys[i],
            field_logits,
            axis=-1
        )
        table = table.at[cols].set(cat_values)

    cont_values = jnp.round(jax.nn.sigmoid(
        continuous
    ) * _TABLE_SCALAR_CONT_MAX_VALUES).astype(jnp.uint8)

    table = table.at[_TABLE_SCALAR_CONT_COLS_ARRAY].set(cont_values)

    return table


def _build_softsynth(logits, key) -> Array:
    """
    Sample softsynth parameters and waveframes from predicted logits.

    logits: dict with keys 'cat', 'cont', 'waveframes' (output of SoftSynthDecoder)
    Returns (synth_row, waveframe_row) — both uint8 arrays.
    """
    synth_row = jnp.zeros(SOFTSYNTH_WIDTH, dtype=jnp.uint8)
    keys = jr.split(key, len(_SOFTSYNTH_CAT_GROUPS))

    for i, (vocab_size, starts, cols) in enumerate(_SOFTSYNTH_CAT_GROUPS):
        gather_idxs = starts[:, None] + jnp.arange(vocab_size)[None, :]
        field_logits = logits['cat'][gather_idxs]
        cat_values = jr.categorical(keys[i], field_logits, axis=-1)
        synth_row = synth_row.at[cols].set(cat_values)

    cont_values = jnp.round(
        jax.nn.sigmoid(logits['cont']) * _SOFTSYNTH_CONT_MAX_VALUES
    ).astype(jnp.uint8)
    synth_row = synth_row.at[_SOFTSYNTH_CONT_COLS_ARRAY].set(cont_values)

    waveframe_row = jnp.round(
        jax.nn.sigmoid(logits['waveframes']) * 15.0
    ).astype(jnp.uint8)

    return synth_row, waveframe_row


# Column offsets in a flat TABLE_WIDTH row.
# Layout: env_vol(16) env_dur(16) transpose(16) cmd1(16) fxv1(16×17) cmd2(16) fxv2(16×17)
_EV_OFF  = 0
_ED_OFF  = 16
_TP_OFF  = 32
_C1_OFF  = 48
_FV1_OFF = 64
_C2_OFF  = 336
_FV2_OFF = 352
_V       = 17   # FX value dim (len(FX_VALUE_KEYS))
_TABLE_FX_IDX = 0   # FX_VALUE_KEYS[0] = TABLE_FX
_HOP_FX_IDX   = 2   # FX_VALUE_KEYS[2] = HOP_FX

# For each of the N_TABLE_SLOTS A-command value positions (order matches
# TABLE_FX_COL_INDICES = sorted({64+i*17, 352+i*17})):
#   slots 0..15  → FV1 TABLE_FX at step i, command col = 48+i  (cmd1)
#   slots 16..31 → FV2 TABLE_FX at step i, command col = 336+i (cmd2)
_TABLE_A_CMD_COLS_ARRAY = jnp.array(
    [_C1_OFF + s for s in range(STEPS_PER_TABLE)] +
    [_C2_OFF + s for s in range(STEPS_PER_TABLE)],
    dtype=jnp.int32,
)  # (N_TABLE_SLOTS,)


def _build_trace_for_table(table_row: Array, banks: SongBanks) -> Array:
    """
    Simulate execution of a table for STEPS_PER_TABLE steps, following H (hop)
    and A (jump) commands, to produce its execution trace.

    Matches the semantics of get_traces() in tokenize.py:
      - H at step i: read from hop_target, then advance from there.
      - A at step i: read from step 0 of banks.traces[target]; continue from
        step 1. CMD1 takes priority over CMD2.
      - env_vol/dur: taken directly from the root table (same heuristic as
        get_traces — env runs on its own timer, not traced through H/A).

    After an A jump we read from banks.traces[target], which is already
    resolved and contains no further A/H to follow.

    table_row : (TABLE_WIDTH,) float32 — output of _build_table
    Returns   : (TABLE_WIDTH,) uint8 trace row

    NOTE This does what tokenize.get_traces() does but more efficiently/
    purely in JAX. TODO: Refactor tokenize to match this efficiency
    """
    S = STEPS_PER_TABLE

    table_row = table_row.astype(jnp.uint8)

    def get_row(is_bank, bank_idx):
        return jax.lax.cond(
            is_bank,
            lambda: banks.traces[bank_idx],
            lambda: table_row,
        )

    def trace_step(carry, _):
        is_bank, bank_idx, step_L, step_R = carry
        row = get_row(is_bank, bank_idx)

        c1  = row[_C1_OFF + step_L]
        fv1 = jax.lax.dynamic_slice(row, (_FV1_OFF + step_L * _V,), (_V,))
        c2  = row[_C2_OFF + step_R]
        fv2 = jax.lax.dynamic_slice(row, (_FV2_OFF + step_R * _V,), (_V,))

        # Only follow H/A while in the new table; bank traces are pre-resolved.
        active = ~is_bank
        has_h1 = active & (c1 == CMD_H)
        has_h2 = active & (c2 == CMD_H)
        has_a1 = active & (c1 == CMD_A)
        has_a2 = active & (c2 == CMD_A)
        has_a  = has_a1 | has_a2

        a_target = jnp.where(
            has_a1,
            fv1[_TABLE_FX_IDX].astype(jnp.int32),
            fv2[_TABLE_FX_IDX].astype(jnp.int32),
        )
        hop1 = fv1[_HOP_FX_IDX].astype(jnp.int32) & 0x0F
        hop2 = fv2[_HOP_FX_IDX].astype(jnp.int32) & 0x0F

        # Resolve actual read positions (matches resolve_map logic)
        read_L = jnp.where(has_a, jnp.int32(0), jnp.where(has_h1, hop1, step_L))
        read_R = jnp.where(has_a, jnp.int32(0), jnp.where(has_h2, hop2, step_R))
        read_is_bank  = has_a | is_bank
        read_bank_idx = jnp.where(has_a, a_target, bank_idx)

        # Read data from the resolved position
        rrow   = get_row(read_is_bank, read_bank_idx)
        out_tp  = rrow[_TP_OFF  + read_L]
        out_c1  = rrow[_C1_OFF  + read_L]
        out_fv1 = jax.lax.dynamic_slice(rrow, (_FV1_OFF + read_L * _V,), (_V,))
        out_c2  = rrow[_C2_OFF  + read_R]
        out_fv2 = jax.lax.dynamic_slice(rrow, (_FV2_OFF + read_R * _V,), (_V,))

        # Advance cursors from resolved positions
        new_carry = (read_is_bank, read_bank_idx, (read_L + 1) % S, (read_R + 1) % S)

        step_out = jnp.concatenate([
            out_tp[None], out_c1[None], out_fv1, out_c2[None], out_fv2
        ])  # (37,)
        return new_carry, step_out

    init = (jnp.bool_(False), jnp.int32(0), jnp.int32(0), jnp.int32(0))
    _, trace_steps = jax.lax.scan(trace_step, init, None, length=S)
    # trace_steps: (S, 37) = [tp(1), c1(1), fv1(17), c2(1), fv2(17)]

    # env_vol/dur: use root table order (not traced), matching get_traces heuristic
    return jnp.concatenate([
        table_row[_EV_OFF:_EV_OFF + S],   # env_vol (16)
        table_row[_ED_OFF:_ED_OFF + S],   # env_dur (16)
        trace_steps[:, 0],                # tp      (16)
        trace_steps[:, 1],                # c1      (16)
        trace_steps[:, 2:19].ravel(),     # fv1     (16×17)
        trace_steps[:, 19],               # c2      (16)
        trace_steps[:, 20:37].ravel(),    # fv2     (16×17)
    ])  # (TABLE_WIDTH,)


def match_groove(
    heads,
    banks: SongBanks,
    groove_ctx: Array,  # table_entity_dim context for the groove decoder
    slot_idx: Array,    # scalar int32 — 0..N_GROOVE_SLOTS-1
    threshold: float,
) -> tuple[Array, SongBanks]:
    """
    Find or create a bank groove for one G-command slot.

    Grooves are leaf entities: all-continuous, no sub-entities, no key needed.
    groove_ctx distinguishes which table/trace level we are at; the groove
    decoder adds per-slot embeddings internally.

    Returns (groove_id, updated_banks).  groove_id is 1-indexed (0 = null).
    """
    predicted = heads.groove_decoder(groove_ctx, slot_idx)     # (GROOVE_CONT_N,)

    groove_scores = jax.vmap(
        lambda row: groove_loss(predicted, row)
    )(banks.grooves[1:])                                        # (NUM_GROOVES,)

    best_score_pos = jnp.argmin(
        jnp.where(banks.grooves_occupied[1:], groove_scores, jnp.inf)
    )
    best_groove_idx = best_score_pos + 1

    use_existing = (
        (groove_scores[best_score_pos] <= threshold)
        | jnp.all(banks.grooves_occupied[1:])
    )

    groove_id = jax.lax.cond(
        use_existing,
        lambda: best_groove_idx,
        lambda: _next_free_idx(banks.grooves_occupied),
    )

    def _create_new_groove():
        groove_row = jnp.round(
            jax.nn.sigmoid(predicted) * _GROOVE_CONT_MAX
        ).astype(jnp.uint8)
        return banks._replace(
            grooves=banks.grooves.at[groove_id].set(groove_row),
            grooves_occupied=banks.grooves_occupied.at[groove_id].set(True),
        )

    banks_out = jax.lax.cond(use_existing, lambda: banks, _create_new_groove)
    return groove_id, banks_out


def match_trace(
    key: Key,
    heads,
    banks: SongBanks,
    table_h: Array,
    slot_idx: Array,            # scalar int32 — which A-command slot (0..N_TABLE_SLOTS-1)
    threshold: float,
) -> tuple[Array, SongBanks]:
    """
    Find or create a bank trace for one A-command slot of a new table.

    The model predicts trace content conditioned on (table_h, slot_idx).
    A/H commands are masked: depth-1 traces contain neither, so trace = table.

    Returns (trace_bank_id, updated_banks).
    """
    # Score all existing bank traces for this slot
    all_trace_rows = banks.traces[1:]                                          # (N_TABLES, TABLE_WIDTH)
    tgr_ids  = all_trace_rows[:, _GROOVE_FX_COLS_ARRAY].astype(jnp.int32)    # (N_TABLES, N_GROOVE_SLOTS)
    tgr_rows = banks.grooves[tgr_ids]                                         # (N_TABLES, N_GROOVE_SLOTS, GROOVE_CONT_N)

    trace_scores = jax.vmap(
        lambda tr, tgr: score_one_trace(heads, table_h, slot_idx, tr, tgr)
    )(all_trace_rows, tgr_rows)                                                # (N_TABLES,)

    best_score_pos = jnp.argmin(
        jnp.where(banks.tables_occupied[1:], trace_scores, jnp.inf)
    )
    best_trace_idx = best_score_pos + 1  # 1-indexed bank ID

    use_existing = (
        (trace_scores[best_score_pos] <= threshold)
        | jnp.all(banks.tables_occupied[1:])
    )

    trace_id = jax.lax.cond(
        use_existing,
        lambda: best_trace_idx,
        lambda: _next_free_idx(banks.tables_occupied),
    )

    def _create_new_trace():
        # Predict trace content for this slot (A/H masked → no commands to resolve)
        trace_ctx  = table_h + heads.table_decoder.slot_embeds[slot_idx]
        trace_h    = jax.nn.gelu(heads.table_decoder.linear_in(trace_ctx))
        trace_logits = {
            'cat':  jnp.where(_TABLE_CAT_TRACE_MASK, -jnp.inf,
                               heads.table_decoder.cat_out(trace_h)),
            'cont': heads.table_decoder.cont_out(trace_h),
        }
        trace_content = _build_table(trace_logits, key)

        # G commands are not masked in traces — resolve groove slots.
        # Context is trace_ctx: encodes both parent-table slot and trace level.
        def resolve_groove_slot(carry, groove_slot_idx):
            rc, bks = carry
            cmd_col = _TABLE_A_CMD_COLS_ARRAY[groove_slot_idx]
            is_g = rc[cmd_col] == CMD_G
            gid, bks = jax.lax.cond(
                is_g,
                lambda: match_groove(heads, bks, trace_ctx, groove_slot_idx, threshold),
                lambda: (jnp.int32(0), bks),
            )
            rc = jax.lax.cond(
                is_g,
                lambda: rc.at[_GROOVE_FX_COLS_ARRAY[groove_slot_idx]].set(
                    gid.astype(rc.dtype)
                ),
                lambda: rc,
            )
            return (rc, bks), None

        (trace_content, banks_g), _ = jax.lax.scan(
            resolve_groove_slot,
            (trace_content, banks),
            jnp.arange(N_GROOVE_SLOTS, dtype=jnp.int32),
        )

        # Depth-1: no A/H → execution trace == raw content (groove IDs resolved)
        return banks_g._replace(
            tables=banks_g.tables.at[trace_id].set(trace_content),
            traces=banks_g.traces.at[trace_id].set(trace_content),
            tables_occupied=banks_g.tables_occupied.at[trace_id].set(True),
        )

    banks_out = jax.lax.cond(use_existing, lambda: banks, _create_new_trace)
    return trace_id, banks_out


def _create_new_table(key, heads, banks, predicted_table_logits, table_h, table_id, threshold):
    """
    Build and store a new table (and its trace) at `table_id` in `banks`.

    1. Sample raw table content from `predicted_table_logits`.
       TABLE_FX and GROOVE_FX values start at 0 (null) — they are not sampled.
    2. Scan over each slot: CMD_A → match_trace (fills TABLE_FX);
       CMD_G → match_groove (fills GROOVE_FX). Both may create new bank entries.
    3. Simulate execution of the finalised table (H/A resolved) to produce
       its execution trace, stored at `banks.traces[table_id]`.
    """
    key_table, key_traces = jr.split(key)
    raw_table = _build_table(predicted_table_logits, key_table)

    # Pre-reserve table_id so inner traces (A-cmd sub-tables) don't steal this slot.
    banks = banks._replace(tables_occupied=banks.tables_occupied.at[table_id].set(True))

    trace_keys = jr.split(key_traces, N_TABLE_SLOTS)  # (N_TABLE_SLOTS, 2)

    def resolve_slot(carry, inputs):
        rt, bks = carry
        slot_idx, sub_key = inputs

        cmd_col = _TABLE_A_CMD_COLS_ARRAY[slot_idx]
        cmd     = rt[cmd_col]
        is_a_cmd = cmd == CMD_A
        is_g_cmd = cmd == CMD_G

        # A command: match/create a depth-1 trace
        traced_id, bks = jax.lax.cond(
            is_a_cmd,
            lambda: match_trace(sub_key, heads, bks, table_h, slot_idx, threshold),
            lambda: (jnp.int32(0), bks),
        )
        rt = jax.lax.cond(
            is_a_cmd,
            lambda: rt.at[_TABLE_FX_COLS_ARRAY[slot_idx]].set(traced_id.astype(rt.dtype)),
            lambda: rt,
        )

        # G command: match/create a groove
        groove_id, bks = jax.lax.cond(
            is_g_cmd,
            lambda: match_groove(heads, bks, table_h, slot_idx, threshold),
            lambda: (jnp.int32(0), bks),
        )
        rt = jax.lax.cond(
            is_g_cmd,
            lambda: rt.at[_GROOVE_FX_COLS_ARRAY[slot_idx]].set(groove_id.astype(rt.dtype)),
            lambda: rt,
        )

        return (rt, bks), None

    slot_indices = jnp.arange(N_TABLE_SLOTS, dtype=jnp.int32)
    (raw_table, banks), _ = jax.lax.scan(
        resolve_slot, (raw_table, banks), (slot_indices, trace_keys)
    )

    outer_trace = _build_trace_for_table(raw_table, banks)

    return banks._replace(
        tables=banks.tables.at[table_id].set(raw_table),
        traces=banks.traces.at[table_id].set(outer_trace),
        tables_occupied=banks.tables_occupied.at[table_id].set(True),
    )


def match_table(
    key,
    heads,
    banks,
    predicted_table_logits,
    table_hidden,
    table_match_threshold,
) -> tuple[Array, SongBanks]:
    table_scalar_scores = jax.vmap(
        lambda target: sum(table_loss(predicted_table_logits, target))
    )(banks.tables[1:])

    table_groove_scores = _score_table_grooves(banks, heads, table_hidden)
    table_trace_scores  = _score_table_traces(banks, heads, table_hidden)
    table_scores = table_scalar_scores + table_groove_scores + table_trace_scores

    best_score_pos = jnp.argmin(
        jnp.where(banks.tables_occupied[1:], table_scores, jnp.inf)
    )  # 0-indexed into table_scores
    best_table_idx = best_score_pos + 1  # 1-indexed bank/token ID

    use_existing_table = (
        (table_scores[best_score_pos] <= table_match_threshold)
        | jnp.all(banks.tables_occupied[1:])
    )
    table_id = jax.lax.cond(
        use_existing_table,
        lambda: best_table_idx,
        lambda: _next_free_idx(banks.tables_occupied),
    )

    banks_out = jax.lax.cond(
        use_existing_table,
        lambda: banks,
        lambda: _create_new_table(
            key, heads, banks, predicted_table_logits, table_hidden, table_id, table_match_threshold
        ),
    )

    return table_id, banks_out


def match_softsynth(
    key: Key,
    banks: SongBanks,
    softsynth_preds: dict,  # precomputed from output_heads.generation_outputs
    threshold: float,
) -> tuple[Array, SongBanks]:
    """
    Find or create a bank softsynth for the current instrument.

    Scores all occupied bank entries using softsynth_loss; reuses the best
    match if its score <= threshold (or if the bank is full), otherwise
    creates a new entry at the next free synth slot.

    softsynth_preds: {'cat', 'cont', 'waveframes'} from generation_outputs.

    Returns (synth_id, updated_banks).  synth_id is 1-indexed (0 = null).
    """
    synth_scores = jax.vmap(
        lambda row: sum(softsynth_loss(softsynth_preds, row))
    )(banks.synth_waves[1:])                                       # (NUM_SYNTHS,)

    best_score_pos = jnp.argmin(
        jnp.where(banks.synths_occupied[1:], synth_scores, jnp.inf)
    )
    best_synth_idx = best_score_pos + 1  # 1-indexed bank ID

    use_existing = (
        (synth_scores[best_score_pos] <= threshold)
        | jnp.all(banks.synths_occupied[1:])
    )

    synth_id = jax.lax.cond(
        use_existing,
        lambda: best_synth_idx,
        lambda: _next_free_idx(banks.synths_occupied),
    )

    def _create_new_synth():
        synth_row, wf_row = _build_softsynth(softsynth_preds, key)
        return banks._replace(
            softsynths=banks.softsynths.at[synth_id].set(synth_row),
            waveframes=banks.waveframes.at[synth_id].set(wf_row),
            synth_waves=banks.synth_waves.at[synth_id].set(
                jnp.concatenate([synth_row, wf_row])
            ),
            synths_occupied=banks.synths_occupied.at[synth_id].set(True),
        )

    banks_out = jax.lax.cond(use_existing, lambda: banks, _create_new_synth)
    return synth_id, banks_out


def _build_instrument(logits, key, instr_type=None) -> Array:
    """
    Sample instrument fields from predicted logits.

    instr_type: if provided (a traced scalar), overrides the TYPE_ID field (col 0)
                instead of re-sampling it from the categorical logits. Pass this
                when the type has already been sampled externally (e.g. in
                match_instrument) to avoid a redundant independent draw.
    """
    instr = jnp.zeros(INSTR_WIDTH)
    categorical, continuous = logits['cat'], logits['cont']
    keys = jr.split(key, len(_INSTR_SCALAR_CAT_GROUPS))

    for i, (vocab_size, starts, cols) in enumerate(_INSTR_SCALAR_CAT_GROUPS):
        gather_idxs = starts[:, None] + jnp.arange(vocab_size)[None, :]
        field_logits = categorical[gather_idxs]
        cat_values = jr.categorical(keys[i], field_logits, axis=-1)
        instr = instr.at[cols].set(cat_values)

    cont_values = jnp.round(jax.nn.sigmoid(
        continuous
    ) * _INSTR_SCALAR_CONT_MAX_VALUES).astype(jnp.uint8)
    instr = instr.at[_INSTR_SCALAR_CONT_COLS_ARRAY].set(cont_values)

    if instr_type is not None:
        instr = instr.at[0].set(instr_type.astype(instr.dtype))

    return instr



def match_instrument(
    key: Key,
    heads,
    banks: SongBanks,
    instr_preds: dict,      # precomputed from generation_outputs['instr']
    instr_table_ctx: Array, # precomputed from generation_outputs latents
    instr_threshold: float,
    table_threshold: float,
    softsynth_threshold: float,
) -> tuple[Array, SongBanks]:
    """
    Find or create a bank instrument for the current step/channel.

    TYPE_ID (col 0, vocab 5) encodes PU/WAV/KIT/NOI (values 1-4) or NULL (0).
    If the predicted type is NULL, returns (0, banks) immediately — no bank update.

    For non-null predictions: scores all occupied bank instruments using instr_loss
    (scalar + table + softsynth sub-losses); reuses the best match if score <=
    instr_threshold (or if the bank is full), otherwise creates a new entry which
    itself calls match_table and match_softsynth to resolve sub-entities.

    instr_preds:      {'cat', 'cont', 'softsynth', 'table'} from generation_outputs.
    instr_table_ctx:  (table_entity_dim,) context for instrument's table matching.

    Returns (instr_id, updated_banks).  instr_id is 1-indexed (0 = null).
    """
    # TYPE_ID is the first categorical field: offset 0, vocab 5 in instr_preds['cat'].
    # Sample it once here so the gate and the built instrument agree.
    instr_type = jr.categorical(jr.fold_in(key, 200), instr_preds['cat'][:5])

    def _do_match():
        instr_scores = jax.vmap(
            lambda row: sum(instr_loss(instr_preds, row, banks))
        )(banks.instruments[1:])                                       # (NUM_INSTRUMENTS,)

        best_score_pos = jnp.argmin(
            jnp.where(banks.instrs_occupied[1:], instr_scores, jnp.inf)
        )
        best_instr_idx = best_score_pos + 1  # 1-indexed bank ID

        use_existing = (
            (instr_scores[best_score_pos] <= instr_threshold)
            | jnp.all(banks.instrs_occupied[1:])
        )

        instr_id = jax.lax.cond(
            use_existing,
            lambda: best_instr_idx,
            lambda: _next_free_idx(banks.instrs_occupied),
        )

        def _create_new_instrument():
            key_instr, key_table, key_synth = jr.split(key, 3)

            raw_instr = _build_instrument(
                {'cat': instr_preds['cat'], 'cont': instr_preds['cont']},
                key_instr,
                instr_type=instr_type,
            )

            table_id, banks_t = match_table(
                key_table, heads, banks,
                instr_preds['table'], instr_table_ctx, table_threshold,
            )
            # Softsynth is only meaningful for WAV instruments (TYPE_ID == WAV == 2).
            synth_id, banks_ts = jax.lax.cond(
                instr_type == WAV,
                lambda: match_softsynth(
                    key_synth, banks_t, instr_preds['softsynth'], softsynth_threshold,
                ),
                lambda: (jnp.int32(0), banks_t),
            )

            raw_instr = raw_instr.at[INSTR_TABLE_COL].set(
                table_id.astype(raw_instr.dtype)
            )
            raw_instr = raw_instr.at[INSTR_SOFTSYNTH_COL].set(
                synth_id.astype(raw_instr.dtype)
            )

            return banks_ts._replace(
                instruments=banks_ts.instruments.at[instr_id].set(raw_instr),
                instrs_occupied=banks_ts.instrs_occupied.at[instr_id].set(True),
            )

        banks_out = jax.lax.cond(use_existing, lambda: banks, _create_new_instrument)
        return instr_id, banks_out

    return jax.lax.cond(
        instr_type == 0,
        lambda: (jnp.int32(0), banks),
        _do_match,
    )
    return instr_id, banks_out
    

def resolve_step(
    heads,
    banks: SongBanks,
    key: Key,
    logits_dict: dict[str, Array],
    latents: dict[str, Array],
    instr_match_threshold: float,
    table_match_threshold: float,
    groove_match_threshold: float,
    synth_match_threshold: float,
) -> tuple[Array, SongBanks]:
    """
    Resolve one generation step across all 4 channels.

    Channels are processed sequentially (lax.scan) because banks is shared
    mutable state: entities created for one channel are visible to the next.

    For each channel:
      1. Sample all non-entity token fields (note, fx_cmd, fx_vals, transpose)
         via jr.categorical, keyed by token position.
      2. Match / create instrument → col 1 (instr_id).
      3. If fx_cmd == CMD_A: match / create phrase-level table → col 3 (TABLE_FX).
      4. If fx_cmd == CMD_G: match / create phrase-level groove → col 4 (GROOVE_FX).

    logits_dict, latents: output of generation_outputs, vmapped over 4 channels —
                          each leaf has a leading channel dim that scan slices per step.

    Returns (next_tokens, banks_out) where next_tokens has shape (NUM_CHANNELS, 21).
    """
    ch_keys = jr.split(key, NUM_CHANNELS)  # (NUM_CHANNELS, 2)

    def resolve_channel(banks, inputs):
        ch_key, ch_logits, ch_latents = inputs

        # 1. Sample non-entity token fields.
        #    TOKEN_HEADS[name] = (pos, vocab); positions 1, 3, 4 are entity refs
        #    and are absent from TOKEN_HEADS — they are filled in below.
        #    jr.fold_in(key, pos) gives a unique subkey per field without
        #    allocating a split array; positions 0-20 are safely below 100.
        next_chan_tokens = jnp.zeros(21, dtype=jnp.uint8)
        for name, (pos, _) in TOKEN_HEADS.items():
            val = jr.categorical(jr.fold_in(ch_key, pos), ch_logits[name])
            next_chan_tokens = next_chan_tokens.at[pos].set(val.astype(jnp.uint8))

        # 2. Instrument → col 1.
        instr_id, banks = match_instrument(
            jr.fold_in(ch_key, 100),
            heads, banks,
            ch_logits['instr'], ch_latents['instr_table_ctx'],
            instr_match_threshold, table_match_threshold, synth_match_threshold,
        )
        next_chan_tokens = next_chan_tokens.at[
            ENTITY_HEADS['instr_id']
        ].set(instr_id.astype(jnp.uint8))

        # 3. CMD_A → phrase-level table → col 3 (TABLE_FX = fx_vals[0]).
        fx_cmd = next_chan_tokens[2]
        table_id, banks = jax.lax.cond(
            fx_cmd == CMD_A,
            lambda: match_table(
                jr.fold_in(ch_key, 101), heads, banks,
                ch_logits['table'], ch_latents['table_ctx'],
                table_match_threshold,
            ),
            lambda: (jnp.int32(0), banks),
        )
        next_chan_tokens = next_chan_tokens.at[
            ENTITY_HEADS['table_id']
        ].set(table_id.astype(jnp.uint8))

        # 4. CMD_G → phrase-level groove → col 4 (GROOVE_FX = fx_vals[1]).
        groove_id, banks = jax.lax.cond(
            fx_cmd == CMD_G,
            lambda: match_groove(
                heads, banks,
                ch_latents['phrase_groove_ctx'], N_GROOVE_SLOTS,
                groove_match_threshold,
            ),
            lambda: (jnp.int32(0), banks),
        )
        next_chan_tokens = next_chan_tokens.at[
            ENTITY_HEADS['groove_id']
        ].set(groove_id.astype(jnp.uint8))

        return banks, next_chan_tokens

    banks_out, next_tokens = jax.lax.scan(
        resolve_channel, banks, (ch_keys, logits_dict, latents)
    )
    return next_tokens, banks_out


def generate_step(
    carry: tuple[Array, SongBanks], 
    key: Key, 
    model,
    instr_match_threshold: float,
    table_match_threshold: float,
    groove_match_threshold: float,
    softsynth_match_threshold: float,
) -> tuple[Array, SongBanks]:
    """
    `carry` contains 
        (generated_sequence, song_banks)
    """
    input_tokens, banks_in = carry

    hiddens                   = model.encode(input_tokens)                    # (S, 4, d_model)
    last                      = hiddens[-1]                                   # (4, d_model)
    logits_dict, latents      = jax.vmap(model.output_heads.generation_outputs)(last)

    # sample non-entity tokens, do bank matching, update banks, build next token
    next_token, banks_out = resolve_step(
        model.output_heads,
        banks_in,
        key,
        logits_dict,
        latents,
        instr_match_threshold,
        table_match_threshold,
        groove_match_threshold,
        softsynth_match_threshold,
    )

    # Slide the context window one step forward (fixed carry shape for lax.scan)
    new_context = jnp.concatenate([input_tokens[1:], next_token[None]], axis=0)
    return (new_context, banks_out), next_token


DEFAULT_NUM_STEPS = 128

def generate(
    model: eqx.Module, 
    input_tokens: Array,
    key: Key,
    banks: SongBanks | None = None,
    num_steps: int = DEFAULT_NUM_STEPS,
    instr_match_threshold: float = 0.05,
    groove_match_threshold: float = 0.05,
    table_match_threshold: float = 0.05,
    softsynth_match_threshold: float = 0.05,
) -> tuple[Array, SongBanks]:
    """
    Generate a (num_steps, NUM_CHANNELS=4, feature_dim) sample conditioned on `input_tokens`.
    Use `banks` as prior entities.

    Returns (input_tokens ++ new_tokens, final_banks).
    """
    keys = jr.split(key, num_steps)   # (num_steps, 2) — one key per step

    banks = banks or SongBanks.default()

    generate_step_fn = partial(
        generate_step,
        model=model,
        instr_match_threshold=instr_match_threshold,
        table_match_threshold=table_match_threshold,
        groove_match_threshold=groove_match_threshold,
        softsynth_match_threshold=softsynth_match_threshold,
    )

    (_, final_banks), new_tokens = jax.lax.scan(
        generate_step_fn, (input_tokens, banks), keys
    )
    return jnp.concatenate([input_tokens, new_tokens], axis=0), final_banks
