import json
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Key

from pe_lsdj.embedding.song import SequenceEmbedder, SongBanks
from pe_lsdj.constants import *

# ---------------------------------------------------------------------------
# Logit groups: direct softmax heads, same-vocab members batched.
# ---------------------------------------------------------------------------
LOGIT_GROUPS = {
    'note':          [('note',             0, NUM_NOTES)],
    'fx_cmd':        [('fx_cmd',           2, 19)],
    'byte_fx':       [('hop_fx',           5, 257),
                      ('volume_fx',       15, 257),
                      ('continuous_fx',   19, 257)],
    'small_enum_fx': [('pan_fx',           6, 5),
                      ('wave_fx',         16, 5)],
    'nibble_fx':     [('chord_fx_1',       7, 17), ('chord_fx_2',       8, 17),
                      ('env_fx_vol',       9, 17), ('env_fx_fade',      10, 17),
                      ('retrig_fx_fade',  11, 17), ('retrig_fx_rate',   12, 17),
                      ('vibrato_fx_speed',13, 17), ('vibrato_fx_depth', 14, 17),
                      ('random_fx_l',     17, 17), ('random_fx_r',      18, 17)],
    'transpose':     [('transpose',       20, 256)],
}

TOKEN_HEADS = {}
for _members in LOGIT_GROUPS.values():
    for _name, _pos, _vocab in _members:
        TOKEN_HEADS[_name] = (_pos, _vocab)

# Token positions for the three entity roots in song_tokens.
ENTITY_HEADS = {
    'instr_id':  1,
    'table_id':  3,
    'groove_id': 4,
}

# ---------------------------------------------------------------------------
# Entity parameter field specs — 3-tuples: (name, vocab, is_continuous).
#
# is_continuous mirrors the embedder:
#   True  → GatedNormedEmbedder (byte/nibble ordinal; use MSE regression)
#   False → EnumEmbedder or EntityEmbedder (discrete category; use CE)
# ---------------------------------------------------------------------------

INSTR_FIELD_SPECS = [
    (TYPE_ID,            5, False),  # enum: PU/WAV/KIT/NOI
    (TABLE,             33, False),  # entity ref  ← index 1
    (TABLE_ON_OFF,       2, False),  # bool
    (TABLE_AUTOMATE,     2, False),  # bool
    (AUTOMATE_2,         2, False),  # bool
    (PAN,                5, False),  # enum: Off/L/R/LR
    (VIBRATO_TYPE,       5, False),  # enum: HF/saw/sine/square
    (VIBRATO_DIRECTION,  3, False),  # enum: down/up
    (ENV_VOLUME,        17, True),   # nibble: envelope volume
    (ENV_FADE,          17, True),   # nibble: fade duration
    (LENGTH,            65, True),   # 6-bit: note duration
    (LENGTH_LIMITED,     3, False),  # bool
    (SWEEP,            257, True),   # byte: sweep intensity
    (VOLUME,             5, False),  # enum: predefined output levels
    (PHASE_TRANSPOSE,  257, True),   # byte: pitch offset
    (WAVE,               5, False),  # enum: duty cycle
    (PHASE_FINETUNE,    17, True),   # nibble: finetune
    (SOFTSYNTH_ID,      17, False),  # entity ref  ← index 17
    (REPEAT,            17, True),   # nibble: repeat count
    (PLAY_TYPE,          5, False),  # enum: once/loop/pingpong/manual
    (WAVE_LENGTH,       17, True),   # nibble: waveform length
    (SPEED,             17, True),   # nibble: playback speed
    (KEEP_ATTACK_1,      3, False),  # bool
    (KEEP_ATTACK_2,      3, False),  # bool
    (KIT_1_ID,          65, True),   # 6-bit: sample index (ordinal)
    (KIT_2_ID,          65, True),   # 6-bit: sample index
    (LENGTH_KIT_1,     257, True),   # byte: sample length
    (LENGTH_KIT_2,     257, True),   # byte
    (LOOP_KIT_1,         3, False),  # bool
    (LOOP_KIT_2,         3, False),  # bool
    (OFFSET_KIT_1,     257, True),   # byte: playback offset
    (OFFSET_KIT_2,     257, True),   # byte
    (HALF_SPEED,         3, False),  # bool
    (PITCH,            257, True),   # byte: pitch
    (DISTORTION_TYPE,    5, False),  # enum: clip/shape/shape2/wrap
]
assert len(INSTR_FIELD_SPECS) == INSTR_WIDTH

# FX value fields in FX_VALUE_KEYS order (j=0..16).
_FX_VAL_VOCABS = [33, 33, 257, 5, 17, 17, 17, 17, 17, 17, 17, 17, 257, 5, 17, 17, 257]
_FX_VAL_IS_CONT = [
    False,  # j=0:  TABLE_FX  — entity ref
    False,  # j=1:  GROOVE_FX — entity ref
    True,   # j=2:  HOP_FX    — byte
    False,  # j=3:  PAN_FX    — enum: Off/L/R/LR
    True,   # j=4:  CHORD_FX_1 — nibble: semitone offset
    True,   # j=5:  CHORD_FX_2 — nibble
    True,   # j=6:  ENV_FX_VOL — nibble
    True,   # j=7:  ENV_FX_FADE — nibble
    True,   # j=8:  RETRIG_FADE — nibble
    True,   # j=9:  RETRIG_RATE — nibble
    True,   # j=10: VIB_SPEED  — nibble
    True,   # j=11: VIB_DEPTH  — nibble
    True,   # j=12: VOLUME_FX  — byte
    False,  # j=13: WAVE_FX   — enum: duty cycle
    True,   # j=14: RANDOM_L  — nibble
    True,   # j=15: RANDOM_R  — nibble
    True,   # j=16: CONT_FX   — byte (delay/finetune/slide/pitch/sweep/tempo)
]

TABLE_FIELD_SPECS = (
    [(f'env_vol_{i}',       17, True)  for i in range(STEPS_PER_TABLE)] +
    [(f'env_dur_{i}',       17, True)  for i in range(STEPS_PER_TABLE)] +
    [(f'transpose_{i}',    257, True)  for i in range(STEPS_PER_TABLE)] +
    [(f'fx1_{i}',           19, False) for i in range(STEPS_PER_TABLE)] +
    [(f'fx1_val_{i}_{j}', _FX_VAL_VOCABS[j], _FX_VAL_IS_CONT[j])
     for i in range(STEPS_PER_TABLE) for j in range(17)] +
    [(f'fx2_{i}',           19, False) for i in range(STEPS_PER_TABLE)] +
    [(f'fx2_val_{i}_{j}', _FX_VAL_VOCABS[j], _FX_VAL_IS_CONT[j])
     for i in range(STEPS_PER_TABLE) for j in range(17)]
)
assert len(TABLE_FIELD_SPECS) == 624

# Grooves: ALL fields are continuous (nibble timing values).
GROOVE_FIELD_SPECS = [
    (f'step{i}_{tick}', 17, True)
    for i in range(STEPS_PER_GROOVE) for tick in ('even', 'odd')
]

SOFTSYNTH_FIELD_SPECS = [
    ('waveform',               4, False),  # enum: sawtooth/square/sine
    ('filter_type',            5, False),  # enum: LP/HP/BP/AP
    ('filter_resonance',     257, True),   # byte: Q level
    ('distortion',             3, False),  # enum: clip/wrap
    ('phase_type',             4, False),  # enum: normal/resync/resync2
    ('start_volume',         257, True),   # byte: initial amplitude
    ('start_filter_cutoff',  257, True),   # byte: initial cutoff
    ('start_phase_amount',   257, True),   # byte: initial phase mod
    ('start_vertical_shift', 257, True),   # byte: initial DC offset
    ('end_volume',           257, True),   # byte: final amplitude
    ('end_filter_cutoff',    257, True),   # byte: final cutoff
    ('end_phase_amount',     257, True),   # byte: final phase mod
    ('end_vertical_shift',   257, True),   # byte: final DC offset
]
assert len(SOFTSYNTH_FIELD_SPECS) == SOFTSYNTH_WIDTH

# ---------------------------------------------------------------------------
# Scalar sub-specs: exclude sub-entity reference columns.
# ---------------------------------------------------------------------------

_INSTR_TABLE_COL     = 1
_INSTR_SOFTSYNTH_COL = 17

INSTR_SCALAR_SPECS = [
    spec for i, spec in enumerate(INSTR_FIELD_SPECS)
    if i not in (_INSTR_TABLE_COL, _INSTR_SOFTSYNTH_COL)
]
INSTR_SCALAR_COL_INDICES = [
    i for i in range(len(INSTR_FIELD_SPECS))
    if i not in (_INSTR_TABLE_COL, _INSTR_SOFTSYNTH_COL)
]
assert len(INSTR_SCALAR_SPECS) == INSTR_WIDTH - 2

_TABLE_FX_COLS  = frozenset(
    [64  + i * 17     for i in range(STEPS_PER_TABLE)] +
    [352 + i * 17     for i in range(STEPS_PER_TABLE)]
)
_GROOVE_FX_COLS = frozenset(
    [64  + i * 17 + 1 for i in range(STEPS_PER_TABLE)] +
    [352 + i * 17 + 1 for i in range(STEPS_PER_TABLE)]
)

TABLE_SCALAR_SPECS = [
    spec for i, spec in enumerate(TABLE_FIELD_SPECS)
    if i not in _TABLE_FX_COLS and i not in _GROOVE_FX_COLS
]
TABLE_SCALAR_COL_INDICES = [
    i for i in range(len(TABLE_FIELD_SPECS))
    if i not in _TABLE_FX_COLS and i not in _GROOVE_FX_COLS
]
assert len(TABLE_SCALAR_SPECS) == 560

TABLE_FX_COL_INDICES   = sorted(_TABLE_FX_COLS)
GROOVE_FX_COL_INDICES  = sorted(_GROOVE_FX_COLS)

_TABLE_FX_COLS_ARRAY  = jnp.array(TABLE_FX_COL_INDICES,  dtype=jnp.int32)
_GROOVE_FX_COLS_ARRAY = jnp.array(GROOVE_FX_COL_INDICES, dtype=jnp.int32)

# ---------------------------------------------------------------------------
# Cat / continuous split.
#
# For each entity-type spec list, split into:
#   cat_specs:  [(name, vocab)]  — discrete fields → CE loss
#   cat_cols:   [int]            — column indices into bank row
#   cont_cols:  [int]            — column indices for continuous fields
#   cont_maxs:  [float]          — vocab-1 per field (for token/(vocab-1) norm)
# ---------------------------------------------------------------------------

def _split(specs, col_indices=None):
    if col_indices is None:
        col_indices = list(range(len(specs)))
    cat_specs = [(n, v)    for (n, v, c), _ in zip(specs, col_indices) if not c]
    cat_cols  = [col       for (_, _, c),  col in zip(specs, col_indices) if not c]
    cont_cols = [col       for (_, _, c),  col in zip(specs, col_indices) if c]
    cont_maxs = [float(v - 1) for (_, v, c) in specs if c]
    return cat_specs, cat_cols, cont_cols, cont_maxs

(INSTR_SCALAR_CAT_SPECS, INSTR_SCALAR_CAT_COL_INDICES,
 _instr_cont_cols, _instr_cont_maxs) = _split(INSTR_SCALAR_SPECS, INSTR_SCALAR_COL_INDICES)
INSTR_SCALAR_CONT_N             = len(_instr_cont_cols)
_INSTR_SCALAR_CONT_COLS_ARRAY   = jnp.array(_instr_cont_cols, dtype=jnp.int32)
_INSTR_SCALAR_CONT_MAX_VALUES   = jnp.array(_instr_cont_maxs, dtype=jnp.float32)

(TABLE_SCALAR_CAT_SPECS, TABLE_SCALAR_CAT_COL_INDICES,
 _table_cont_cols, _table_cont_maxs) = _split(TABLE_SCALAR_SPECS, TABLE_SCALAR_COL_INDICES)
TABLE_SCALAR_CONT_N             = len(_table_cont_cols)
_TABLE_SCALAR_CONT_COLS_ARRAY   = jnp.array(_table_cont_cols, dtype=jnp.int32)
_TABLE_SCALAR_CONT_MAX_VALUES   = jnp.array(_table_cont_maxs, dtype=jnp.float32)

(SOFTSYNTH_CAT_SPECS, SOFTSYNTH_CAT_COL_INDICES,
 _synth_cont_cols, _synth_cont_maxs) = _split(SOFTSYNTH_FIELD_SPECS)
SOFTSYNTH_CONT_N                = len(_synth_cont_cols)
_SOFTSYNTH_CONT_COLS_ARRAY      = jnp.array(_synth_cont_cols, dtype=jnp.int32)
_SOFTSYNTH_CONT_MAX_VALUES      = jnp.array(_synth_cont_maxs, dtype=jnp.float32)

# Grooves: all continuous (no cat fields).
GROOVE_CONT_N        = len(GROOVE_FIELD_SPECS)   # 32
_GROOVE_CONT_MAX = 16.0                       # all vocab=17 → max token = 16

# ---------------------------------------------------------------------------
# Hierarchical entity head specs.
#
# ENTITY_HEAD_SPECS: CE heads — only entity types that have discrete fields.
# _CONT_N:           continuous field counts — all entity head types.
# Outputs named as `{head}` (CE) and `{head}_cont` (regression).
# ---------------------------------------------------------------------------

ENTITY_HEAD_SPECS = {
    'instr_scalar':      INSTR_SCALAR_CAT_SPECS,
    'instr_table':       TABLE_SCALAR_CAT_SPECS,
    'instr_table_trace': TABLE_SCALAR_CAT_SPECS,
    'instr_softsynth':   SOFTSYNTH_CAT_SPECS,
    'table_scalar':      TABLE_SCALAR_CAT_SPECS,
    'table_trace':       TABLE_SCALAR_CAT_SPECS,
}

ENTITY_HEAD_TOTAL_VOCAB = {
    name: sum(v for _, v in specs)
    for name, specs in ENTITY_HEAD_SPECS.items()
}

# All entity head names that have a continuous (regression) decoder.
_CONT_N = {
    'instr_scalar':             INSTR_SCALAR_CONT_N,
    'instr_table':              TABLE_SCALAR_CONT_N,
    'instr_table_groove':       GROOVE_CONT_N,
    'instr_table_trace':        TABLE_SCALAR_CONT_N,
    'instr_table_trace_groove': GROOVE_CONT_N,
    'instr_softsynth':          SOFTSYNTH_CONT_N,
    'table_scalar':             TABLE_SCALAR_CONT_N,
    'table_groove':             GROOVE_CONT_N,
    'table_trace':              TABLE_SCALAR_CONT_N,
    'table_trace_groove':       GROOVE_CONT_N,
    'groove_id':                GROOVE_CONT_N,
}

# All entity output head names: CE heads + regression heads + waveframes.
ENTITY_OUTPUT_HEADS = (
    list(ENTITY_HEAD_SPECS.keys())
    + [f'{n}_cont' for n in _CONT_N]
    + ['instr_waveframes']
)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def hard_targets(tokens):
    return {
        name: jax.nn.one_hot(tokens[pos], vocab)
        for name, (pos, vocab) in TOKEN_HEADS.items()
    }


def token_loss(logits_dict, target_dists):
    total = 0.0
    for name in TOKEN_HEADS:
        log_probs = jax.nn.log_softmax(logits_dict[name])
        total += -jnp.sum(target_dists[name] * log_probs)
    return total


def _cat_ce(flat_logits, row, col_indices, cat_specs):
    """CE loss over discrete (categorical) fields. col_indices and cat_specs are Python lists."""
    offset = 0
    total = 0.0
    for k, (_, vocab) in enumerate(cat_specs):
        lp = jax.nn.log_softmax(flat_logits[offset:offset + vocab])
        total += -lp[row[col_indices[k]]]
        offset += vocab
    return total / len(cat_specs)


def _cont_mse(cont_logits, row, cont_cols_array, max_vals_array):
    """Vectorized MSE regression for continuous fields (single bank row)."""
    targets = row[cont_cols_array].astype(jnp.float32) / max_vals_array
    preds   = jax.nn.sigmoid(cont_logits)
    return jnp.mean((preds - targets) ** 2)


def _groove_mse_batch(cont_logits, groove_rows):
    """
    MSE regression for a batch of groove rows against the same logits.

    groove_rows: (N, 32) — N groove rows
    cont_logits: (32,)   — broadcast over all N rows
    """
    targets = groove_rows.astype(jnp.float32) / _GROOVE_CONT_MAX
    preds   = jax.nn.sigmoid(cont_logits)
    return jnp.mean((preds - targets) ** 2)


def entity_loss(entity_logits_dict, banks: SongBanks, target_tokens):
    """
    Hierarchical loss for one (step, channel).

    CE loss over discrete entity fields + MSE regression over continuous fields.
    Grooves are purely regression (no discrete fields). Trace-groove lookups
    use batched 3-D indexing — no nested vmap required.

    Returns scalar = sum of 12 components / 12.
    """
    target_tokens = jnp.int32(target_tokens)

    # ─── Instrument ──────────────────────────────────────────────────────────
    instr_id  = target_tokens[ENTITY_HEADS['instr_id']]
    instr_row = banks.instruments[instr_id]                      # (INSTR_WIDTH,)

    instr_scalar_cat = _cat_ce(
        entity_logits_dict['instr_scalar'], instr_row,
        INSTR_SCALAR_CAT_COL_INDICES, INSTR_SCALAR_CAT_SPECS,
    )
    instr_scalar_cont = _cont_mse(
        entity_logits_dict['instr_scalar_cont'], instr_row,
        _INSTR_SCALAR_CONT_COLS_ARRAY, _INSTR_SCALAR_CONT_MAX_VALUES,
    )

    # Instrument's table
    it_row = banks.tables[instr_row[_INSTR_TABLE_COL]]           # (624,)

    instr_table_cat = _cat_ce(
        entity_logits_dict['instr_table'], it_row,
        TABLE_SCALAR_CAT_COL_INDICES, TABLE_SCALAR_CAT_SPECS,
    )
    instr_table_cont = _cont_mse(
        entity_logits_dict['instr_table_cont'], it_row,
        _TABLE_SCALAR_CONT_COLS_ARRAY, _TABLE_SCALAR_CONT_MAX_VALUES,
    )

    # Instrument's table → grooves (32 refs, batch MSE — no vmap)
    it_groove_rows = banks.grooves[it_row[_GROOVE_FX_COLS_ARRAY]]  # (32, 32)
    instr_table_groove = _groove_mse_batch(
        entity_logits_dict['instr_table_groove_cont'], it_groove_rows,
    )

    # Instrument's table → traces (32 refs)
    it_trace_rows = banks.traces[it_row[_TABLE_FX_COLS_ARRAY]]     # (32, 624)

    def _it_trace_cat(trace_row):
        return _cat_ce(
            entity_logits_dict['instr_table_trace'], trace_row,
            TABLE_SCALAR_CAT_COL_INDICES, TABLE_SCALAR_CAT_SPECS,
        )
    instr_table_trace_cat = jax.vmap(_it_trace_cat)(it_trace_rows).mean()

    # Batch regression over 32 traces (no vmap)
    it_trace_cont_tgts = (
        it_trace_rows[:, _TABLE_SCALAR_CONT_COLS_ARRAY].astype(jnp.float32)
        / _TABLE_SCALAR_CONT_MAX_VALUES
    )
    instr_table_trace_cont = jnp.mean(
        (jax.nn.sigmoid(entity_logits_dict['instr_table_trace_cont'])
         - it_trace_cont_tgts) ** 2
    )

    # Instrument's table → traces → grooves (3-D gather, no vmap)
    it_tg_rows = banks.grooves[it_trace_rows[:, _GROOVE_FX_COLS_ARRAY]]  # (32, 32, 32)
    instr_table_trace_groove = jnp.mean(
        (jax.nn.sigmoid(entity_logits_dict['instr_table_trace_groove_cont'])
         - it_tg_rows.astype(jnp.float32) / _GROOVE_CONT_MAX) ** 2
    )

    # Instrument's softsynth
    synth_id   = instr_row[_INSTR_SOFTSYNTH_COL]
    synth_row  = banks.softsynths[synth_id]                      # (SOFTSYNTH_WIDTH,)

    instr_softsynth_cat = _cat_ce(
        entity_logits_dict['instr_softsynth'], synth_row,
        SOFTSYNTH_CAT_COL_INDICES, SOFTSYNTH_CAT_SPECS,
    )
    instr_softsynth_cont = _cont_mse(
        entity_logits_dict['instr_softsynth_cont'], synth_row,
        _SOFTSYNTH_CONT_COLS_ARRAY, _SOFTSYNTH_CONT_MAX_VALUES,
    )

    # Instrument's waveframes (regression, MSE)
    wf_row  = banks.waveframes[synth_id]                         # (WAVEFRAME_DIM,)
    wf_tgts = wf_row.astype(jnp.float32) / 15.0
    instr_waveframe_mse = jnp.mean(
        (jax.nn.sigmoid(entity_logits_dict['instr_waveframes']) - wf_tgts) ** 2
    )

    # ─── Phrase-level table ──────────────────────────────────────────────────
    pt_row = banks.tables[target_tokens[ENTITY_HEADS['table_id']]]  # (624,)

    table_scalar_cat = _cat_ce(
        entity_logits_dict['table_scalar'], pt_row,
        TABLE_SCALAR_CAT_COL_INDICES, TABLE_SCALAR_CAT_SPECS,
    )
    table_scalar_cont = _cont_mse(
        entity_logits_dict['table_scalar_cont'], pt_row,
        _TABLE_SCALAR_CONT_COLS_ARRAY, _TABLE_SCALAR_CONT_MAX_VALUES,
    )

    pt_groove_rows = banks.grooves[pt_row[_GROOVE_FX_COLS_ARRAY]]   # (32, 32)
    table_groove = _groove_mse_batch(
        entity_logits_dict['table_groove_cont'], pt_groove_rows,
    )

    pt_trace_rows = banks.traces[pt_row[_TABLE_FX_COLS_ARRAY]]      # (32, 624)

    def _pt_trace_cat(trace_row):
        return _cat_ce(
            entity_logits_dict['table_trace'], trace_row,
            TABLE_SCALAR_CAT_COL_INDICES, TABLE_SCALAR_CAT_SPECS,
        )
    table_trace_cat = jax.vmap(_pt_trace_cat)(pt_trace_rows).mean()

    pt_trace_cont_tgts = (
        pt_trace_rows[:, _TABLE_SCALAR_CONT_COLS_ARRAY].astype(jnp.float32)
        / _TABLE_SCALAR_CONT_MAX_VALUES
    )
    table_trace_cont = jnp.mean(
        (jax.nn.sigmoid(entity_logits_dict['table_trace_cont'])
         - pt_trace_cont_tgts) ** 2
    )

    pt_tg_rows = banks.grooves[pt_trace_rows[:, _GROOVE_FX_COLS_ARRAY]]  # (32, 32, 32)
    table_trace_groove = jnp.mean(
        (jax.nn.sigmoid(entity_logits_dict['table_trace_groove_cont'])
         - pt_tg_rows.astype(jnp.float32) / _GROOVE_CONT_MAX) ** 2
    )

    # ─── Phrase-level groove ─────────────────────────────────────────────────
    groove_row = banks.grooves[target_tokens[ENTITY_HEADS['groove_id']]]  # (32,)
    groove_id_cont = jnp.mean(
        (jax.nn.sigmoid(entity_logits_dict['groove_id_cont'])
         - groove_row.astype(jnp.float32) / _GROOVE_CONT_MAX) ** 2
    )

    return (
        instr_scalar_cat
        + instr_scalar_cont
        + instr_table_cat
        + instr_table_cont
        + instr_table_groove
        + instr_table_trace_cat
        + instr_table_trace_cont
        + instr_table_trace_groove
        + instr_softsynth_cat
        + instr_softsynth_cont
        + instr_waveframe_mse
        + table_scalar_cat
        + table_scalar_cont
        + table_groove
        + table_trace_cat
        + table_trace_cont
        + table_trace_groove
        + groove_id_cont
    ) / 18


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class EntityDecoder(eqx.Module):
    """Two-layer MLP: d_model → entity_dim (GELU) → output_dim."""
    linear_in:  eqx.nn.Linear
    linear_out: eqx.nn.Linear

    def __init__(self, d_model, entity_dim, output_dim, key):
        k1, k2 = jr.split(key)
        self.linear_in  = eqx.nn.Linear(d_model, entity_dim,   use_bias=False, key=k1)
        self.linear_out = eqx.nn.Linear(entity_dim, output_dim, use_bias=False, key=k2)

    def __call__(self, x):
        return self.linear_out(jax.nn.gelu(self.linear_in(x)))

    def encode(self, x):
        """d_model → entity_dim latent (for cosine-similarity matching)."""
        return self.linear_in(x)


class OutputHeads(eqx.Module):
    """
    Output projection heads.

    weights:          logit-group linear heads (shared-vocab batching)
    cat_decoders:     CE entity heads (discrete fields only)
    cont_decoders:    regression heads (continuous fields, pre-sigmoid)
    waveframe_decoder: 512-dim regression head for softsynth waveframes
    """
    weights:           dict[str, Array]
    cat_decoders:      dict[str, EntityDecoder]
    cont_decoders:     dict[str, EntityDecoder]
    waveframe_decoder: EntityDecoder

    def __init__(self, d_model, entity_dim, key):
        n_keys = len(LOGIT_GROUPS) + len(ENTITY_HEAD_SPECS) + len(_CONT_N) + 1
        keys   = jr.split(key, n_keys)
        ki     = 0

        # Logit-group heads
        weights = {}
        for group_name, members in LOGIT_GROUPS.items():
            n     = len(members)
            vocab = members[0][2]
            weights[group_name] = jr.normal(keys[ki], (n, vocab, d_model)) / jnp.sqrt(d_model)
            ki += 1
        self.weights = weights

        # CE decoders (discrete fields)
        cat_decoders = {}
        for name, total_vocab in ENTITY_HEAD_TOTAL_VOCAB.items():
            cat_decoders[name] = EntityDecoder(d_model, entity_dim, total_vocab, keys[ki])
            ki += 1
        self.cat_decoders = cat_decoders

        # Regression decoders (continuous fields)
        cont_decoders = {}
        for name, n_cont in _CONT_N.items():
            cont_decoders[name] = EntityDecoder(d_model, entity_dim, n_cont, keys[ki])
            ki += 1
        self.cont_decoders = cont_decoders

        self.waveframe_decoder = EntityDecoder(d_model, entity_dim, WAVEFRAME_DIM, keys[ki])

    def __call__(self, x):
        result = {}
        for group_name, members in LOGIT_GROUPS.items():
            logits = self.weights[group_name] @ x
            for i, (head_name, _, _) in enumerate(members):
                result[head_name] = logits[i]
        for name in ENTITY_HEAD_SPECS:
            result[name] = self.cat_decoders[name](x)
        for name in _CONT_N:
            result[f'{name}_cont'] = self.cont_decoders[name](x)
        result['instr_waveframes'] = self.waveframe_decoder(x)
        return result

    def log_probs(self, x):
        raw = self(x)
        return {name: jax.nn.log_softmax(raw[name]) for name in TOKEN_HEADS}


def _norm2d(norm, x):
    return jax.vmap(jax.vmap(norm))(x)


class AxialTransformerBlock(eqx.Module):
    temporal_attn: eqx.nn.MultiheadAttention
    channel_attn:  eqx.nn.MultiheadAttention
    mlp:           eqx.nn.MLP
    norm_t:        eqx.nn.LayerNorm
    norm_c:        eqx.nn.LayerNorm
    norm_mlp:      eqx.nn.LayerNorm

    def __init__(self, d_model, num_heads_t, num_heads_c, key):
        k1, k2, k3 = jr.split(key, 3)
        self.temporal_attn = eqx.nn.MultiheadAttention(num_heads_t, d_model, key=k1)
        self.channel_attn  = eqx.nn.MultiheadAttention(num_heads_c, d_model, key=k2)
        self.mlp      = eqx.nn.MLP(d_model, d_model, d_model * 4, depth=1, key=k3)
        self.norm_t   = eqx.nn.LayerNorm(d_model)
        self.norm_c   = eqx.nn.LayerNorm(d_model)
        self.norm_mlp = eqx.nn.LayerNorm(d_model)

    def __call__(self, x: Array, causal_mask: Bool[Array, "S S"]) -> Array:
        normed = _norm2d(self.norm_c, x)
        x = x + jax.vmap(lambda x_t: self.channel_attn(x_t, x_t, x_t))(normed)
        normed = _norm2d(self.norm_t, x)
        x = x + jax.vmap(
            lambda x_ch: self.temporal_attn(x_ch, x_ch, x_ch, mask=causal_mask),
            in_axes=1, out_axes=1,
        )(normed)
        normed = _norm2d(self.norm_mlp, x)
        x = x + jax.vmap(jax.vmap(self.mlp))(normed)
        return x


class LSDJTransformer(eqx.Module):
    embedder:     SequenceEmbedder
    blocks:       list[AxialTransformerBlock]
    final_norm:   eqx.nn.LayerNorm
    output_heads: OutputHeads
    d_model:      int
    metadata:     dict

    def __init__(
        self,
        key: Key,
        *,
        d_model: int = 256,
        entity_dim: int = 128,
        num_heads_t: int = 4,
        num_heads_c: int = 2,
        num_blocks: int = 6,
        banks: SongBanks | None = None,
        **embedder_kwargs,
    ):
        self.metadata = {
            "d_model": d_model, "entity_dim": entity_dim,
            "num_heads_t": num_heads_t, "num_heads_c": num_heads_c,
            "num_blocks": num_blocks,
            "embedder": {k: v for k, v in embedder_kwargs.items() if isinstance(v, int)},
        }
        keys = jr.split(key, num_blocks + 3)
        self.d_model = d_model
        self.embedder = SequenceEmbedder.create(
            keys[0], banks=banks, out_dim=d_model * 4, **embedder_kwargs,
        )
        self.blocks = [
            AxialTransformerBlock(d_model, num_heads_t, num_heads_c, keys[i + 1])
            for i in range(num_blocks)
        ]
        self.final_norm   = eqx.nn.LayerNorm(d_model)
        self.output_heads = OutputHeads(d_model, entity_dim, keys[-1])

    def encode(self, song_tokens: Array) -> Array:
        x = self.embedder(song_tokens)
        S = x.shape[0]
        causal_mask = jnp.tril(jnp.ones((S, S), dtype=bool))
        for block in self.blocks:
            x = block(x, causal_mask)
        return _norm2d(self.final_norm, x)

    def __call__(self, song_tokens: Array):
        return jax.vmap(jax.vmap(self.output_heads))(self.encode(song_tokens))

    def with_banks(self, banks: SongBanks):
        new_embedder = self.embedder.with_banks(banks)
        return eqx.tree_at(lambda m: m.embedder, self, new_embedder)

    def write_metadata(self, filepath):
        with open(filepath, "w") as f:
            f.write(json.dumps(self.metadata))
