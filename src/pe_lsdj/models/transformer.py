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
GROOVE_CONT_N    = len(GROOVE_FIELD_SPECS)   # 32
_GROOVE_CONT_MAX = 16.0                      # all vocab=17 → max token = 16

# ---------------------------------------------------------------------------
# Cat CE groups: fields grouped by vocab size for vectorized loss.
#
# Instead of a Python loop over N fields, we loop over D distinct vocab sizes
# (D ≤ 3 for all entity types) and do a batched (n_fields, vocab) operation
# per group. Precomputed at module load time; used as static data in JIT.
# ---------------------------------------------------------------------------

def _build_cat_groups(cat_specs, col_indices):
    """Return [(vocab, starts_array, cols_array)] sorted by vocab, one entry per distinct vocab."""
    offset   = 0
    by_vocab = {}
    for k, (_, vocab) in enumerate(cat_specs):
        if vocab not in by_vocab:
            by_vocab[vocab] = ([], [])
        by_vocab[vocab][0].append(offset)
        by_vocab[vocab][1].append(col_indices[k])
        offset += vocab
    return [
        (v, jnp.array(starts, dtype=jnp.int32), jnp.array(cols, dtype=jnp.int32))
        for v, (starts, cols) in sorted(by_vocab.items())
    ]

_INSTR_SCALAR_CAT_GROUPS = _build_cat_groups(INSTR_SCALAR_CAT_SPECS, INSTR_SCALAR_CAT_COL_INDICES)
_TABLE_SCALAR_CAT_GROUPS = _build_cat_groups(TABLE_SCALAR_CAT_SPECS, TABLE_SCALAR_CAT_COL_INDICES)
_SOFTSYNTH_CAT_GROUPS    = _build_cat_groups(SOFTSYNTH_CAT_SPECS,    SOFTSYNTH_CAT_COL_INDICES)

# ---------------------------------------------------------------------------
# Vocab totals for decoder output sizes.
# ---------------------------------------------------------------------------

TABLE_SCALAR_CAT_TOTAL_VOCAB = sum(v for _, v in TABLE_SCALAR_CAT_SPECS)
INSTR_SCALAR_CAT_TOTAL_VOCAB = sum(v for _, v in INSTR_SCALAR_CAT_SPECS)
SOFTSYNTH_CAT_TOTAL_VOCAB    = sum(v for _, v in SOFTSYNTH_CAT_SPECS)

N_TABLE_SLOTS  = len(TABLE_FX_COL_INDICES)   # 32 — TABLE_FX slots per table
N_GROOVE_SLOTS = len(GROOVE_FX_COL_INDICES)  # 32 — GROOVE_FX slots per table

# Mask for trace cat_out: A (CMD_A=1) and H (CMD_H=7) commands are invalid in traces.
# Applied to all 19-vocab fx_cmd fields in TABLE_SCALAR_CAT logits.
_TABLE_CAT_TRACE_MASK = jnp.zeros(TABLE_SCALAR_CAT_TOTAL_VOCAB, dtype=bool)
for _vocab, _starts, _cols in _TABLE_SCALAR_CAT_GROUPS:
    if _vocab == 19:
        for _cmd in (CMD_A, CMD_H):
            _TABLE_CAT_TRACE_MASK = _TABLE_CAT_TRACE_MASK.at[_starts + _cmd].set(True)
del _vocab, _starts, _cols, _cmd


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


def _ce_loss_grouped(flat_logits, row, groups):
    """
    Vectorized CE over categorical fields, grouped by vocab size.

    groups: [(vocab, starts_array, cols_array)] — precomputed by _build_cat_groups.
    Python loop has ≤3 iterations (one per distinct vocab); inner ops are batched.
    """
    total   = 0.0
    n_total = 0
    for vocab, starts, cols in groups:
        indices      = starts[:, None] + jnp.arange(vocab)[None, :]  # (n, vocab)
        logits_block = flat_logits[indices]                           # (n, vocab)
        lp           = jax.nn.log_softmax(logits_block, axis=-1)
        targets      = row[cols]                                      # (n,)
        total       += -jnp.sum(lp[jnp.arange(cols.shape[0]), targets])
        n_total     += cols.shape[0]
    return total / n_total


def _mse_loss(cont_logits, row, cont_cols_array, max_vals_array):
    """Vectorized MSE regression for continuous fields (single bank row)."""
    targets = row[cont_cols_array].astype(jnp.float32) / max_vals_array
    preds   = jax.nn.sigmoid(cont_logits)
    return jnp.mean((preds - targets) ** 2)


def _table_loss(preds, table_row, banks):
    """
    Loss components for one table (or trace) prediction.
    Returns a Python list of scalar losses — 6 components:
      scalar cat, scalar cont, groove slots, trace cat, trace cont, trace groove slots.
    Called for both instrument's table and phrase-level table.
    """
    losses = []

    # Table scalar fields
    losses.append(_ce_loss_grouped(preds['cat'], table_row, _TABLE_SCALAR_CAT_GROUPS))
    losses.append(_mse_loss(preds['cont'], table_row,
                            _TABLE_SCALAR_CONT_COLS_ARRAY, _TABLE_SCALAR_CONT_MAX_VALUES))

    # Per-slot groove predictions: preds['grooves'] is (N_GROOVE_SLOTS, GROOVE_CONT_N)
    groove_ids   = table_row[_GROOVE_FX_COLS_ARRAY]          # (N_GROOVE_SLOTS,)
    groove_rows  = banks.grooves[groove_ids]                  # (N_GROOVE_SLOTS, GROOVE_CONT_N)
    groove_tgts  = groove_rows.astype(jnp.float32) / _GROOVE_CONT_MAX
    losses.append(jnp.mean((jax.nn.sigmoid(preds['grooves']) - groove_tgts) ** 2))

    # Per-slot trace predictions: preds['traces'] is a dict of batched arrays
    trace_ids  = table_row[_TABLE_FX_COLS_ARRAY]             # (N_TABLE_SLOTS,)
    trace_rows = banks.traces[trace_ids]                     # (N_TABLE_SLOTS, TABLE_WIDTH)
    tp = preds['traces']

    # Trace cat — vmap over N_TABLE_SLOTS
    def _trace_cat(p, row):
        return _ce_loss_grouped(p, row, _TABLE_SCALAR_CAT_GROUPS)
    losses.append(jax.vmap(_trace_cat)(tp['cat'], trace_rows).mean())

    # Trace cont — batched MSE
    trace_cont_tgts = (
        trace_rows[:, _TABLE_SCALAR_CONT_COLS_ARRAY].astype(jnp.float32)
        / _TABLE_SCALAR_CONT_MAX_VALUES
    )
    losses.append(jnp.mean((jax.nn.sigmoid(tp['cont']) - trace_cont_tgts) ** 2))

    # Trace groove slots — 3D gather, no vmap needed
    trace_groove_ids  = trace_rows[:, _GROOVE_FX_COLS_ARRAY]   # (N_TABLE_SLOTS, N_GROOVE_SLOTS)
    trace_groove_rows = banks.grooves[trace_groove_ids]         # (N_TABLE_SLOTS, N_GROOVE_SLOTS, GROOVE_CONT_N)
    trace_groove_tgts = trace_groove_rows.astype(jnp.float32) / _GROOVE_CONT_MAX
    losses.append(jnp.mean((jax.nn.sigmoid(tp['grooves']) - trace_groove_tgts) ** 2))

    return losses


def entity_loss(entity_preds, banks: SongBanks, target_tokens):
    """
    Hierarchical entity loss for one (step, channel).

    entity_preds: nested dict from OutputHeads.__call__() — keys 'instr', 'table', 'groove'.
    Returns scalar = mean of 18 loss components.
    """
    target_tokens = jnp.int32(target_tokens)
    losses = []

    # ─── Instrument ──────────────────────────────────────────────────────────
    instr_id  = target_tokens[ENTITY_HEADS['instr_id']]
    instr_row = banks.instruments[instr_id]
    p = entity_preds['instr']

    losses.append(_ce_loss_grouped(p['cat'], instr_row, _INSTR_SCALAR_CAT_GROUPS))
    losses.append(_mse_loss(p['cont'], instr_row,
                            _INSTR_SCALAR_CONT_COLS_ARRAY, _INSTR_SCALAR_CONT_MAX_VALUES))

    # Instrument's table
    it_row = banks.tables[instr_row[_INSTR_TABLE_COL]]
    losses.extend(_table_loss(p['table'], it_row, banks))

    # Instrument's softsynth
    synth_id  = instr_row[_INSTR_SOFTSYNTH_COL]
    synth_row = banks.softsynths[synth_id]
    ps = p['softsynth']
    losses.append(_ce_loss_grouped(ps['cat'], synth_row, _SOFTSYNTH_CAT_GROUPS))
    losses.append(_mse_loss(ps['cont'], synth_row,
                            _SOFTSYNTH_CONT_COLS_ARRAY, _SOFTSYNTH_CONT_MAX_VALUES))

    # Waveframes
    wf_row = banks.waveframes[synth_id]
    losses.append(jnp.mean(
        (jax.nn.sigmoid(ps['waveframes']) - wf_row.astype(jnp.float32) / 15.0) ** 2
    ))

    # ─── Phrase-level table ──────────────────────────────────────────────────
    pt_row = banks.tables[target_tokens[ENTITY_HEADS['table_id']]]
    losses.extend(_table_loss(entity_preds['table'], pt_row, banks))

    # ─── Phrase-level groove ─────────────────────────────────────────────────
    groove_row = banks.grooves[target_tokens[ENTITY_HEADS['groove_id']]]
    losses.append(jnp.mean(
        (jax.nn.sigmoid(entity_preds['groove'])
         - groove_row.astype(jnp.float32) / _GROOVE_CONT_MAX) ** 2
    ))

    return jnp.mean(jnp.array(losses))


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


class GrooveDecoder(eqx.Module):
    """
    Predicts one groove (GROOVE_CONT_N fields) per slot.

    Slot embeddings (shape: N_GROOVE_SLOTS × entity_dim) are added to the
    entity_dim context before decoding. Shared across all groove slot levels
    (table and trace) — context distinguishes the level, slot_embeds
    distinguish position within the level.
    """
    slot_embeds: Array          # (N_GROOVE_SLOTS, entity_dim)
    linear_in:   eqx.nn.Linear  # entity_dim → entity_dim
    linear_out:  eqx.nn.Linear  # entity_dim → GROOVE_CONT_N

    def __init__(self, entity_dim, key):
        k1, k2, k3 = jr.split(key, 3)
        self.slot_embeds = jr.normal(k1, (N_GROOVE_SLOTS, entity_dim)) * 0.02
        self.linear_in   = eqx.nn.Linear(entity_dim, entity_dim,   use_bias=False, key=k2)
        self.linear_out  = eqx.nn.Linear(entity_dim, GROOVE_CONT_N, use_bias=False, key=k3)

    def __call__(self, context, slot_idx):
        """context: (entity_dim,) → (GROOVE_CONT_N,) groove logits."""
        h = jax.nn.gelu(self.linear_in(context + self.slot_embeds[slot_idx]))
        return self.linear_out(h)

    def all_slots(self, context):
        """Predict all N_GROOVE_SLOTS grooves: (N_GROOVE_SLOTS, GROOVE_CONT_N)."""
        # context: (entity_dim,); slot_embeds: (N, entity_dim) — broadcasts
        h = jax.nn.gelu(jax.vmap(self.linear_in)(context + self.slot_embeds))
        return jax.vmap(self.linear_out)(h)

    def encode(self, context, slot_idx):
        """entity_dim latent for cosine-similarity bank matching."""
        return self.linear_in(context + self.slot_embeds[slot_idx])


class TableDecoder(eqx.Module):
    """
    Unified table/trace decoder.

    Tables and traces have identical structure and share all weights.
    The only structural difference: traces mask CMD_A (TABLE) and CMD_H (HOP)
    command logits to −∞, preventing further table chaining.

    Slot embeddings distinguish which of the N_TABLE_SLOTS trace sub-slots is
    being predicted. Added to the entity_dim context before sub-decoding.

    Shared GrooveDecoder instance handles GROOVE_FX slot predictions at all depths.
    sub_table_decoder=None at trace level (explicit depth cap).
    """
    is_trace:          bool = eqx.field(static=True)
    slot_embeds:       Array            # (N_TABLE_SLOTS, entity_dim) — for trace sub-slots
    linear_in:         eqx.nn.Linear   # entity_dim → entity_dim
    cat_out:           eqx.nn.Linear   # entity_dim → TABLE_SCALAR_CAT_TOTAL_VOCAB
    cont_out:          eqx.nn.Linear   # entity_dim → TABLE_SCALAR_CONT_N
    groove_decoder:    GrooveDecoder
    sub_table_decoder: 'TableDecoder | None'

    def __init__(self, entity_dim, is_trace, groove_decoder, key, *,
                 sub_table_decoder=None,
                 _slot_embeds=None, _linear_in=None, _cat_out=None, _cont_out=None):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.is_trace = is_trace
        self.slot_embeds = (
            _slot_embeds if _slot_embeds is not None
            else jr.normal(k1, (N_TABLE_SLOTS, entity_dim)) * 0.02
        )
        self.linear_in = (
            _linear_in if _linear_in is not None
            else eqx.nn.Linear(entity_dim, entity_dim, use_bias=False, key=k2)
        )
        self.cat_out = (
            _cat_out if _cat_out is not None
            else eqx.nn.Linear(entity_dim, TABLE_SCALAR_CAT_TOTAL_VOCAB, use_bias=False, key=k3)
        )
        self.cont_out = (
            _cont_out if _cont_out is not None
            else eqx.nn.Linear(entity_dim, TABLE_SCALAR_CONT_N, use_bias=False, key=k4)
        )
        self.groove_decoder    = groove_decoder
        self.sub_table_decoder = sub_table_decoder

    def __call__(self, context):
        """
        context: (entity_dim,) → nested dict:
          {
            'cat':    (TABLE_SCALAR_CAT_TOTAL_VOCAB,)  [A/H masked if is_trace]
            'cont':   (TABLE_SCALAR_CONT_N,)
            'grooves':(N_GROOVE_SLOTS, GROOVE_CONT_N)
            'traces': {                                 [only if sub_table_decoder set]
              'cat':    (N_TABLE_SLOTS, TABLE_SCALAR_CAT_TOTAL_VOCAB)
              'cont':   (N_TABLE_SLOTS, TABLE_SCALAR_CONT_N)
              'grooves':(N_TABLE_SLOTS, N_GROOVE_SLOTS, GROOVE_CONT_N)
            }
          }
        """
        h = jax.nn.gelu(self.linear_in(context))

        cat_logits = self.cat_out(h)
        if self.is_trace:
            cat_logits = jnp.where(_TABLE_CAT_TRACE_MASK, -jnp.inf, cat_logits)

        preds = {
            'cat':    cat_logits,
            'cont':   self.cont_out(h),
            'grooves': self.groove_decoder.all_slots(h),
        }

        if self.sub_table_decoder is not None:
            # Each trace slot: h + slot_embed[i] as context
            trace_contexts = h + self.slot_embeds   # (N_TABLE_SLOTS, entity_dim)
            preds['traces'] = jax.vmap(self.sub_table_decoder)(trace_contexts)

        return preds

    def encode(self, context):
        """entity_dim latent for cosine-similarity bank matching."""
        return self.linear_in(context)


class SoftSynthDecoder(eqx.Module):
    """
    Softsynth + waveframe decoder.
    Conditions on instrument entity_dim latent (subordinate — not backbone x directly).
    """
    linear_in:     eqx.nn.Linear   # entity_dim → entity_dim
    cat_out:       eqx.nn.Linear   # entity_dim → SOFTSYNTH_CAT_TOTAL_VOCAB
    cont_out:      eqx.nn.Linear   # entity_dim → SOFTSYNTH_CONT_N
    waveframe_out: eqx.nn.Linear   # entity_dim → WAVEFRAME_DIM

    def __init__(self, entity_dim, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.linear_in     = eqx.nn.Linear(entity_dim, entity_dim,           use_bias=False, key=k1)
        self.cat_out       = eqx.nn.Linear(entity_dim, SOFTSYNTH_CAT_TOTAL_VOCAB, use_bias=False, key=k2)
        self.cont_out      = eqx.nn.Linear(entity_dim, SOFTSYNTH_CONT_N,     use_bias=False, key=k3)
        self.waveframe_out = eqx.nn.Linear(entity_dim, WAVEFRAME_DIM,        use_bias=False, key=k4)

    def __call__(self, instr_h):
        """instr_h: (entity_dim,) GELU'd instrument latent."""
        h = jax.nn.gelu(self.linear_in(instr_h))
        return {
            'cat':        self.cat_out(h),
            'cont':       self.cont_out(h),
            'waveframes': self.waveframe_out(h),
        }

    def encode(self, instr_h):
        return self.linear_in(instr_h)


class InstrumentDecoder(eqx.Module):
    """
    Phrase-level instrument decoder. Conditions on backbone x (d_model).
    Instrument's table uses the shared table_decoder with instrument latent as context.
    Softsynth/waveframes condition on instrument entity_dim latent (subordinate).
    """
    linear_in:         eqx.nn.Linear    # d_model → entity_dim
    cat_out:           eqx.nn.Linear    # entity_dim → INSTR_SCALAR_CAT_TOTAL_VOCAB
    cont_out:          eqx.nn.Linear    # entity_dim → INSTR_SCALAR_CONT_N
    softsynth_decoder: SoftSynthDecoder

    def __init__(self, d_model, entity_dim, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.linear_in         = eqx.nn.Linear(d_model,     entity_dim,              use_bias=False, key=k1)
        self.cat_out           = eqx.nn.Linear(entity_dim,  INSTR_SCALAR_CAT_TOTAL_VOCAB, use_bias=False, key=k2)
        self.cont_out          = eqx.nn.Linear(entity_dim,  INSTR_SCALAR_CONT_N,     use_bias=False, key=k3)
        self.softsynth_decoder = SoftSynthDecoder(entity_dim, k4)

    def __call__(self, x):
        """
        x: (d_model,) backbone repr.
        Returns (preds_dict, h) where h is the GELU'd entity_dim latent.
        preds_dict has keys 'cat', 'cont', 'softsynth'.
        Instrument's 'table' is added by OutputHeads using the shared table_decoder.
        """
        h = jax.nn.gelu(self.linear_in(x))
        return {
            'cat':       self.cat_out(h),
            'cont':      self.cont_out(h),
            'softsynth': self.softsynth_decoder(h),
        }, h

    def encode(self, x):
        """entity_dim latent for cosine-similarity bank matching."""
        return self.linear_in(x)


class OutputHeads(eqx.Module):
    """
    All output projection heads.

    weights:           logit-group linear heads (TOKEN_HEADS, shared-vocab batching)
    groove_decoder:    shared GrooveDecoder — used at ALL depths (table and trace groove slots)
    table_decoder:     shared TableDecoder — phrase-level table AND instrument's table
                       (table_decoder.sub_table_decoder is the trace decoder, sharing weights)
    instr_decoder:     InstrumentDecoder (instrument scalars + softsynth)
    table_proj:        d_model → entity_dim projection for phrase-level table context
    phrase_groove_dec: simple EntityDecoder for phrase-level groove (no slot conditioning)
    """
    weights:           dict[str, Array]
    groove_decoder:    GrooveDecoder
    table_decoder:     TableDecoder
    instr_decoder:     InstrumentDecoder
    table_proj:        eqx.nn.Linear
    phrase_groove_dec: EntityDecoder

    def __init__(self, d_model, entity_dim, key):
        keys = jr.split(key, 7)

        # Logit-group heads (unchanged)
        weights = {}
        for group_name, members in LOGIT_GROUPS.items():
            n     = len(members)
            vocab = members[0][2]
            weights[group_name] = jr.normal(keys[0], (n, vocab, d_model)) / jnp.sqrt(d_model)
        self.weights = weights

        # Shared GrooveDecoder
        groove_dec = GrooveDecoder(entity_dim, keys[1])
        self.groove_decoder = groove_dec

        # Trace decoder (is_trace=True, depth cap — no sub_table_decoder)
        trace_dec = TableDecoder(entity_dim, is_trace=True,
                                 groove_decoder=groove_dec, key=keys[2])

        # Table decoder — shares all weights with trace_dec except is_trace flag
        table_dec = TableDecoder(entity_dim, is_trace=False,
                                 groove_decoder=groove_dec, key=keys[3],
                                 sub_table_decoder=trace_dec,
                                 _slot_embeds=trace_dec.slot_embeds,
                                 _linear_in=trace_dec.linear_in,
                                 _cat_out=trace_dec.cat_out,
                                 _cont_out=trace_dec.cont_out)
        self.table_decoder = table_dec

        # Instrument decoder
        self.instr_decoder = InstrumentDecoder(d_model, entity_dim, keys[4])

        # Phrase-level projections
        self.table_proj        = eqx.nn.Linear(d_model, entity_dim, use_bias=False, key=keys[5])
        self.phrase_groove_dec = EntityDecoder(d_model, entity_dim, GROOVE_CONT_N, keys[6])

    def __call__(self, x):
        """x: (d_model,) → nested output dict."""
        result = {}

        # Token heads
        for group_name, members in LOGIT_GROUPS.items():
            logits = self.weights[group_name] @ x
            for i, (name, _, _) in enumerate(members):
                result[name] = logits[i]

        # Phrase-level table
        table_ctx = jax.nn.gelu(self.table_proj(x))
        result['table'] = self.table_decoder(table_ctx)

        # Phrase-level groove
        result['groove'] = self.phrase_groove_dec(x)

        # Instrument
        instr_preds, instr_h = self.instr_decoder(x)
        instr_preds['table'] = self.table_decoder(instr_h)  # shared weights, instr latent as context
        result['instr'] = instr_preds

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
