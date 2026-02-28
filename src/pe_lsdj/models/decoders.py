import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from pe_lsdj.constants import *


# ---------------------------------------------------------------------------
# Logit groups: direct softmax heads, same-vocab members batched.
#    { group: [ (name, token position, vocab size), ...] }
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
# Entity parameter field specs: (name, vocab, is_continuous).
#
# is_continuous mirrors the embedder:
#   True  → GatedNormedEmbedder (byte/nibble ordinal; use MSE regression)
#   False → EnumEmbedder or EntityEmbedder (discrete category; use CE)
# ---------------------------------------------------------------------------

INSTR_FIELD_SPECS = [
    (TYPE_ID,            5, False),  # enum: PU/WAV/KIT/NOI
    (TABLE,             33, False),  # entity reference  ← index 1
    (TABLE_ON_OFF,       2, False),  # bool
    (TABLE_AUTOMATE,     2, False),  # bool
    (AUTOMATE_2,         2, False),  # bool
    (PAN,                5, False),  # enum: Off/L/R/LR
    (VIBRATO_TYPE,       5, False),  # enum: HF/saw/sine/square  (PU/WAV/KIT)
    (VIBRATO_DIRECTION,  3, False),  # enum: down/up             (PU/WAV/KIT)
    (ENV_VOLUME,        17, True),   # nibble: envelope volume       (PU/NOI)
    (ENV_FADE,          17, True),   # nibble: fade duration         (PU/NOI)
    (LENGTH,            65, True),   # 6-bit: note duration          (PU/NOI)
    (LENGTH_LIMITED,     3, False),  # bool                          (PU/NOI)
    (SWEEP,            257, True),   # byte: sweep intensity         (PU/NOI)
    (VOLUME,             5, False),  # enum: discrete output level  (WAV/KIT)
    (PHASE_TRANSPOSE,  257, True),   # byte: pitch offset                (PU)
    (WAVE,               5, False),  # enum: duty cycle                  (PU)
    (PHASE_FINETUNE,    17, True),   # nibble: finetune                  (PU)
    (SOFTSYNTH_ID,      17, False),  # entity reference  ← index 17.    (WAV)
    (REPEAT,            17, True),   # nibble: repeat count             (WAV)
    (PLAY_TYPE,          5, False),  # enum: once/loop/pingpong/manual  (WAV)
    (WAVE_LENGTH,       17, True),   # nibble: waveform length          (WAV)
    (SPEED,             17, True),   # nibble: playback speed           (WAV)
    (KEEP_ATTACK_1,      3, False),  # bool                             (KIT)
    (KEEP_ATTACK_2,      3, False),  # bool                             (KIT)
    (KIT_1_ID,          65, True),   # 6-bit: sample index (ordinal)    (KIT)
    (KIT_2_ID,          65, True),   # 6-bit: sample index              (KIT)
    (LENGTH_KIT_1,     257, True),   # byte: sample length              (KIT)
    (LENGTH_KIT_2,     257, True),   # byte                             (KIT)
    (LOOP_KIT_1,         3, False),  # bool                             (KIT)
    (LOOP_KIT_2,         3, False),  # bool                             (KIT)
    (OFFSET_KIT_1,     257, True),   # byte: playback offset            (KIT)
    (OFFSET_KIT_2,     257, True),   # byte                             (KIT)
    (HALF_SPEED,         3, False),  # bool                             (KIT)
    (PITCH,            257, True),   # byte: pitch                      (KIT)
    (DISTORTION_TYPE,    5, False),  # enum: clip/shape/shape2/wrap     (KIT)
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

GROOVE_FIELD_SPECS = [
    (f'step{i}_{tick}', 17, True)
    for i in range(STEPS_PER_GROOVE) for tick in ('even', 'odd')
]

# Only present if instr_type == WAV, so add a +1 offset to vocab
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
# Categorical / continuous split.
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
# Categorical CE groups: fields grouped by vocab size for vectorized loss.
#
# Loop over D distinct vocab sizes (D ≤ 3 for all entity types) and do a 
# batched (n_fields, vocab) operation per group.
# Precomputed at module load time; used as static data in JIT.
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
# Decoder classes
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

    Slot embeddings (shape: (N_GROOVE_SLOTS + 1) × entity_dim) are added to the
    entity_dim context before decoding. Shared across ALL groove levels:
      indices 0..N_GROOVE_SLOTS-1 — per-table/trace slot predictions
      index  N_GROOVE_SLOTS       — phrase-level groove (special phrase slot)
    Context distinguishes the level; slot_embeds distinguish position.
    """
    slot_embeds: Array          # (N_GROOVE_SLOTS + 1, entity_dim)
    linear_in:   eqx.nn.Linear  # entity_dim → entity_dim
    linear_out:  eqx.nn.Linear  # entity_dim → GROOVE_CONT_N

    def __init__(self, entity_dim, key):
        k1, k2, k3 = jr.split(key, 3)
        self.slot_embeds = jr.normal(k1, (N_GROOVE_SLOTS + 1, entity_dim)) * 0.02
        self.linear_in   = eqx.nn.Linear(entity_dim, entity_dim,   use_bias=False, key=k2)
        self.linear_out  = eqx.nn.Linear(entity_dim, GROOVE_CONT_N, use_bias=False, key=k3)

    def __call__(self, context, slot_idx):
        """context: (entity_dim,) → (GROOVE_CONT_N,) groove logits."""
        h = jax.nn.gelu(self.linear_in(context + self.slot_embeds[slot_idx]))
        return self.linear_out(h)

    def encode(self, context, slot_idx):
        """entity_dim latent for cosine-similarity bank matching."""
        return self.linear_in(context + self.slot_embeds[slot_idx])


class TableDecoder(eqx.Module):
    """
    Unified table/trace decoder.

    Tables and traces share all parameters (slot_embeds, linear_in, cat_out, cont_out).
    Forward call produces only scalar predictions {'cat', 'cont'}.

    Groove and trace sub-entity losses are computed externally in cond_entity_scan_loss
    using these weights with lax.scan + lax.cond for genuine conditionality, avoiding
    the O(L × 4 × N_TABLE_SLOTS × N_GROOVE_SLOTS × entity_dim) memory cost of
    unconditional materialization.

    slot_embeds are used by cond_entity_scan_loss to contextualize each trace sub-slot.
    """
    slot_embeds: Array          # (N_TABLE_SLOTS, entity_dim) — for trace sub-slot context
    linear_in:   eqx.nn.Linear  # entity_dim → entity_dim
    cat_out:     eqx.nn.Linear  # entity_dim → TABLE_SCALAR_CAT_TOTAL_VOCAB
    cont_out:    eqx.nn.Linear  # entity_dim → TABLE_SCALAR_CONT_N

    def __init__(self, entity_dim, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.slot_embeds = jr.normal(k1, (N_TABLE_SLOTS, entity_dim)) * 0.02
        self.linear_in   = eqx.nn.Linear(entity_dim, entity_dim,               use_bias=False, key=k2)
        self.cat_out     = eqx.nn.Linear(entity_dim, TABLE_SCALAR_CAT_TOTAL_VOCAB, use_bias=False, key=k3)
        self.cont_out    = eqx.nn.Linear(entity_dim, TABLE_SCALAR_CONT_N,      use_bias=False, key=k4)

    def __call__(self, context):
        """context: (entity_dim,) → {'cat': (TABLE_SCALAR_CAT_TOTAL_VOCAB,), 'cont': (TABLE_SCALAR_CONT_N,)}"""
        h = jax.nn.gelu(self.linear_in(context))
        return {'cat': self.cat_out(h), 'cont': self.cont_out(h)}

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

    weights:            logit-group linear heads (TOKEN_HEADS, shared-vocab batching)
    groove_decoder:     shared GrooveDecoder — phrase-level groove + cond scan losses
                        (slot index N_GROOVE_SLOTS = phrase-level slot)
    table_decoder:      TableDecoder — phrase-level table AND instrument's table scalar preds;
                        its weights are also reused for trace scalar preds in cond scan loss
    instr_decoder:      InstrumentDecoder (instrument scalars + softsynth)
    table_proj:         d_model → entity_dim projection for phrase-level table context
    phrase_groove_proj: d_model → entity_dim projection for phrase-level groove context
    """
    weights:            dict[str, Array]
    groove_decoder:     GrooveDecoder
    table_decoder:      TableDecoder
    instr_decoder:      InstrumentDecoder
    table_proj:         eqx.nn.Linear
    phrase_groove_proj: eqx.nn.Linear

    def __init__(self, d_model, entity_dim, key):
        keys = jr.split(key, 6)

        # Logit-group heads
        weights = {}
        for group_name, members in LOGIT_GROUPS.items():
            n     = len(members)
            vocab = members[0][2]
            weights[group_name] = jr.normal(keys[0], (n, vocab, d_model)) / jnp.sqrt(d_model)
        self.weights = weights

        # Shared GrooveDecoder
        self.groove_decoder = GrooveDecoder(entity_dim, keys[1])

        # Single TableDecoder — used for both table and trace predictions
        # (trace predictions in cond_entity_scan_loss reuse these same weights)
        self.table_decoder = TableDecoder(entity_dim, keys[2])

        # Instrument decoder
        self.instr_decoder = InstrumentDecoder(d_model, entity_dim, keys[3])

        # Phrase-level projections
        self.table_proj         = eqx.nn.Linear(d_model, entity_dim, use_bias=False, key=keys[4])
        self.phrase_groove_proj = eqx.nn.Linear(d_model, entity_dim, use_bias=False, key=keys[5])

    def __call__(self, x):
        """
        x: (d_model,) → nested output dict.
        Produces token heads, instrument/table scalar predictions, and phrase-level groove.
        Groove-slot and trace sub-entity losses are computed separately in
        cond_entity_scan_loss using lax.scan + lax.cond.
        """
        result = {}

        # Token heads
        for group_name, members in LOGIT_GROUPS.items():
            logits = self.weights[group_name] @ x
            for i, (name, _, _) in enumerate(members):
                result[name] = logits[i]

        # Phrase-level table (scalar cat + cont only)
        table_ctx = jax.nn.gelu(self.table_proj(x))
        result['table'] = self.table_decoder(table_ctx)

        # Phrase-level groove — shared GrooveDecoder, phrase slot = N_GROOVE_SLOTS
        result['groove'] = self.groove_decoder(jax.nn.gelu(self.phrase_groove_proj(x)), N_GROOVE_SLOTS)

        # Instrument (scalar cat + cont, softsynth; table scalar cat + cont)
        instr_preds, instr_h = self.instr_decoder(x)
        instr_preds['table'] = self.table_decoder(instr_h)
        result['instr'] = instr_preds

        return result

    def log_probs(self, x):
        raw = self(x)
        return {name: jax.nn.log_softmax(raw[name]) for name in TOKEN_HEADS}
