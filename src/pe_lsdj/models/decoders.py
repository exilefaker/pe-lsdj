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

# FX value groups: these heads are conditioned on the sampled/target fx_cmd.
FX_VAL_GROUPS = ('byte_fx', 'small_enum_fx', 'nibble_fx')

LOGIT_GROUPS = {
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

FX_VAL_HEAD_NAMES = frozenset(
    name for gname in FX_VAL_GROUPS for name, _, _ in LOGIT_GROUPS[gname]
)

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

INSTR_SCALAR_SPECS = [
    spec for i, spec in enumerate(INSTR_FIELD_SPECS)
    if i not in (INSTR_TABLE_COL, INSTR_SOFTSYNTH_COL)
]
INSTR_SCALAR_COL_INDICES = [
    i for i in range(len(INSTR_FIELD_SPECS))
    if i not in (INSTR_TABLE_COL, INSTR_SOFTSYNTH_COL)
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

    Slot embeddings (shape: (N_GROOVE_SLOTS + 1) × context_dim) are added to the
    context_dim context before decoding. Shared across ALL groove levels:
      indices 0..N_GROOVE_SLOTS-1 — per-table/trace slot predictions
      index  N_GROOVE_SLOTS       — phrase-level groove (special phrase slot)
    Context distinguishes the level; slot_embeds distinguish position.
    """
    slot_embeds: Array          # (N_GROOVE_SLOTS + 1, context_dim)
    linear_in:   eqx.nn.Linear  # context_dim → context_dim
    linear_out:  eqx.nn.Linear  # context_dim → GROOVE_CONT_N

    def __init__(self, context_dim, key):
        k1, k2, k3 = jr.split(key, 3)
        self.slot_embeds = jr.normal(k1, (N_GROOVE_SLOTS + 1, context_dim)) * 0.02
        self.linear_in   = eqx.nn.Linear(context_dim, context_dim,       use_bias=False, key=k2)
        self.linear_out  = eqx.nn.Linear(context_dim, GROOVE_CONT_N,     use_bias=False, key=k3)

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
    slot_embeds: Array          # (N_TABLE_SLOTS, context_dim) — for trace sub-slot context
    linear_in:   eqx.nn.Linear  # context_dim → context_dim
    cat_out:     eqx.nn.Linear  # context_dim → TABLE_SCALAR_CAT_TOTAL_VOCAB
    cont_out:    eqx.nn.Linear  # context_dim → TABLE_SCALAR_CONT_N

    def __init__(self, context_dim, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.slot_embeds = jr.normal(k1, (N_TABLE_SLOTS, context_dim)) * 0.02
        self.linear_in   = eqx.nn.Linear(context_dim, context_dim,                   use_bias=False, key=k2)
        self.cat_out     = eqx.nn.Linear(context_dim, TABLE_SCALAR_CAT_TOTAL_VOCAB,  use_bias=False, key=k3)
        self.cont_out    = eqx.nn.Linear(context_dim, TABLE_SCALAR_CONT_N,           use_bias=False, key=k4)

    def __call__(self, context):
        """context: (entity_dim,) → {
            'cat': (TABLE_SCALAR_CAT_TOTAL_VOCAB,), 
            'cont': (TABLE_SCALAR_CONT_N,)
        }"""
        h = jax.nn.gelu(self.linear_in(context))
        return {'cat': self.cat_out(h), 'cont': self.cont_out(h)}

    def encode(self, context):
        """entity_dim latent for cosine-similarity bank matching."""
        return self.linear_in(context)


class SoftSynthDecoder(eqx.Module):
    """
    Softsynth + waveframe decoder.
    Conditions on the instrument latent (in_dim = instr_entity_dim), projects to out_dim = softsynth_entity_dim.
    """
    linear_in:     eqx.nn.Linear   # in_dim → out_dim
    cat_out:       eqx.nn.Linear   # out_dim → SOFTSYNTH_CAT_TOTAL_VOCAB
    cont_out:      eqx.nn.Linear   # out_dim → SOFTSYNTH_CONT_N
    waveframe_out: eqx.nn.Linear   # out_dim → WAVEFRAME_DIM

    def __init__(self, in_dim, out_dim, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.linear_in     = eqx.nn.Linear(in_dim,  out_dim,                   use_bias=False, key=k1)
        self.cat_out       = eqx.nn.Linear(out_dim, SOFTSYNTH_CAT_TOTAL_VOCAB, use_bias=False, key=k2)
        self.cont_out      = eqx.nn.Linear(out_dim, SOFTSYNTH_CONT_N,           use_bias=False, key=k3)
        self.waveframe_out = eqx.nn.Linear(out_dim, WAVEFRAME_DIM,              use_bias=False, key=k4)

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
    Instrument's table uses the shared table_decoder (via OutputHeads.instr_to_table_proj)
    with the instr_dim latent bridged to table_entity_dim context.
    Softsynth/waveframes condition on the instr_dim latent (subordinate).
    """
    linear_in:         eqx.nn.Linear    # d_model → instr_dim
    cat_out:           eqx.nn.Linear    # instr_dim → INSTR_SCALAR_CAT_TOTAL_VOCAB
    cont_out:          eqx.nn.Linear    # instr_dim → INSTR_SCALAR_CONT_N
    softsynth_decoder: SoftSynthDecoder

    def __init__(self, d_model, instr_dim, softsynth_dim, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.linear_in         = eqx.nn.Linear(d_model,   instr_dim,                   use_bias=False, key=k1)
        self.cat_out           = eqx.nn.Linear(instr_dim, INSTR_SCALAR_CAT_TOTAL_VOCAB, use_bias=False, key=k2)
        self.cont_out          = eqx.nn.Linear(instr_dim, INSTR_SCALAR_CONT_N,           use_bias=False, key=k3)
        self.softsynth_decoder = SoftSynthDecoder(instr_dim, softsynth_dim, k4)

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

    weights:             logit-group linear heads (TOKEN_HEADS, shared-vocab batching)
    groove_decoder:      shared GrooveDecoder — phrase-level groove + cond scan losses
                         (slot index N_GROOVE_SLOTS = phrase-level slot)
    table_decoder:       TableDecoder — phrase-level table AND instrument's table scalar preds;
                         its weights are also reused for trace scalar preds in conditional loss
    instr_decoder:       InstrumentDecoder (instrument scalars + softsynth)
    table_proj:          d_model → table_entity_dim projection for phrase-level table context
    phrase_groove_proj:  d_model → table_entity_dim projection for phrase-level groove context
    instr_to_table_proj: instr_entity_dim → table_entity_dim bridge for instrument's table context
    """
    weights:             dict[str, Array]
    fx_cmd_cond:         eqx.nn.Linear
    groove_decoder:      GrooveDecoder
    table_decoder:       TableDecoder
    instr_decoder:       InstrumentDecoder
    table_proj:          eqx.nn.Linear
    phrase_groove_proj:  eqx.nn.Linear
    instr_to_table_proj: eqx.nn.Linear

    def __init__(self, d_model, instr_entity_dim, table_entity_dim, softsynth_entity_dim, key):
        keys = jr.split(key, 10)

        # Logit-group heads
        weights = {}
        for group_name, members in LOGIT_GROUPS.items():
            n     = len(members)
            vocab = members[0][2]
            weights[group_name] = jr.normal(keys[0], (n, vocab, d_model)) / jnp.sqrt(d_model)
        # Factorized note heads: chroma (0=NULL, 1-12=C..B) and octave (0=NULL, 1-13=oct3..octF)
        weights['note_chroma'] = jr.normal(keys[8], (1, NUM_CHROMA,  d_model)) / jnp.sqrt(d_model)
        weights['note_oct']    = jr.normal(keys[9], (1, NUM_OCTAVES, d_model)) / jnp.sqrt(d_model)
        self.weights = weights

        # FX command conditioning: Linear(19, d_model) used as one_hot(fx_cmd) @ weight
        self.fx_cmd_cond = eqx.nn.Linear(19, d_model, use_bias=False, key=keys[1])

        # Shared GrooveDecoder
        self.groove_decoder = GrooveDecoder(table_entity_dim, keys[2])

        # Single TableDecoder — used for both table and trace predictions
        # (trace predictions in cond_entity_scan_loss reuse these same weights)
        self.table_decoder = TableDecoder(table_entity_dim, keys[3])

        # Instrument decoder
        self.instr_decoder = InstrumentDecoder(d_model, instr_entity_dim, softsynth_entity_dim, keys[4])

        # Phrase-level projections
        self.table_proj          = eqx.nn.Linear(d_model, table_entity_dim,           use_bias=False, key=keys[5])
        self.phrase_groove_proj  = eqx.nn.Linear(d_model, table_entity_dim,           use_bias=False, key=keys[6])
        self.instr_to_table_proj = eqx.nn.Linear(instr_entity_dim, table_entity_dim,  use_bias=False, key=keys[7])
    


    def __call__(self, x, fx_cmd_token=None):
        """
        x: (d_model,) → nested output dict.
        fx_cmd_token:
          None (default, scalar) — inference mode: fx_cmd logits are computed first;
            their softmax gives a soft expected conditioning signal for fx_val heads.
            Differentiable and self-consistent (no external information needed).
          scalar int — teacher-forcing mode: fx_val heads are conditioned on this
            known target cmd via one-hot. Used by sequence_loss (per-channel scalar
            after double-vmap over (S, 4)).
        Produces token heads, instrument/table scalar predictions, and phrase-level groove.
        Groove-slot and trace sub-entity losses are computed separately in cond_entity_loss.
        """
        result = {}

        # FX cmd always computed from plain x (unconditioned)
        fx_cmd_logits = (self.weights['fx_cmd'] @ x)[0]
        result['fx_cmd'] = fx_cmd_logits

        # Conditioning signal for fx_val heads
        if fx_cmd_token is None:
            # Inference: soft conditioning — expected embedding over predicted cmd distribution
            fx_cmd_signal = jax.nn.softmax(fx_cmd_logits)
        else:
            # Teacher forcing: hard one-hot on the known target cmd
            fx_cmd_signal = jax.nn.one_hot(jnp.int32(fx_cmd_token), 19)
        x_cond = x + self.fx_cmd_cond(fx_cmd_signal)

        # Factorized note heads (plain x, not fx-conditioned)
        result['note_chroma'] = (self.weights['note_chroma'] @ x)[0]
        result['note_oct']    = (self.weights['note_oct']    @ x)[0]

        # Token heads: fx_val groups conditioned on fx_cmd; all others use plain x
        for group_name, members in LOGIT_GROUPS.items():
            if group_name == 'fx_cmd':
                continue  # already done above
            src = x_cond if group_name in FX_VAL_GROUPS else x
            logits = self.weights[group_name] @ src
            for i, (name, _, _) in enumerate(members):
                result[name] = logits[i]

        # Phrase-level table (scalar cat + cont only)
        table_ctx = jax.nn.gelu(self.table_proj(x))
        result['table'] = self.table_decoder(table_ctx)

        # Phrase-level groove — shared GrooveDecoder, phrase slot = N_GROOVE_SLOTS
        result['groove'] = self.groove_decoder(jax.nn.gelu(self.phrase_groove_proj(x)), N_GROOVE_SLOTS)

        # Instrument (scalar cat + cont, softsynth; table scalar cat + cont)
        instr_preds, instr_h = self.instr_decoder(x)
        instr_table_ctx = jax.nn.gelu(self.instr_to_table_proj(instr_h))
        instr_preds['table'] = self.table_decoder(instr_table_ctx)
        result['instr'] = instr_preds

        return result

    def conditioned_fx_val_logits(self, x, fx_cmd):
        """
        Compute only the fx_val head logits, conditioned on a (sampled) fx_cmd.

        x:      (d_model,) backbone repr
        fx_cmd: scalar int — the sampled fx_cmd token value
        Returns dict of {name: logits} for all names in FX_VAL_HEAD_NAMES.
        """
        x_cond = x + self.fx_cmd_cond(jax.nn.one_hot(jnp.int32(fx_cmd), 19))
        result = {}
        for group_name in FX_VAL_GROUPS:
            logits = self.weights[group_name] @ x_cond
            for i, (name, _, _) in enumerate(LOGIT_GROUPS[group_name]):
                result[name] = logits[i]
        return result

    def generation_outputs(self, x):
        """
        Like __call__ but also returns the context vectors needed by match_* and
        the raw backbone repr 'x' needed for conditioned_fx_val_logits.

        Returns (logits, latents) where latents contains:
          'x'              — (d_model,) raw backbone repr (for conditioned_fx_val_logits)
          'table_ctx'      — (table_entity_dim,) phrase-level table context
          'instr_table_ctx'— (table_entity_dim,) instrument's table context
          'phrase_groove_ctx' — (table_entity_dim,) phrase-level groove context

        These are the 'table_hidden' arguments to match_table / match_groove /
        match_trace for the two table-matching call sites in resolve_step.
        Computing them here avoids recomputing inside the match functions.
        Note: fx_cmd logits are included here with unconditioned x (fx_cmd is not yet known).
        fx_val logits are NOT included here — call conditioned_fx_val_logits after sampling fx_cmd.
        """
        logits = {}

        # Factorized note heads
        logits['note_chroma'] = (self.weights['note_chroma'] @ x)[0]
        logits['note_oct']    = (self.weights['note_oct']    @ x)[0]

        for group_name, members in LOGIT_GROUPS.items():
            if group_name in FX_VAL_GROUPS:
                continue  # fx_val logits computed later via conditioned_fx_val_logits
            w = self.weights[group_name] @ x
            for i, (name, _, _) in enumerate(members):
                logits[name] = w[i]

        table_ctx = jax.nn.gelu(self.table_proj(x))
        logits['table'] = self.table_decoder(table_ctx)

        phrase_groove_ctx = jax.nn.gelu(self.phrase_groove_proj(x))
        logits['groove'] = self.groove_decoder(phrase_groove_ctx, N_GROOVE_SLOTS)

        instr_preds, instr_h = self.instr_decoder(x)
        instr_table_ctx = jax.nn.gelu(self.instr_to_table_proj(instr_h))
        instr_preds['table'] = self.table_decoder(instr_table_ctx)
        logits['instr'] = instr_preds

        latents = {
            'x':                 x,
            'table_ctx':         table_ctx,
            'instr_table_ctx':   instr_table_ctx,
            'phrase_groove_ctx': phrase_groove_ctx,
        }
        return logits, latents

    def log_probs(self, x):
        raw = self(x)  # uses soft inference-mode conditioning
        return {name: jax.nn.log_softmax(raw[name]) for name in TOKEN_HEADS}
