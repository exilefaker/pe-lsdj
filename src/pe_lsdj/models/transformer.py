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

# Flat index of logit-group heads: head_name → (token_position, vocab_size).
TOKEN_HEADS = {}
for _members in LOGIT_GROUPS.values():
    for _name, _pos, _vocab in _members:
        TOKEN_HEADS[_name] = (_pos, _vocab)

# Entity heads: token position for each entity type in song_tokens.
# These are NOT in TOKEN_HEADS; they use a separate generative decoder.
ENTITY_HEADS = {
    'instr_id':  1,   # instrument ID
    'table_id':  3,   # TABLE_FX value (A command)
    'groove_id': 4,   # GROOVE_FX value (G command)
}

# ---------------------------------------------------------------------------
# Entity parameter field specs.
#
# Each list defines (field_label, vocab_size) for every column of the
# corresponding SongBanks array (row 0 is the null sentinel, so column
# ordering matches the underlying parse_* dict / reshape order).
#
# The entity decoder produces a flat logit vector; these specs define how
# to split it into per-field slices for the CE loss and for generation.
# ---------------------------------------------------------------------------

# Instruments bank columns match parse_instruments() dict insertion order.
INSTR_FIELD_SPECS = [
    (TYPE_ID,            5),   # 0=NULL, 1=PU, 2=WAV, 3=KIT, 4=NOI
    (TABLE,             33),   # 0=NULL, 1-32
    (TABLE_ON_OFF,       2),   # 0/1 bit (no +1 offset)
    (TABLE_AUTOMATE,     2),   # 0/1 bit
    (AUTOMATE_2,         2),   # 0/1 bit
    (PAN,                5),   # 0=NULL, 1-4
    (VIBRATO_TYPE,       5),   # 0=NULL, 1-4
    (VIBRATO_DIRECTION,  3),   # 0=NULL, 1-2
    (ENV_VOLUME,        17),   # 0=NULL, 1-16 (nibble+1)
    (ENV_FADE,          17),
    (LENGTH,            65),   # 0=NULL, 1-64 (6-bit+1)
    (LENGTH_LIMITED,     3),   # 0=NULL, 1-2
    (SWEEP,            257),   # 0=NULL, 1-256 (byte+1)
    (VOLUME,             5),   # 0=NULL, 1-4
    (PHASE_TRANSPOSE,  257),
    (WAVE,               5),   # 0=NULL, 1-4
    (PHASE_FINETUNE,    17),
    (SOFTSYNTH_ID,      17),   # 0=NULL, 1-16
    (REPEAT,            17),
    (PLAY_TYPE,          5),   # 0=NULL, 1-4
    (WAVE_LENGTH,       17),
    (SPEED,             17),
    (KEEP_ATTACK_1,      3),   # 0=NULL, 1-2
    (KEEP_ATTACK_2,      3),
    (KIT_1_ID,          65),   # 0=NULL, 1-64
    (KIT_2_ID,          65),
    (LENGTH_KIT_1,     257),
    (LENGTH_KIT_2,     257),
    (LOOP_KIT_1,         3),
    (LOOP_KIT_2,         3),
    (OFFSET_KIT_1,     257),
    (OFFSET_KIT_2,     257),
    (HALF_SPEED,         3),
    (PITCH,            257),
    (DISTORTION_TYPE,    5),
]
assert len(INSTR_FIELD_SPECS) == INSTR_WIDTH

# FX value vocab sizes in FX_VALUE_KEYS order (shared by both FX slots).
_FX_VAL_VOCABS = [33, 33, 257, 5, 17, 17, 17, 17, 17, 17, 17, 17, 257, 5, 17, 17, 257]

# Tables bank: column-stacked from parse_tables() dict, each field reshaped
# to (NUM_TABLES, -1) before stacking.  Layout is field-major; within
# TABLE_FX_VALUE_* the sub-layout is step-major (from reshape of (N, 16, 17)).
TABLE_FIELD_SPECS = (
    [(f'env_vol_{i}',          17) for i in range(STEPS_PER_TABLE)] +
    [(f'env_dur_{i}',          17) for i in range(STEPS_PER_TABLE)] +
    [(f'transpose_{i}',       257) for i in range(STEPS_PER_TABLE)] +
    [(f'fx1_{i}',              19) for i in range(STEPS_PER_TABLE)] +
    [(f'fx1_val_{i}_{j}', _FX_VAL_VOCABS[j])
     for i in range(STEPS_PER_TABLE) for j in range(17)] +
    [(f'fx2_{i}',              19) for i in range(STEPS_PER_TABLE)] +
    [(f'fx2_val_{i}_{j}', _FX_VAL_VOCABS[j])
     for i in range(STEPS_PER_TABLE) for j in range(17)]
)
assert len(TABLE_FIELD_SPECS) == 624

# Grooves bank: reshape of (NUM_GROOVES, STEPS_PER_GROOVE, 2) → step-major
# [even_ticks, odd_ticks] per step.
GROOVE_FIELD_SPECS = [
    (f'step{i}_{tick}', 17)
    for i in range(STEPS_PER_GROOVE) for tick in ('even', 'odd')
]

ENTITY_HEAD_SPECS = {
    'instr_id':  INSTR_FIELD_SPECS,
    'table_id':  TABLE_FIELD_SPECS,
    'groove_id': GROOVE_FIELD_SPECS,
}

# Precomputed total output size per entity head.
ENTITY_HEAD_TOTAL_VOCAB = {
    name: sum(v for _, v in specs)
    for name, specs in ENTITY_HEAD_SPECS.items()
}


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def hard_targets(tokens):
    """
    Convert integer tokens (21,) → dict of one-hot arrays for logit-group
    heads only.  Entity heads use entity_param_loss instead.
    """
    return {
        name: jax.nn.one_hot(tokens[pos], vocab)
        for name, (pos, vocab) in TOKEN_HEADS.items()
    }


def token_loss(logits_dict, target_dists):
    """
    Soft cross-entropy over all logit-group positions.

    logits_dict:  dict[str, (vocab_i,)]
    target_dists: dict[str, (vocab_i,)] probability vectors
    Returns: scalar
    """
    total = 0.0
    for name in TOKEN_HEADS:
        log_probs = jax.nn.log_softmax(logits_dict[name])
        total += -jnp.sum(target_dists[name] * log_probs)
    return total


def entity_loss(entity_logits_dict, banks: SongBanks, target_tokens):
    """
    Cross-entropy loss over entity descriptions for one (step, channel).

    For each entity head the model outputs a flat logit vector covering all
    parameter fields.  We look up the target entity's parameters from the
    null-prepended bank (token 0 → null sentinel row, token k → entity k-1),
    then compute per-field CE and average over fields.

    entity_logits_dict: dict[str, (total_field_vocab,)] from OutputHeads
    banks:              SongBanks (null rows already prepended)
    target_tokens:      (21,) integer array for one (step, channel)
    Returns: scalar loss
    """
    target_tokens = jnp.int32(target_tokens)
    bank_arrays = {
        'instr_id':  banks.instruments[target_tokens[ENTITY_HEADS['instr_id']]],
        'table_id':  banks.tables     [target_tokens[ENTITY_HEADS['table_id']]],
        'groove_id': banks.grooves    [target_tokens[ENTITY_HEADS['groove_id']]],
    }

    total = 0.0
    for name, specs in ENTITY_HEAD_SPECS.items():
        flat_logits = entity_logits_dict[name]   # (total_field_vocab,)
        param_vals  = bank_arrays[name]          # (n_fields,) integer targets
        n_fields = len(specs)
        offset = 0
        field_ce = 0.0
        for i, (_, vocab) in enumerate(specs):
            lp = jax.nn.log_softmax(flat_logits[offset:offset + vocab])
            field_ce += -lp[param_vals[i]]
            offset += vocab
        total += field_ce / n_fields
    return total


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class EntityDecoder(eqx.Module):
    """
    Two-layer MLP: d_model → entity_dim → total_field_vocab.

    The entity_dim latent is exposed via encode() for cosine-similarity
    matching at inference time (before the non-linearity is applied to
    the entity_dim output, it serves as the latent interface between
    the transformer and a future reconstruction-based entity model).
    """
    linear_in:  eqx.nn.Linear   # d_model    → entity_dim
    linear_out: eqx.nn.Linear   # entity_dim → total_field_vocab

    def __init__(self, d_model, entity_dim, total_field_vocab, key):
        k1, k2 = jr.split(key)
        self.linear_in  = eqx.nn.Linear(d_model, entity_dim, use_bias=False, key=k1)
        self.linear_out = eqx.nn.Linear(entity_dim, total_field_vocab, use_bias=False, key=k2)

    def __call__(self, x):
        return self.linear_out(jax.nn.gelu(self.linear_in(x)))

    def encode(self, x):
        """Project d_model → entity_dim (for cosine-similarity entity matching)."""
        return self.linear_in(x)


class OutputHeads(eqx.Module):
    """
    Output heads mapping d_model → per-token logits.

    Logit heads:  grouped linear projections (batched by vocab size).
    Entity heads: two-layer MLP d_model → entity_dim → total_field_vocab,
                  producing flat logits split by ENTITY_HEAD_SPECS.

    __call__ returns dict[str, array]:
      logit-group heads → (vocab,)
      entity heads      → (total_field_vocab,)  [split via ENTITY_HEAD_SPECS]
    """
    weights:         dict[str, Array]          # group_name → (N, vocab, d_model)
    entity_decoders: dict[str, EntityDecoder]  # entity_name → MLP

    def __init__(self, d_model, entity_dim, key):
        keys = jr.split(key, len(LOGIT_GROUPS) + len(ENTITY_HEAD_SPECS))

        weights = {}
        for i, (group_name, members) in enumerate(LOGIT_GROUPS.items()):
            n = len(members)
            vocab = members[0][2]
            weights[group_name] = (
                jr.normal(keys[i], (n, vocab, d_model)) / jnp.sqrt(d_model)
            )
        self.weights = weights

        entity_decoders = {}
        offset = len(LOGIT_GROUPS)
        for j, (name, total_vocab) in enumerate(ENTITY_HEAD_TOTAL_VOCAB.items()):
            entity_decoders[name] = EntityDecoder(
                d_model, entity_dim, total_vocab, keys[offset + j]
            )
        self.entity_decoders = entity_decoders

    def __call__(self, x):
        """x: (d_model,) → dict[str, array] logit arrays."""
        result = {}
        for group_name, members in LOGIT_GROUPS.items():
            logits = self.weights[group_name] @ x      # (N, vocab)
            for i, (head_name, _, _) in enumerate(members):
                result[head_name] = logits[i]
        for name in ENTITY_HEAD_SPECS:
            result[name] = self.entity_decoders[name](x)  # (total_field_vocab,)
        return result

    def log_probs(self, x):
        """Logit-group heads only: x → dict[str, (vocab,)] log-softmax."""
        raw = self(x)
        return {name: jax.nn.log_softmax(raw[name]) for name in TOKEN_HEADS}


def _norm2d(norm, x):
    """Apply LayerNorm(d) over (S, 4, d) by double-vmapping."""
    return jax.vmap(jax.vmap(norm))(x)


class AxialTransformerBlock(eqx.Module):
    """
    One block of axial attention over (S, 4, d):
      1. Cross-channel self-attention (full, per timestep)
      2. Temporal self-attention (causal, per channel)
      3. Position-wise MLP

    Axial attention: https://arxiv.org/abs/1912.12180
    """
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
        """x: (S, 4, d) → (S, 4, d)"""
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
    """
    Axial transformer for LSDJ 4-channel music data.

    Input:  song_tokens (S, 4, 21)
    Output: dict[str, (S, 4, ...)] logit arrays
      logit-group heads → (S, 4, vocab_i)
      entity heads      → (S, 4, total_field_vocab_i)
    """
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
            "d_model": d_model,
            "entity_dim": entity_dim,
            "num_heads_t": num_heads_t,
            "num_heads_c": num_heads_c,
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
        self.final_norm  = eqx.nn.LayerNorm(d_model)
        self.output_heads = OutputHeads(d_model, entity_dim, keys[-1])

    def __call__(self, song_tokens: Array):
        """song_tokens: (S, 4, 21) → dict[str, (S, 4, ...)] logit arrays"""
        x = self.embedder(song_tokens)          # (S, 4, d_model)
        S = x.shape[0]
        causal_mask = jnp.tril(jnp.ones((S, S), dtype=bool))
        for block in self.blocks:
            x = block(x, causal_mask)
        x = _norm2d(self.final_norm, x)         # (S, 4, d_model)
        return jax.vmap(jax.vmap(self.output_heads))(x)

    def with_banks(self, banks: SongBanks):
        """Return a new model with swapped entity banks in the input embedder."""
        new_embedder = self.embedder.with_banks(banks)
        return eqx.tree_at(lambda m: m.embedder, self, new_embedder)

    def write_metadata(self, filepath):
        with open(filepath, "w") as f:
            f.write(json.dumps(self.metadata))
