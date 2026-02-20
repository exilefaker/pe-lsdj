import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Key

from pe_lsdj.embedding.song import SequenceEmbedder, SongBanks

# Logit groups: direct softmax heads (same-vocab members batched).
LOGIT_GROUPS = {
    'note': [('note', 0, 158)],
    'fx_cmd': [('fx_cmd', 2, 19)],
    'byte_fx': [('hop_fx', 5, 257), ('volume_fx', 15, 257), ('continuous_fx', 19, 257)],
    'small_enum_fx': [('pan_fx', 6, 5), ('wave_fx', 16, 5)],
    'nibble_fx': [
        ('chord_fx_1', 7, 17), ('chord_fx_2', 8, 17),
        ('env_fx_vol', 9, 17), ('env_fx_fade', 10, 17),
        ('retrig_fx_fade', 11, 17), ('retrig_fx_rate', 12, 17),
        ('vibrato_fx_speed', 13, 17), ('vibrato_fx_depth', 14, 17),
        ('random_fx_l', 17, 17), ('random_fx_r', 18, 17),
    ],
    'transpose': [('transpose', 20, 256)],
}

# Entity reference heads: query-based similarity against embedded banks.
# name → (token_position, vocab_size)
ENTITY_HEADS = {
    'instr_id': (1, 65),
    'table_id': (3, 33),
    'groove_id': (4, 33),
}

# Flat index over all 21 positions: head_name → (token_position, vocab_size)
TOKEN_HEADS = {}
for _members in LOGIT_GROUPS.values():
    for _name, _pos, _vocab in _members:
        TOKEN_HEADS[_name] = (_pos, _vocab)
for _name, (_pos, _vocab) in ENTITY_HEADS.items():
    TOKEN_HEADS[_name] = (_pos, _vocab)


def embed_entity_banks(step_embedder):
    """
    Pre-compute entity bank embeddings for similarity-based output heads.

    Returns dict mapping entity head name → (vocab, query_dim) array.
    Uses the step_embedder's own sub-embedders so that bank embeddings
    and query projections live in the same learned space.
    """
    se = step_embedder

    # Instruments: (64, instr_dim) — no null row (raw 0-63 IDs)
    instr_embs = jax.vmap(se.instrument_embedder.embedder)(
        se.instrument_embedder.entity_bank
    )

    # Tables: (32, table_dim) → prepend null row → (33, table_dim)
    table_entity = se.fx_embedder.embedders['value'].embedders['table_fx']
    table_embs = jax.vmap(table_entity.embedder)(table_entity.entity_bank)
    table_embs = jnp.concatenate([
        jnp.zeros((1, table_embs.shape[1])), table_embs
    ], axis=0)

    # Grooves: bank already has null row (33 entries) → (33, groove_dim)
    groove_entity = se.fx_embedder.embedders['value'].embedders['groove']
    groove_embs = jax.vmap(groove_entity.embedder)(groove_entity.entity_bank)

    return {
        'instr_id': instr_embs,
        'table_id': table_embs,
        'groove_id': groove_embs,
    }


def _entity_dims_from_embedder(step_embedder):
    """Read entity embedding dimensions from a constructed step embedder."""
    se = step_embedder
    return {
        'instr_id': se.instrument_embedder.out_dim,
        'table_id': se.fx_embedder.embedders['value'].embedders['table_fx'].out_dim,
        'groove_id': se.fx_embedder.embedders['value'].embedders['groove'].out_dim,
    }


def hard_targets(tokens):
    """Convert integer tokens (21,) to dict of one-hot distribution arrays."""
    return {
        name: jax.nn.one_hot(tokens[pos], vocab)
        for name, (pos, vocab) in TOKEN_HEADS.items()
    }


def token_loss(logits_dict, target_dists):
    """
    Soft cross-entropy loss over all 21 token positions.

    logits_dict: dict[str, (vocab_i,)] logit arrays
    target_dists: dict[str, (vocab_i,)] probability vectors
    Returns: scalar (sum of per-position soft CE)
    """
    total = 0.0
    for name in logits_dict:
        log_probs = jax.nn.log_softmax(logits_dict[name])
        total += -jnp.sum(target_dists[name] * log_probs)
    return total


class OutputHeads(eqx.Module):
    """
    Output heads mapping d_model → per-token logits.

    Two kinds of heads:
    - Logit heads: grouped linear projections (batched by vocab size)
    - Entity heads: query projections → scaled dot-product similarity
      against pre-computed entity bank embeddings

    Both produce the same output format: dict[str, (vocab,)] logit arrays.
    """
    weights: dict[str, Array]              # group_name → (N, vocab, d_model)
    entity_projections: dict[str, Array]   # entity_name → (query_dim, d_model)
    entity_bank_embs: dict[str, Array]     # entity_name → (vocab, query_dim)

    def __init__(self, d_model, key, entity_dims=None):
        if entity_dims is None:
            entity_dims = {'instr_id': 128, 'table_id': 64, 'groove_id': 64}

        keys = jr.split(key, len(LOGIT_GROUPS) + len(ENTITY_HEADS))

        # Logit group weights
        weights = {}
        for i, (group_name, members) in enumerate(LOGIT_GROUPS.items()):
            n = len(members)
            vocab = members[0][2]
            weights[group_name] = (
                jr.normal(keys[i], (n, vocab, d_model)) / jnp.sqrt(d_model)
            )
        self.weights = weights

        # Entity query projections + placeholder bank embeddings
        entity_projections = {}
        entity_bank_embs = {}
        offset = len(LOGIT_GROUPS)
        for j, (name, (pos, vocab)) in enumerate(ENTITY_HEADS.items()):
            q_dim = entity_dims[name]
            entity_projections[name] = (
                jr.normal(keys[offset + j], (q_dim, d_model)) / jnp.sqrt(d_model)
            )
            entity_bank_embs[name] = jnp.zeros((vocab, q_dim))
        self.entity_projections = entity_projections
        self.entity_bank_embs = entity_bank_embs

    def __call__(self, x):
        """x: (d_model,) → dict[str, (vocab,)] logit arrays."""
        result = {}
        # Logit groups (batched matmul)
        for group_name, members in LOGIT_GROUPS.items():
            logits = self.weights[group_name] @ x  # (N, vocab)
            for i, (head_name, _, _) in enumerate(members):
                result[head_name] = logits[i]
        # Entity heads (query similarity)
        for name in ENTITY_HEADS:
            query = self.entity_projections[name] @ x  # (query_dim,)
            scale = jnp.sqrt(jnp.float32(query.shape[0]))
            result[name] = (self.entity_bank_embs[name] @ query) / scale
        return result

    def log_probs(self, x):
        """x: (d_model,) → dict[str, (vocab,)] log-softmax arrays."""
        raw = self(x)
        return {name: jax.nn.log_softmax(v) for name, v in raw.items()}


def _norm2d(norm, x):
    """Apply LayerNorm(d) over (S, 4, d) by double-vmapping."""
    return jax.vmap(jax.vmap(norm))(x)


class AxialTransformerBlock(eqx.Module):
    """
    One block of axial attention over (S, 4, d):
      1. Temporal self-attention (causal, per channel)
      2. Cross-channel self-attention (full, per timestep)
      3. Position-wise MLP
    """
    temporal_attn: eqx.nn.MultiheadAttention
    channel_attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    norm_t: eqx.nn.LayerNorm
    norm_c: eqx.nn.LayerNorm
    norm_mlp: eqx.nn.LayerNorm

    def __init__(self, d_model, num_heads_t, num_heads_c, key):
        k1, k2, k3 = jr.split(key, 3)
        self.temporal_attn = eqx.nn.MultiheadAttention(
            num_heads_t, d_model, key=k1,
        )
        self.channel_attn = eqx.nn.MultiheadAttention(
            num_heads_c, d_model, key=k2,
        )
        self.mlp = eqx.nn.MLP(
            d_model, d_model, d_model * 4, depth=1, key=k3,
        )
        self.norm_t = eqx.nn.LayerNorm(d_model)
        self.norm_c = eqx.nn.LayerNorm(d_model)
        self.norm_mlp = eqx.nn.LayerNorm(d_model)

    def __call__(self, x: Array, causal_mask: Bool[Array, "S S"]) -> Array:
        """
        x: (S, 4, d)
        causal_mask: (S, S) boolean lower-triangular
        Returns: (S, 4, d)
        """
        # 1. Channel attention — vmap over S timesteps
        normed = _norm2d(self.norm_c, x)

        def _c_attn(x_t):  # (4, d) → (4, d)
            return self.channel_attn(x_t, x_t, x_t)

        x = x + jax.vmap(_c_attn)(normed)

        # 2. Temporal attention — vmap over 4 channels
        normed = _norm2d(self.norm_t, x)  # (S, 4, d)

        def _t_attn(x_ch):  # (S, d) → (S, d)
            return self.temporal_attn(x_ch, x_ch, x_ch, mask=causal_mask)

        x = x + jax.vmap(_t_attn, in_axes=1, out_axes=1)(normed)

        # 3. MLP — per token
        normed = _norm2d(self.norm_mlp, x)
        x = x + jax.vmap(jax.vmap(self.mlp))(normed)

        return x


class LSDJTransformer(eqx.Module):
    """
    Axial transformer for LSDJ 4-channel music data.

    Input:  song_tokens (S, 4, 21)
    Output: dict[str, (S, 4, vocab)] logit arrays (one per token head)
    """
    embedder: SequenceEmbedder
    blocks: list[AxialTransformerBlock]
    final_norm: eqx.nn.LayerNorm
    output_heads: OutputHeads
    d_model: int

    def __init__(
        self,
        key: Key,
        *,
        d_model: int = 256,
        num_heads_t: int = 4,
        num_heads_c: int = 2,
        num_blocks: int = 6,
        banks: SongBanks | None = None,
        **embedder_kwargs,
    ):
        keys = jr.split(key, num_blocks + 3)

        self.d_model = d_model
        self.embedder = SequenceEmbedder.create(
            keys[0],
            banks=banks,
            out_dim=d_model * 4,
            **embedder_kwargs,
        )

        self.blocks = [
            AxialTransformerBlock(d_model, num_heads_t, num_heads_c, keys[i + 1])
            for i in range(num_blocks)
        ]

        self.final_norm = eqx.nn.LayerNorm(d_model)

        entity_dims = _entity_dims_from_embedder(self.embedder.step_embedder)
        self.output_heads = OutputHeads(d_model, keys[-1], entity_dims)

    def __call__(self, song_tokens: Array):
        """
        song_tokens: (S, 4, 21)
        Returns: dict[str, (S, 4, vocab_i)] logit arrays
        """
        x = self.embedder(song_tokens)  # (S, 4, d_model)
        S = x.shape[0]
        causal_mask = jnp.tril(jnp.ones((S, S), dtype=bool))

        for block in self.blocks:
            x = block(x, causal_mask)

        x = _norm2d(self.final_norm, x)  # (S, 4, d_model)

        # Apply output heads per-token → dict of (S, 4, vocab_i)
        return jax.vmap(jax.vmap(self.output_heads))(x)

    def with_banks(self, banks: SongBanks):
        """Return a new model with swapped entity banks and refreshed output head embeddings."""
        new_embedder = self.embedder.with_banks(banks)
        new_bank_embs = embed_entity_banks(new_embedder.step_embedder)
        model = eqx.tree_at(lambda m: m.embedder, self, new_embedder)
        return eqx.tree_at(
            lambda m: m.output_heads.entity_bank_embs, model, new_bank_embs
        )
