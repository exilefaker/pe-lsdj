import json
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Key

from pe_lsdj.embedding.song import SequenceEmbedder, SongBanks
from pe_lsdj.models.decoders import (
    # Public specs / constants
    LOGIT_GROUPS, TOKEN_HEADS, ENTITY_HEADS, WAVEFRAME_DIM,
    INSTR_FIELD_SPECS, TABLE_FIELD_SPECS, GROOVE_FIELD_SPECS, SOFTSYNTH_FIELD_SPECS,
    INSTR_SCALAR_SPECS, INSTR_SCALAR_COL_INDICES,
    TABLE_SCALAR_SPECS, TABLE_SCALAR_COL_INDICES,
    TABLE_FX_COL_INDICES, GROOVE_FX_COL_INDICES,
    INSTR_SCALAR_CAT_SPECS, INSTR_SCALAR_CAT_COL_INDICES, INSTR_SCALAR_CONT_N,
    TABLE_SCALAR_CAT_SPECS, TABLE_SCALAR_CAT_COL_INDICES, TABLE_SCALAR_CONT_N,
    SOFTSYNTH_CAT_SPECS, SOFTSYNTH_CAT_COL_INDICES, SOFTSYNTH_CONT_N,
    GROOVE_CONT_N,
    TABLE_SCALAR_CAT_TOTAL_VOCAB, INSTR_SCALAR_CAT_TOTAL_VOCAB, SOFTSYNTH_CAT_TOTAL_VOCAB,
    N_TABLE_SLOTS, N_GROOVE_SLOTS,
    # Private arrays needed by loss functions
    _INSTR_TABLE_COL, _INSTR_SOFTSYNTH_COL,
    _TABLE_FX_COLS_ARRAY, _GROOVE_FX_COLS_ARRAY, _TABLE_CAT_TRACE_MASK,
    _INSTR_SCALAR_CAT_GROUPS, _TABLE_SCALAR_CAT_GROUPS, _SOFTSYNTH_CAT_GROUPS,
    _INSTR_SCALAR_CONT_COLS_ARRAY, _TABLE_SCALAR_CONT_COLS_ARRAY, _SOFTSYNTH_CONT_COLS_ARRAY,
    _INSTR_SCALAR_CONT_MAX_VALUES, _TABLE_SCALAR_CONT_MAX_VALUES, _SOFTSYNTH_CONT_MAX_VALUES,
    _GROOVE_CONT_MAX,
    # Decoder classes
    EntityDecoder, GrooveDecoder, TableDecoder, SoftSynthDecoder, InstrumentDecoder, OutputHeads,
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


def _table_loss(preds, table_row):
    """
    Scalar loss components for one table (or trace) prediction.
    Returns [cat_loss, cont_loss] — 2 components.
    Groove-slot and trace sub-entity losses are computed separately by
    cond_entity_scan_loss using lax.scan + lax.cond.
    """
    return [
        _ce_loss_grouped(preds['cat'], table_row, _TABLE_SCALAR_CAT_GROUPS),
        _mse_loss(preds['cont'], table_row,
                  _TABLE_SCALAR_CONT_COLS_ARRAY, _TABLE_SCALAR_CONT_MAX_VALUES),
    ]


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
    losses.extend(_table_loss(p['table'], it_row))

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
    losses.extend(_table_loss(entity_preds['table'], pt_row))

    # ─── Phrase-level groove ─────────────────────────────────────────────────
    groove_row = banks.grooves[target_tokens[ENTITY_HEADS['groove_id']]]
    losses.append(jnp.mean(
        (jax.nn.sigmoid(entity_preds['groove'])
         - groove_row.astype(jnp.float32) / _GROOVE_CONT_MAX) ** 2
    ))

    return jnp.mean(jnp.array(losses))


# ---------------------------------------------------------------------------
# Conditional entity loss helpers (lax.scan + lax.cond)
#
# These functions compute groove and trace sub-entity losses conditionally:
# under lax.scan, lax.cond is a genuine runtime conditional — only the active
# branch executes. This avoids materializing O(N_TABLE_SLOTS × N_GROOVE_SLOTS)
# predictions for every sequence position.
# ---------------------------------------------------------------------------

def _cond_groove_scan(groove_decoder, table_h, table_row, banks):
    """
    Scan over N_GROOVE_SLOTS. For each slot, if groove_id != 0 (non-null),
    predict one groove and compute MSE loss vs the bank target.
    Returns scalar = sum of active groove losses.
    """
    groove_ids = table_row[_GROOVE_FX_COLS_ARRAY]  # (N_GROOVE_SLOTS,)

    def step(carry, xs):
        slot_idx, groove_id = xs

        def active(args):
            slot_idx, groove_id = args
            logits    = groove_decoder(table_h, slot_idx)
            groove_row = banks.grooves[groove_id]
            tgt = groove_row.astype(jnp.float32) / _GROOVE_CONT_MAX
            return jnp.mean((jax.nn.sigmoid(logits) - tgt) ** 2)

        def null(_):
            return jnp.float32(0.0)

        return carry, jax.lax.cond(groove_id != 0, active, null, (slot_idx, groove_id))

    _, losses = jax.lax.scan(step, None, (jnp.arange(N_GROOVE_SLOTS, dtype=jnp.int32), groove_ids))
    return jnp.sum(losses)


def _cond_trace_scan(table_decoder, groove_decoder, table_h, table_row, banks):
    """
    Scan over N_TABLE_SLOTS. For each slot, if trace_id != 0 (non-null):
      - compute trace scalar cat + cont loss (using table_decoder weights)
      - compute trace groove losses via nested _cond_groove_scan
    A/H commands are masked to -inf for trace cat logits.
    Returns scalar = sum of active trace losses.
    """
    trace_ids = table_row[_TABLE_FX_COLS_ARRAY]  # (N_TABLE_SLOTS,)

    def step(carry, xs):
        slot_idx, trace_id = xs

        def active(args):
            slot_idx, trace_id = args
            trace_row = banks.traces[trace_id]
            trace_ctx = table_h + table_decoder.slot_embeds[slot_idx]
            trace_h   = jax.nn.gelu(table_decoder.linear_in(trace_ctx))

            cat_logits = jnp.where(_TABLE_CAT_TRACE_MASK, -jnp.inf, table_decoder.cat_out(trace_h))
            cat_loss   = _ce_loss_grouped(cat_logits, trace_row, _TABLE_SCALAR_CAT_GROUPS)
            cont_loss  = _mse_loss(table_decoder.cont_out(trace_h), trace_row,
                                   _TABLE_SCALAR_CONT_COLS_ARRAY, _TABLE_SCALAR_CONT_MAX_VALUES)
            groove_loss = _cond_groove_scan(groove_decoder, trace_h, trace_row, banks)
            return cat_loss + cont_loss + groove_loss

        def null(_):
            return jnp.float32(0.0)

        return carry, jax.lax.cond(trace_id != 0, active, null, (slot_idx, trace_id))

    _, losses = jax.lax.scan(step, None, (jnp.arange(N_TABLE_SLOTS, dtype=jnp.int32), trace_ids))
    return jnp.sum(losses)


def cond_entity_scan_loss(heads, hiddens, target_tokens, banks):
    """
    Conditional groove and trace entity losses for a full sequence.

    Scans over L*4 positions. For each position, conditionally predicts
    groove and trace content only for active (non-null) entity slots.

    heads:         OutputHeads
    hiddens:       (L, 4, d_model) backbone representations
    target_tokens: (L, 4, 21) target song tokens (float or int)
    banks:         SongBanks for the current song
    Returns: scalar loss (sum over all positions and active slots)
    """
    L = hiddens.shape[0]
    hiddens_flat = hiddens.reshape(L * 4, hiddens.shape[-1])
    targets_flat = jnp.int32(target_tokens).reshape(L * 4, 21)

    def scan_step(carry, xs):
        h_bb, target = xs  # (d_model,), (21,)

        # ── Instrument's table grooves + traces ──────────────────────────────
        instr_id  = target[ENTITY_HEADS['instr_id']]
        instr_row = banks.instruments[instr_id]
        it_row    = banks.tables[instr_row[_INSTR_TABLE_COL]]

        instr_h = jax.nn.gelu(heads.instr_decoder.linear_in(h_bb))
        it_h    = jax.nn.gelu(heads.table_decoder.linear_in(instr_h))

        it_groove = _cond_groove_scan(heads.groove_decoder, it_h, it_row, banks)
        it_trace  = _cond_trace_scan(heads.table_decoder, heads.groove_decoder, it_h, it_row, banks)

        # ── Phrase-level table grooves + traces ──────────────────────────────
        pt_id  = target[ENTITY_HEADS['table_id']]
        pt_row = banks.tables[pt_id]

        pt_ctx = jax.nn.gelu(heads.table_proj(h_bb))
        pt_h   = jax.nn.gelu(heads.table_decoder.linear_in(pt_ctx))

        pt_groove = _cond_groove_scan(heads.groove_decoder, pt_h, pt_row, banks)
        pt_trace  = _cond_trace_scan(heads.table_decoder, heads.groove_decoder, pt_h, pt_row, banks)

        total = it_groove + it_trace + pt_groove + pt_trace
        return carry, total

    _, per_pos = jax.lax.scan(scan_step, None, (hiddens_flat, targets_flat))
    return jnp.sum(per_pos)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

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
