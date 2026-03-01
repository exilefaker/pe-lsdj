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

def _groove_loss_vmap(groove_decoder, table_h, table_row, groove_rows):
    """
    Predict losses for all N_GROOVE_SLOTS in parallel via vmap.
    Null slots (groove_id == 0) contribute 0 via jnp.where masking.

    Replaces the previous lax.scan+lax.cond approach: nested scans caused XLA
    to track per-step AD state across all three nesting levels simultaneously,
    producing enormous intermediate buffers (~818 GB for reverse-mode AD).
    vmap computes all slots in one parallel batch — no sequential dependency chain.

    groove_rows: (N_GROOVE_SLOTS, GROOVE_CONT_N) — pre-fetched bank rows.
    Returns scalar = sum of active groove losses.
    """
    groove_ids = table_row[_GROOVE_FX_COLS_ARRAY]  # (N_GROOVE_SLOTS,)

    def groove_step(slot_idx, groove_id, groove_row):
        logits = groove_decoder(table_h, slot_idx)
        tgt = groove_row.astype(jnp.float32) / _GROOVE_CONT_MAX
        loss = jnp.mean((jax.nn.sigmoid(logits) - tgt) ** 2)
        return jnp.where(groove_id != 0, loss, jnp.float32(0.0))

    losses = jax.vmap(groove_step)(
        jnp.arange(N_GROOVE_SLOTS, dtype=jnp.int32), groove_ids, groove_rows,
    )
    return jnp.sum(losses)


def conditional_entity_loss(heads, hiddens, target_tokens, banks):
    """
    Conditional groove and trace entity losses for a full sequence.

    Memory-efficient design (of necessity):
      - All bank lookups pre-fetched into dense arrays before any vmap.
      - Context vectors (it_h, pt_h) computed once for all T=L*4 positions.
      - Direct groove losses: vmap over T×N_GROOVE_SLOTS (small; masked nulls ok).
      - Trace losses: jnp.nonzero gathers only non-null slots before vmap, so
        memory is O(active_traces) not O(T × N_TABLE_SLOTS).

    MAX_ACTIVE_TRACES caps gather size. It must be >= the true number of
    non-null trace IDs in the sequence; silent truncation causes incorrect
    gradients. Default T = L*4 handles ~1 active trace slot per position on
    average — generous for typical LSDJ tracks.

    heads:         OutputHeads
    hiddens:       (L, 4, d_model) backbone representations
    target_tokens: (L, 4, 21) target song tokens (float or int)
    banks:         SongBanks for the current song
    Returns: scalar loss (sum over all positions and active slots)
    """
    L = hiddens.shape[0]
    T = L * 4
    hiddens_flat = hiddens.reshape(T, hiddens.shape[-1])
    targets_flat = jnp.int32(target_tokens).reshape(T, 21)

    # ── Pre-fetch ALL bank data ────────────────────────────────────────────
    instr_ids     = targets_flat[:, ENTITY_HEADS['instr_id']]
    instr_rows    = banks.instruments[instr_ids]                                # (T, INSTR_WIDTH)
    it_table_ids  = instr_rows[:, _INSTR_TABLE_COL].astype(jnp.int32)
    it_table_rows = banks.tables[it_table_ids]                                  # (T, table_bank_w)

    pt_ids        = targets_flat[:, ENTITY_HEADS['table_id']]
    pt_table_rows = banks.tables[pt_ids]                                        # (T, table_bank_w)

    it_groove_ids  = it_table_rows[:, _GROOVE_FX_COLS_ARRAY].astype(jnp.int32) # (T, N_GROOVE_SLOTS)
    it_groove_rows = banks.grooves[it_groove_ids]                               # (T, N_GROOVE_SLOTS, GROOVE_CONT_N)
    pt_groove_ids  = pt_table_rows[:, _GROOVE_FX_COLS_ARRAY].astype(jnp.int32)
    pt_groove_rows = banks.grooves[pt_groove_ids]

    it_trace_ids  = it_table_rows[:, _TABLE_FX_COLS_ARRAY].astype(jnp.int32)   # (T, N_TABLE_SLOTS)
    it_trace_rows = banks.traces[it_trace_ids]                                  # (T, N_TABLE_SLOTS, table_bank_w)
    pt_trace_ids  = pt_table_rows[:, _TABLE_FX_COLS_ARRAY].astype(jnp.int32)
    pt_trace_rows = banks.traces[pt_trace_ids]

    it_tgr_ids  = it_trace_rows[:, :, _GROOVE_FX_COLS_ARRAY].astype(jnp.int32) # (T, N_TABLE_SLOTS, N_GROOVE_SLOTS)
    it_tgr_rows = banks.grooves[it_tgr_ids]                                     # (T, N_TABLE_SLOTS, N_GROOVE_SLOTS, GROOVE_CONT_N)
    pt_tgr_ids  = pt_trace_rows[:, :, _GROOVE_FX_COLS_ARRAY].astype(jnp.int32)
    pt_tgr_rows = banks.grooves[pt_tgr_ids]

    # ── Context vectors for all T positions ───────────────────────────────
    instr_h_all = jax.vmap(lambda h: jax.nn.gelu(heads.instr_decoder.linear_in(h)))(hiddens_flat)
    it_h_all    = jax.vmap(lambda h: jax.nn.gelu(heads.table_decoder.linear_in(h)))(instr_h_all)
    pt_ctx_all  = jax.vmap(lambda h: jax.nn.gelu(heads.table_proj(h)))(hiddens_flat)
    pt_h_all    = jax.vmap(lambda h: jax.nn.gelu(heads.table_decoder.linear_in(h)))(pt_ctx_all)

    # ── Direct groove losses: vmap over T positions, 32 slots each ────────
    # T × 32 = 32768 ops; null slots masked; cheap enough without gather.
    it_groove_total = jnp.sum(jax.vmap(
        lambda h, row, grs: _groove_loss_vmap(heads.groove_decoder, h, row, grs)
    )(it_h_all, it_table_rows, it_groove_rows))

    pt_groove_total = jnp.sum(jax.vmap(
        lambda h, row, grs: _groove_loss_vmap(heads.groove_decoder, h, row, grs)
    )(pt_h_all, pt_table_rows, pt_groove_rows))

    # ── Trace losses: gather only non-null slots, then vmap ───────────────
    MAX_ACTIVE_TRACES = T  # per gather (it and pt handled separately)

    def _trace_gather_loss(h_all, trace_ids_2d, trace_rows_3d, tgr_rows_4d):
        """Gather active trace slots and compute their losses in one vmap."""
        flat_ids    = trace_ids_2d.reshape(-1)                         # (T*N_TABLE_SLOTS,)
        active_idx, = jnp.nonzero(flat_ids, size=MAX_ACTIVE_TRACES, fill_value=0)
        valid       = jnp.arange(MAX_ACTIVE_TRACES) < jnp.sum(flat_ids != 0)

        pos_idx   = active_idx // N_TABLE_SLOTS
        slot_idx  = active_idx  % N_TABLE_SLOTS
        active_h  = h_all[pos_idx]                                     # (MAX, entity_dim)
        active_tr = trace_rows_3d.reshape(-1, trace_rows_3d.shape[-1])[active_idx]  # (MAX, trace_bank_w)
        active_tgr = tgr_rows_4d.reshape(-1, N_GROOVE_SLOTS, GROOVE_CONT_N)[active_idx]  # (MAX, 32, 32)

        def one_trace(h, sidx, tr_row, tgr_rows, valid):
            trace_ctx  = h + heads.table_decoder.slot_embeds[sidx]
            trace_h    = jax.nn.gelu(heads.table_decoder.linear_in(trace_ctx))
            cat_logits = jnp.where(_TABLE_CAT_TRACE_MASK, -jnp.inf,
                                   heads.table_decoder.cat_out(trace_h))
            cat_loss   = _ce_loss_grouped(cat_logits, tr_row, _TABLE_SCALAR_CAT_GROUPS)
            cont_loss  = _mse_loss(heads.table_decoder.cont_out(trace_h), tr_row,
                                   _TABLE_SCALAR_CONT_COLS_ARRAY, _TABLE_SCALAR_CONT_MAX_VALUES)
            groove_loss = _groove_loss_vmap(heads.groove_decoder, trace_h, tr_row, tgr_rows)
            return jnp.where(valid, cat_loss + cont_loss + groove_loss, jnp.float32(0.0))

        return jnp.sum(jax.vmap(one_trace)(active_h, slot_idx, active_tr, active_tgr, valid))

    it_trace_total = _trace_gather_loss(it_h_all, it_trace_ids, it_trace_rows, it_tgr_rows)
    pt_trace_total = _trace_gather_loss(pt_h_all, pt_trace_ids, pt_trace_rows, pt_tgr_rows)

    return it_groove_total + pt_groove_total + it_trace_total + pt_trace_total


def entity_alignment_loss(model, hiddens, target_tokens, banks):
    """
    Contrastive alignment losses for entity embedding/decoder pairs.

    Pulls each decoder's intermediate latent (pre-GELU) toward the corresponding
    entity embedder's output using 1 - cosine_similarity, ensuring both sides
    live in the same metric space for cosine-similarity bank matching at generation.

    Pairs covered:
      - Instrument:        instr_decoder.encode(h) vs instrument_embedder(bank_row)
      - Table (instr's):   table_decoder.encode(instr_h) vs table_embedder(it_table_row)
      - Table (phrase):    table_decoder.encode(pt_ctx) vs table_embedder(pt_table_row)
      - Softsynth:         softsynth_decoder.encode(instr_h) vs softsynth_embedder(synth_row)
      - Groove (phrase):   groove_decoder.encode(groove_ctx, N_GROOVE_SLOTS)
                             vs groove_embedder(groove_id)

    Groove is covered only at the phrase level (groove_id directly in song tokens).
    Table-slot groove alignment would require scanning groove references inside
    table bank rows and is not implemented here.

    Requires: instr_dim == table_dim == softsynth_dim == entity_dim
    (enforced by assertions in LSDJTransformer.__init__).
    Null entities (ID == 0) contribute zero.
    Returns a sum to be normalised alongside other loss terms in sequence_loss.
    """
    L = hiddens.shape[0]
    T = L * 4
    h_flat = hiddens.reshape(T, hiddens.shape[-1])           # (T, d_model)
    t_flat = jnp.int32(target_tokens).reshape(T, 21)         # (T, 21)

    instr_ids        = t_flat[:, ENTITY_HEADS['instr_id']]   # (T,)
    pt_ids           = t_flat[:, ENTITY_HEADS['table_id']]   # (T,)
    instr_rows       = banks.instruments[instr_ids]           # (T, INSTR_WIDTH)
    it_table_ids     = instr_rows[:, _INSTR_TABLE_COL].astype(jnp.int32)      # (T,)
    it_softsynth_ids = instr_rows[:, _INSTR_SOFTSYNTH_COL].astype(jnp.int32)  # (T,)

    heads    = model.output_heads
    step_emb = model.embedder.step_embedder

    # Decoder-side latents
    instr_q    = jax.vmap(heads.instr_decoder.encode)(h_flat)                           # (T, entity_dim) pre-GELU
    instr_gelu = jax.nn.gelu(instr_q)                                                   # (T, entity_dim) post-GELU, used as parent context
    it_table_q = jax.vmap(heads.table_decoder.encode)(instr_gelu)                       # (T, entity_dim)
    pt_ctx     = jax.nn.gelu(jax.vmap(heads.table_proj)(h_flat))                        # (T, entity_dim)
    pt_table_q = jax.vmap(heads.table_decoder.encode)(pt_ctx)                           # (T, entity_dim)
    synth_q    = jax.vmap(heads.instr_decoder.softsynth_decoder.encode)(instr_gelu)     # (T, entity_dim)
    groove_ctx = jax.nn.gelu(jax.vmap(heads.phrase_groove_proj)(h_flat))                # (T, entity_dim)
    groove_q   = jax.vmap(lambda ctx: heads.groove_decoder.encode(ctx, N_GROOVE_SLOTS))(groove_ctx)  # (T, entity_dim)

    # Embedder-side keys
    instr_emb   = step_emb.instrument_embedder
    table_emb   = step_emb.fx_embedder.embedders['value'].embedders['table_fx']
    synth_emb   = step_emb.instrument_embedder.embedder.embedders['softsynth']
    groove_emb  = step_emb.fx_embedder.embedders['value'].embedders['groove']

    groove_ids = t_flat[:, ENTITY_HEADS['groove_id']]             # (T,)

    instr_k    = jax.vmap(instr_emb)(instr_ids)         # (T, entity_dim)
    it_table_k = jax.vmap(table_emb)(it_table_ids)      # (T, entity_dim)
    pt_table_k = jax.vmap(table_emb)(pt_ids)            # (T, entity_dim)
    synth_k    = jax.vmap(synth_emb)(it_softsynth_ids)  # (T, entity_dim)
    groove_k   = jax.vmap(groove_emb)(groove_ids)        # (T, entity_dim)

    def _cos_loss(a, b):
        sim = jnp.dot(a, b) / (jnp.linalg.norm(a) + 1e-8) / (jnp.linalg.norm(b) + 1e-8)
        return jnp.float32(1.0) - sim

    def _pair_loss(queries, keys, valid_mask):
        losses = jax.vmap(_cos_loss)(queries, keys)
        return jnp.sum(jnp.where(valid_mask, losses, jnp.float32(0.0)))

    return (
        _pair_loss(instr_q,    instr_k,    instr_ids        != 0) +
        _pair_loss(it_table_q, it_table_k, it_table_ids     != 0) +
        _pair_loss(pt_table_q, pt_table_k, pt_ids           != 0) +
        _pair_loss(synth_q,    synth_k,    it_softsynth_ids != 0) +
        _pair_loss(groove_q,   groove_k,   groove_ids        != 0)
    )


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
        step = self.embedder.step_embedder
        instr_emb_dim = step.instrument_embedder.out_dim
        table_emb_dim = step.fx_embedder.embedders['value'].embedders['table_fx'].out_dim
        synth_emb_dim = step.instrument_embedder.embedder.embedders['softsynth'].out_dim
        assert instr_emb_dim == entity_dim, (
            f"instr_dim ({instr_emb_dim}) must equal entity_dim ({entity_dim}) "
            f"for entity_alignment_loss. Pass instr_dim={entity_dim} explicitly."
        )
        assert table_emb_dim == entity_dim, (
            f"table_dim ({table_emb_dim}) must equal entity_dim ({entity_dim}) "
            f"for entity_alignment_loss. Pass table_dim={entity_dim} explicitly."
        )
        assert synth_emb_dim == entity_dim, (
            f"softsynth_dim ({synth_emb_dim}) must equal entity_dim ({entity_dim}) "
            f"for entity_alignment_loss. Pass softsynth_dim={entity_dim} explicitly."
        )

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
