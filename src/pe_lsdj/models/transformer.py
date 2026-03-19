import json
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from pe_lsdj.embedding.song import SequenceEmbedder, SongBanks
from pe_lsdj.constants import SOFTSYNTH_WIDTH
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
    N_TABLE_SLOTS, N_GROOVE_SLOTS, INSTR_TABLE_COL, INSTR_SOFTSYNTH_COL,
    # Private arrays needed by loss functions
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


def token_loss(logits_dict, target_dists, label_smoothing=0.0):
    total = 0.0
    for name, (_, vocab) in TOKEN_HEADS.items():
        log_probs = jax.nn.log_softmax(logits_dict[name])
        targets   = target_dists[name]
        if label_smoothing > 0.0:
            targets = (1.0 - label_smoothing) * targets + label_smoothing / vocab
        total += -jnp.sum(targets * log_probs)
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


def _mse_loss(raw, row, cont_cols_array, max_vals_array):
    """MSE loss for continuous fields (single bank row).
    raw: (n,) — sigmoid-space logits.
    """
    pred    = jax.nn.sigmoid(raw)
    targets = row[cont_cols_array].astype(jnp.float32) / max_vals_array
    return jnp.mean((pred - targets) ** 2)


def table_loss(prediction, target):
    """
    Scalar loss components for one table (or trace) prediction.
    Returns [cat_loss, cont_loss] — 2 components.
    Groove-slot and trace sub-entity losses are computed separately by
    cond_entity_scan_loss using lax.scan + lax.cond.
    """

    return [
        _ce_loss_grouped(prediction['cat'], target, _TABLE_SCALAR_CAT_GROUPS),
        _mse_loss(prediction['cont'], target,
                  _TABLE_SCALAR_CONT_COLS_ARRAY, _TABLE_SCALAR_CONT_MAX_VALUES),
    ]


def softsynth_loss(prediction, target):
    synth_target, wf_target = (
        target[:SOFTSYNTH_WIDTH], 
        target[SOFTSYNTH_WIDTH:]
    )
    synth_losses = []

    synth_losses.append(_ce_loss_grouped(
        prediction['cat'], 
        synth_target, 
        _SOFTSYNTH_CAT_GROUPS
    ))
    synth_losses.append(_mse_loss(
        prediction['cont'],
        synth_target,
        _SOFTSYNTH_CONT_COLS_ARRAY,
        _SOFTSYNTH_CONT_MAX_VALUES,
    ))

    # Waveframes
    wf_pred = jax.nn.sigmoid(prediction['waveframes'])
    wf_tgt  = wf_target.astype(jnp.float32) / 15.0
    synth_losses.append(jnp.mean((wf_pred - wf_tgt) ** 2))

    return synth_losses


def instr_scalar_loss(prediction, target):
    """
    Loss for scalar components of instrument 
    (table and softsynth losses computed separately)
    """
    return [
        _ce_loss_grouped(
            prediction['cat'], 
            target, 
            _INSTR_SCALAR_CAT_GROUPS,
        ),
        _mse_loss(
            prediction['cont'],
            target,
            _INSTR_SCALAR_CONT_COLS_ARRAY,
            _INSTR_SCALAR_CONT_MAX_VALUES,
        )
    ]


def instr_loss(instr_prediction, instr_target, banks):
    instr_losses = []
    instr_losses.extend(
        instr_scalar_loss(instr_prediction, instr_target)
    )
    # Instrument's table
    table_id = instr_target[INSTR_TABLE_COL]
    instr_table_target = banks.tables[table_id]
    instr_losses.extend(
        table_loss(instr_prediction['table'], instr_table_target)
    )
    # Instrument's softsynth
    synth_id = instr_target[INSTR_SOFTSYNTH_COL]
    synth_wave_target = banks.synth_waves[synth_id]
    instr_losses.extend(
        softsynth_loss(instr_prediction['softsynth'], synth_wave_target)
    )

    return instr_losses


def groove_loss(prediction, target):
    """MSE loss for one groove row. prediction: (GROOVE_CONT_N,) sigmoid-space logits."""
    pred = jax.nn.sigmoid(prediction)
    tgt  = target.astype(jnp.float32) / _GROOVE_CONT_MAX
    return jnp.mean((pred - tgt) ** 2)


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
    instr_target = banks.instruments[instr_id]
    losses.extend(
        instr_loss(entity_preds['instr'], instr_target, banks)
    )

    # ─── Phrase-level table ──────────────────────────────────────────────────
    table_id = target_tokens[ENTITY_HEADS['table_id']]
    phrase_table = banks.tables[table_id]
    losses.extend(table_loss(entity_preds['table'], phrase_table))

    # ─── Phrase-level groove ─────────────────────────────────────────────────
    groove_id = target_tokens[ENTITY_HEADS['groove_id']]
    phrase_groove = banks.grooves[groove_id]
    losses.append(groove_loss(entity_preds['groove'], phrase_groove))

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

    groove_rows: (N_GROOVE_SLOTS, GROOVE_CONT_N) — pre-fetched bank rows.
    Returns scalar = sum of active groove losses.
    """
    groove_ids = table_row[_GROOVE_FX_COLS_ARRAY]  # (N_GROOVE_SLOTS,)

    def groove_step(slot_idx, groove_id, groove_row):
        logits = groove_decoder(table_h, slot_idx)
        loss = groove_loss(logits, groove_row)
        return jnp.where(groove_id != 0, loss, jnp.float32(0.0))

    losses = jax.vmap(groove_step)(
        jnp.arange(N_GROOVE_SLOTS, dtype=jnp.int32), groove_ids, groove_rows,
    )
    return jnp.sum(losses)


def score_one_trace(heads, h, sidx, tr_row, tgr_rows):
    """
    Score one bank trace row against the model's prediction for slot sidx.

    h:        table-level context vector (entity_dim,)
    sidx:     slot index scalar
    tr_row:   bank trace row (TABLE_WIDTH,)
    tgr_rows: groove rows for this trace (N_GROOVE_SLOTS, GROOVE_CONT_N)
    Returns:  scalar score
    """
    trace_ctx  = h + heads.table_decoder.slot_embeds[sidx]
    trace_h    = jax.nn.gelu(heads.table_decoder.linear_in(trace_ctx))
    cat_logits = jnp.where(_TABLE_CAT_TRACE_MASK, -jnp.inf,
                           heads.table_decoder.cat_out(trace_h))
    cat_s    = _ce_loss_grouped(cat_logits, tr_row, _TABLE_SCALAR_CAT_GROUPS)
    cont_s   = _mse_loss(heads.table_decoder.cont_out(trace_h), tr_row,
                        _TABLE_SCALAR_CONT_COLS_ARRAY, _TABLE_SCALAR_CONT_MAX_VALUES)
    groove_s = _groove_loss_vmap(heads.groove_decoder, trace_h, tr_row, tgr_rows)
    return cat_s + cont_s + groove_s


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
    it_table_ids  = instr_rows[:, INSTR_TABLE_COL].astype(jnp.int32)
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
    it_h_all    = jax.vmap(lambda h: jax.nn.gelu(heads.instr_to_table_proj(h)))(instr_h_all)
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
    # NOTE: Unused currently to simplify arch. and avoid OOM issues.
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

    Each pair operates in its own entity dim:
      - Instrument:   instr_entity_dim
      - Table pairs:  table_entity_dim (instrument's table bridged via instr_to_table_proj)
      - Softsynth:    softsynth_entity_dim
      - Groove:       table_entity_dim
    Dims enforced by assertions in LSDJTransformer.__init__.
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
    it_table_ids     = instr_rows[:, INSTR_TABLE_COL].astype(jnp.int32)      # (T,)
    it_softsynth_ids = instr_rows[:, INSTR_SOFTSYNTH_COL].astype(jnp.int32)  # (T,)

    heads    = model.output_heads
    step_emb = model.embedder.step_embedder

    # Decoder-side latents
    instr_q      = jax.vmap(heads.instr_decoder.encode)(h_flat)                                        # (T, instr_entity_dim) pre-GELU
    instr_gelu   = jax.nn.gelu(instr_q)                                                               # (T, instr_entity_dim) post-GELU
    it_table_ctx = jax.nn.gelu(jax.vmap(heads.instr_to_table_proj)(instr_gelu))                       # (T, table_entity_dim) bridge
    it_table_q   = jax.vmap(heads.table_decoder.encode)(it_table_ctx)                                 # (T, table_entity_dim)
    pt_ctx       = jax.nn.gelu(jax.vmap(heads.table_proj)(h_flat))                                    # (T, table_entity_dim)
    pt_table_q   = jax.vmap(heads.table_decoder.encode)(pt_ctx)                                       # (T, table_entity_dim)
    synth_q      = jax.vmap(heads.instr_decoder.softsynth_decoder.encode)(instr_gelu)                 # (T, softsynth_entity_dim)
    groove_ctx   = jax.nn.gelu(jax.vmap(heads.phrase_groove_proj)(h_flat))                            # (T, table_entity_dim)
    groove_q     = jax.vmap(lambda ctx: heads.groove_decoder.encode(ctx, N_GROOVE_SLOTS))(groove_ctx) # (T, table_entity_dim)

    # Embedder-side keys
    instr_emb  = step_emb.instrument_embedder
    table_emb  = step_emb.fx_embedder.embedders['value'].embedders['table_fx']
    synth_emb  = step_emb.instrument_embedder.embedder.embedders['synth_wave']
    groove_emb = step_emb.fx_embedder.embedders['value'].embedders['groove']

    groove_ids = t_flat[:, ENTITY_HEADS['groove_id']]             # (T,)

    instr_k    = jax.vmap(lambda x: instr_emb(x, banks))(instr_ids)         # (T, entity_dim)
    it_table_k = jax.vmap(lambda x: table_emb(x, banks))(it_table_ids)      # (T, entity_dim)
    pt_table_k = jax.vmap(lambda x: table_emb(x, banks))(pt_ids)            # (T, entity_dim)
    synth_k    = jax.vmap(lambda x: synth_emb(x, banks))(it_softsynth_ids)  # (T, entity_dim)
    groove_k   = jax.vmap(lambda x: groove_emb(x, banks))(groove_ids)       # (T, entity_dim)

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


def apply_rope(x: Array, positions: Array) -> Array:
    """
    Apply Rotary Position Embeddings to a Q or K tensor.

    Rotates each (head_dim/2)-pair of dimensions by m*theta_i, where m is the
    absolute token position and theta_i is a geometric frequency. The dot product
    Q·K then depends only on the relative distance (m-n), not absolute positions.
    This means cached K values remain valid as the sliding window advances.

    x         : (S, H, head_dim) — head_dim must be even
    positions : (S,) integer position indices (absolute, 0-based)
    Returns   : (S, H, head_dim) with each head_dim/2 pair rotated by m*theta_i
    """
    S, H, head_dim = x.shape
    half = head_dim // 2
    inv_freq = 10000.0 ** (-jnp.arange(half, dtype=jnp.float32) * 2 / head_dim)
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]  # (S, half)
    cos_a = jnp.cos(angles)[:, None, :]   # (S, 1, half) — broadcast over H
    sin_a = jnp.sin(angles)[:, None, :]
    x_even = x[:, :, 0::2]                # (S, H, half)
    x_odd  = x[:, :, 1::2]
    out_even = x_even * cos_a - x_odd * sin_a
    out_odd  = x_even * sin_a + x_odd * cos_a
    # stack + reshape interleaves: [e0, o0, e1, o1, ...]
    return jnp.stack([out_even, out_odd], axis=-1).reshape(S, H, head_dim)


def _rope_full_attention(attn, x_ch, positions):
    """
    Full-sequence multi-head attention with RoPE applied to Q and K, for one channel.

    RoPE is applied after projecting Q and K, so the dot product Q·K encodes only
    relative distance. V is not rotated. Used in both encode and prefill.

    Uses jax.nn.dot_product_attention with is_causal=True for memory-efficient
    Flash Attention — no O(S²) score matrix is materialised.

    attn      : eqx.nn.MultiheadAttention
    x_ch      : (S, d_model) — pre-norm input for this channel
    positions : (S,) integer absolute positions
    Returns   : (S, d_model)
    """
    S, d = x_ch.shape
    H = attn.num_heads
    head_dim = d // H

    Q = jax.vmap(attn.query_proj)(x_ch).reshape(S, H, head_dim)
    K = jax.vmap(attn.key_proj)(x_ch).reshape(S, H, head_dim)
    V = jax.vmap(attn.value_proj)(x_ch).reshape(S, H, head_dim)

    Q = apply_rope(Q, positions)
    K = apply_rope(K, positions)   # V is NOT rotated

    # (S, H, head_dim) — Flash Attention handles causal masking internally
    out = jax.nn.dot_product_attention(Q, K, V, is_causal=True)
    out = jax.vmap(attn.output_proj)(out.reshape(S, d))
    return out                                               # (S, d_model)


def _rope_cached_attention(attn, q_ch, k_cache_ch, v_cache_ch, new_pos):
    """
    Single-query attention against a post-RoPE K cache, for one channel.

    The cache stores K values that were already rotated by their absolute positions
    at prefill time. Here we rotate Q by new_pos and attend against the cached K —
    the Q·K dot product automatically encodes the relative distance, regardless of
    where old tokens sit in the sliding window.

    attn       : eqx.nn.MultiheadAttention
    q_ch       : (1, d_model)  — new token input (will be Q-projected + RoPE'd)
    k_cache_ch : (W, d_model)  — cached POST-ROPE projected keys
    v_cache_ch : (W, d_model)  — cached projected values (NOT RoPE'd)
    new_pos    : scalar int    — absolute position of the new token
    Returns    : (1, d_model)
    """
    d = q_ch.shape[-1]
    H = attn.num_heads
    head_dim = d // H
    W = k_cache_ch.shape[0]

    q = attn.query_proj(q_ch[0]).reshape(1, H, head_dim)
    q = apply_rope(q, jnp.asarray(new_pos)[None])[0]      # (H, head_dim) — RoPE at new_pos

    k = k_cache_ch.reshape(W, H, head_dim)                 # post-RoPE cached K
    v = v_cache_ch.reshape(W, H, head_dim)

    scale   = head_dim ** -0.5
    scores  = jnp.einsum('hd,whd->hw', q, k) * scale      # (H, W)
    weights = jax.nn.softmax(scores, axis=-1)              # (H, W)
    out     = jnp.einsum('hw,whd->hd', weights, v)        # (H, head_dim)
    out     = attn.output_proj(out.reshape(d))
    return out[None]                                       # (1, d_model)


class AxialTransformerBlock(eqx.Module):
    temporal_attn: eqx.nn.MultiheadAttention
    channel_attn:  eqx.nn.MultiheadAttention
    mlp:           eqx.nn.MLP
    norm_t:        eqx.nn.LayerNorm
    norm_c:        eqx.nn.LayerNorm
    norm_mlp:      eqx.nn.LayerNorm
    dropout:       eqx.nn.Dropout

    def __init__(self, d_model, num_heads_t, num_heads_c, key, dropout_p=0.0):
        k1, k2, k3 = jr.split(key, 3)
        self.temporal_attn = eqx.nn.MultiheadAttention(num_heads_t, d_model, key=k1)
        self.channel_attn  = eqx.nn.MultiheadAttention(num_heads_c, d_model, key=k2)
        self.mlp      = eqx.nn.MLP(d_model, d_model, d_model * 4, depth=1, key=k3)
        self.norm_t   = eqx.nn.LayerNorm(d_model)
        self.norm_c   = eqx.nn.LayerNorm(d_model)
        self.norm_mlp = eqx.nn.LayerNorm(d_model)
        self.dropout  = eqx.nn.Dropout(p=dropout_p)

    def __call__(self, x: Array, positions: Array, key: Key | None = None) -> Array:
        inference = key is None
        k1, k2, k3 = jr.split(key, 3) if key is not None else (None, None, None)

        normed = _norm2d(self.norm_c, x)
        x = x + self.dropout(
            jax.vmap(lambda x_t: self.channel_attn(x_t, x_t, x_t))(normed),
            key=k1, inference=inference,
        )
        normed = _norm2d(self.norm_t, x)
        x = x + self.dropout(
            jax.vmap(
                lambda x_ch: _rope_full_attention(self.temporal_attn, x_ch, positions),
                in_axes=1, out_axes=1,
            )(normed),
            key=k2, inference=inference,
        )
        normed = _norm2d(self.norm_mlp, x)
        x = x + self.dropout(
            jax.vmap(jax.vmap(self.mlp))(normed),
            key=k3, inference=inference,
        )
        return x

    def build_layer_cache(self, x: Array, positions: Array) -> tuple[Array, Array, Array]:
        """
        Full forward pass for this block (inference only — no dropout), also
        returning the KV cache tensors for the temporal attention layer.

        K is stored POST-ROPE: each key vector has already been rotated by its
        absolute position. This is essential for correctness — when the sliding
        window advances, the cached K values remain valid because Q·K depends
        only on relative distance (new_pos - cached_pos), not absolute positions.
        V is stored without rotation.

        x         : (W, 4, d_model) — input to this block
        positions : (W,) integer absolute positions
        Returns   : (x_out, K, V) where K and V are (W, 4, d_model).
        """
        W = x.shape[0]
        d_model = x.shape[-1]
        H = self.temporal_attn.num_heads
        head_dim = d_model // H

        # Channel attention (across 4 channels at each timestep)
        normed = _norm2d(self.norm_c, x)
        x = x + jax.vmap(lambda x_t: self.channel_attn(x_t, x_t, x_t))(normed)

        # Post-RoPE K for caching; V without RoPE
        normed = _norm2d(self.norm_t, x)
        def _proj_and_rope_k(x_ch):
            K = jax.vmap(self.temporal_attn.key_proj)(x_ch).reshape(W, H, head_dim)
            return apply_rope(K, positions).reshape(W, d_model)
        K = jax.vmap(_proj_and_rope_k, in_axes=1, out_axes=1)(normed)  # (W, 4, d_model) post-RoPE
        V = jax.vmap(jax.vmap(self.temporal_attn.value_proj))(normed)  # (W, 4, d_model) no RoPE

        x = x + jax.vmap(
            lambda x_ch: _rope_full_attention(self.temporal_attn, x_ch, positions),
            in_axes=1, out_axes=1,
        )(normed)

        # MLP
        normed = _norm2d(self.norm_mlp, x)
        x = x + jax.vmap(jax.vmap(self.mlp))(normed)
        return x, K, V

    def cached_step(self, x_new: Array, k_cache: Array, v_cache: Array,
                    new_pos) -> tuple[Array, Array, Array]:
        """
        Process a single new token through this block using the KV cache.

        K for the new token is stored POST-ROPE (rotated by new_pos) before being
        appended to the cache, matching the format of the prefilled cache from
        build_layer_cache. Q is rotated by new_pos inside _rope_cached_attention,
        so Q·K encodes only relative distance. V is not rotated.

        x_new   : (1, 4, d_model) — embedded new token
        k_cache : (W, 4, d_model) — cached POST-ROPE keys from all prior positions
        v_cache : (W, 4, d_model) — cached values (no RoPE)
        new_pos : scalar int      — absolute position of the new token
        Returns : (x_new_out, new_k_cache, new_v_cache)
        """
        d_model = x_new.shape[-1]
        H = self.temporal_attn.num_heads
        head_dim = d_model // H

        # Channel attention — only the new timestep attends to itself (4 channels)
        normed = _norm2d(self.norm_c, x_new)
        x_new = x_new + jax.vmap(lambda x_t: self.channel_attn(x_t, x_t, x_t))(normed)

        # Project + RoPE K for new token; project V without RoPE
        normed = _norm2d(self.norm_t, x_new)
        def _new_k_ch(x_ch):  # x_ch: (1, d_model)
            K = jax.vmap(self.temporal_attn.key_proj)(x_ch).reshape(1, H, head_dim)
            return apply_rope(K, jnp.asarray(new_pos)[None]).reshape(1, d_model)
        k_new = jax.vmap(_new_k_ch, in_axes=1, out_axes=1)(normed)        # (1, 4, d_model) post-RoPE
        v_new = jax.vmap(jax.vmap(self.temporal_attn.value_proj))(normed)  # (1, 4, d_model) no RoPE

        k_cache = jnp.concatenate([k_cache[1:], k_new], axis=0)            # (W, 4, d_model)
        v_cache = jnp.concatenate([v_cache[1:], v_new], axis=0)

        # Temporal attention: single query against the full W-length post-RoPE cache
        attn_out = jax.vmap(
            lambda q_ch, k_ch, v_ch: _rope_cached_attention(
                self.temporal_attn, q_ch, k_ch, v_ch, new_pos
            ),
            in_axes=(1, 1, 1), out_axes=1,
        )(normed, k_cache, v_cache)                                         # (1, 4, d_model)
        x_new = x_new + attn_out

        # MLP
        normed = _norm2d(self.norm_mlp, x_new)
        x_new = x_new + jax.vmap(jax.vmap(self.mlp))(normed)
        return x_new, k_cache, v_cache


class LSDJTransformer(eqx.Module):
    embedder:     SequenceEmbedder
    blocks:       list[AxialTransformerBlock]
    final_norm:   eqx.nn.LayerNorm
    output_heads: OutputHeads
    d_model:      int
    noise_sd:     float

    def __init__(
        self,
        key: Key,
        *,
        d_model: int = 256,
        instr_entity_dim: int = 128,
        table_entity_dim: int = 64,
        softsynth_entity_dim: int = 64,
        num_heads_t: int = 4,
        num_heads_c: int = 2,
        num_blocks: int = 6,
        noise_sd: float = 0.0,
        dropout_p: float = 0.0,
        **embedder_kwargs,
    ):
        assert (d_model // num_heads_t) % 2 == 0, (
            f"head_dim = d_model // num_heads_t = {d_model // num_heads_t} must be even for RoPE"
        )
        keys = jr.split(key, num_blocks + 3)
        self.d_model = d_model
        self.noise_sd = noise_sd
        self.embedder = SequenceEmbedder.create(
            keys[0], out_dim=d_model * 4, **embedder_kwargs,
        )
        self.blocks = [
            AxialTransformerBlock(d_model, num_heads_t, num_heads_c, keys[i + 1], dropout_p)
            for i in range(num_blocks)
        ]
        self.final_norm   = eqx.nn.LayerNorm(d_model)
        self.output_heads = OutputHeads(d_model, instr_entity_dim, table_entity_dim, softsynth_entity_dim, keys[-1])

    def encode(self, song_tokens: Array, banks: SongBanks, *,
               positions: Array | None = None, song_length=None,
               key: Key | None = None) -> Array:
        x = self.embedder(song_tokens, banks, positions=positions, song_length=song_length)
        if key is not None:
            # Split one key for noise, one per block for dropout
            keys = jr.split(key, len(self.blocks) + 1)
            noise_key, block_keys = keys[0], keys[1:]
            if self.noise_sd > 0.0:
                scale = jnp.mean(jnp.linalg.norm(x, axis=-1))
                x = x + jr.normal(noise_key, x.shape) * (self.noise_sd * scale)
        else:
            block_keys = [None] * len(self.blocks)
        S = x.shape[0]
        if positions is None:
            positions = jnp.arange(S)
        for block, bkey in zip(self.blocks, block_keys):
            x = block(x, positions, bkey)
        return _norm2d(self.final_norm, x)

    def __call__(self, song_tokens: Array, banks: SongBanks, *, key: Key | None = None):
        return jax.vmap(jax.vmap(self.output_heads))(self.encode(song_tokens, banks, key=key))

    def prefill(self, input_tokens: Array, banks: SongBanks,
               song_length=None) -> tuple[Array, Array, Array]:
        """
        Run a full forward pass on the prompt and return the last hidden state plus
        the KV cache pre-filled for all prompt positions.

        This is the setup step for KV-cached autoregressive generation:
          - last_hidden provides the first prediction (what comes after the prompt)
          - k_cache / v_cache allow subsequent steps to run in O(d_model^2) per step
            rather than O(W * d_model^2) by reusing the projected keys and values
            from all prior positions.

        input_tokens : (W, 4, 21)
        song_length  : total expected song length (prompt + generation steps), used
                       for the progress embedding. Defaults to W if not provided.
        Returns : (last_hidden, k_cache, v_cache)
            last_hidden : (4, d_model)
            k_cache     : (num_blocks, W, 4, d_model)
            v_cache     : (num_blocks, W, 4, d_model)
        """
        x = self.embedder(input_tokens, banks, song_length=song_length)
        W = x.shape[0]
        positions = jnp.arange(W)
        all_K, all_V = [], []
        for block in self.blocks:
            x, K, V = block.build_layer_cache(x, positions)
            all_K.append(K)
            all_V.append(V)
        x = _norm2d(self.final_norm, x)          # (W, 4, d_model)
        last_hidden = x[-1]                       # (4, d_model)
        return last_hidden, jnp.stack(all_K), jnp.stack(all_V)

    def _encode_one_cached(self, x_new: Array, k_cache: Array, v_cache: Array,
                           new_pos) -> tuple[Array, Array, Array]:
        """
        Process one already-embedded new token through all blocks using the KV cache.

        new_pos is the absolute position of the new token (prompt_len + step_idx).
        It is passed to each block's cached_step to apply RoPE at the correct position.

        x_new   : (1, 4, d_model) — embedded new token (output of embedder)
        k_cache : (num_blocks, W, 4, d_model)
        v_cache : (num_blocks, W, 4, d_model)
        new_pos : scalar int — absolute position of the new token
        Returns : (hidden, new_k_cache, new_v_cache)
            hidden      : (4, d_model) — hidden state for output heads
            new_k_cache : (num_blocks, W, 4, d_model)
            new_v_cache : (num_blocks, W, 4, d_model)
        """
        new_Ks, new_Vs = [], []
        for i, block in enumerate(self.blocks):
            x_new, k_i, v_i = block.cached_step(x_new, k_cache[i], v_cache[i], new_pos)
            new_Ks.append(k_i)
            new_Vs.append(v_i)
        x_new = _norm2d(self.final_norm, x_new)  # (1, 4, d_model)
        return x_new[0], jnp.stack(new_Ks), jnp.stack(new_Vs)

    def write_metadata(self, filepath):
        step  = self.embedder.step_embedder
        heads = self.output_heads
        block = self.blocks[0]
        metadata = {
            "d_model":             self.d_model,
            "instr_entity_dim":    heads.instr_to_table_proj.in_features,
            "table_entity_dim":    heads.table_proj.out_features,
            "softsynth_entity_dim": heads.instr_decoder.softsynth_decoder.linear_in.out_features,
            "num_heads_t":         block.temporal_attn.num_heads,
            "num_heads_c":         block.channel_attn.num_heads,
            "num_blocks":          len(self.blocks),
            "noise_sd":            self.noise_sd,
            "dropout_p":           block.dropout.p,
            "note_dim":        step.note_embedder.out_dim,
            "instr_dim":       step.instrument_embedder.out_dim,
            "fx_dim":          step.fx_embedder.out_dim,
            "transpose_dim":   step.transpose_embedder.out_dim,
            "value_out_dim":   heads.table_proj.out_features,
            "synth_waves_dim": step.instrument_embedder.embedder.embedders["synth_wave"].out_dim,
        }
        with open(filepath, "w") as f:
            f.write(json.dumps(metadata))
