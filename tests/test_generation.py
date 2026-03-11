import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from pe_lsdj.models import LSDJTransformer
from pe_lsdj.generation import (
    _generate,
    generate,
    match_groove,
    match_trace,
    match_table,
    match_softsynth,
    match_instrument,
)
from pe_lsdj.embedding.song import SongBanks
from pe_lsdj.models.transformer import (
    groove_loss,
    N_GROOVE_SLOTS,
    _GROOVE_CONT_MAX,
)
from pe_lsdj.constants import INSTR_WIDTH, WAV, PU


KEY = jr.PRNGKey(0)
D_MODEL = 64  # matches model fixture


@pytest.fixture(scope="module")
def model():
    return LSDJTransformer(
        KEY, d_model=D_MODEL, num_heads_t=2, num_heads_c=2, num_blocks=1,
    )


@pytest.fixture(scope="module")
def step_logits(model):
    """Representative logits and latents from a zero backbone output (d_model=64)."""
    x = jnp.zeros(D_MODEL)
    return model.output_heads.generation_outputs(x)


# ---------------------------------------------------------------------------
# Full-pipeline shape tests
# ---------------------------------------------------------------------------

def test_generate_jit_smoke(model):
    """generate must be traceable by eqx.filter_jit without errors."""
    S_in, num_steps = 8, 4
    input_tokens = jnp.zeros((S_in, 4, 21), dtype=jnp.uint16)
    out, _ = eqx.filter_jit(_generate)(model, input_tokens, KEY, num_steps=num_steps)
    assert out.shape == (S_in + num_steps, 4, 21)


def test_generated_seq_length_shape_matches_num_steps(model):
    S_in, num_steps = 8, 4
    input_tokens = jnp.zeros((S_in, 4, 21), dtype=jnp.uint16)
    output, _ = _generate(model, input_tokens, KEY, num_steps=num_steps)
    assert output.shape == (S_in + num_steps, 4, 21)


def test_occupied_slot_mask_updates(model):
    """After generating a few steps from empty banks, some slots should be occupied."""
    S_in = 4
    input_tokens = jnp.zeros((S_in, 4, 21), dtype=jnp.uint16)
    banks_init = SongBanks.default()
    _, banks_out = _generate(model, input_tokens, KEY, banks=banks_init, num_steps=2)
    # 4 channels × 2 steps each resolve an instrument
    assert jnp.any(banks_out.instrs_occupied)


# ---------------------------------------------------------------------------
# Loss ordering test (sanity check for matching heuristic)
# ---------------------------------------------------------------------------

def test_match_loss_groove_ordering(model, step_logits):
    """
    The rounded-sigmoid row (what _create_new_groove stores) should have lower
    groove_loss against the raw predictions than an all-zero row.
    """
    logits, latents = step_logits
    groove_ctx = latents['phrase_groove_ctx']
    predicted = model.output_heads.groove_decoder(groove_ctx, jnp.int32(N_GROOVE_SLOTS))

    close_row = jnp.round(
        jax.nn.sigmoid(predicted) * _GROOVE_CONT_MAX
    ).astype(jnp.uint8)
    far_row = jnp.zeros_like(close_row)

    assert groove_loss(predicted, close_row) <= groove_loss(predicted, far_row)


# ---------------------------------------------------------------------------
# Entity matching: slot allocation behaviour
#
# Convention for thresholds used in tests:
#   threshold = -1.0  → never match  (all losses ≥ 0 > -1) → always create new
#   threshold = inf   → always match (any loss ≤ inf)       → always reuse if occupied
# ---------------------------------------------------------------------------

def test_match_grooves(model, step_logits):
    logits, latents = step_logits
    groove_ctx = latents['phrase_groove_ctx']
    heads = model.output_heads
    slot_idx = jnp.int32(0)

    # Empty banks + threshold=-1 → creates slot 1
    banks = SongBanks.default()
    gid1, banks1 = match_groove(heads, banks, groove_ctx, slot_idx, threshold=-1.0)
    assert int(gid1) == 1
    assert bool(banks1.grooves_occupied[1])
    assert not jnp.any(banks1.grooves_occupied[2:])

    # One slot occupied + threshold=inf → reuses slot 1
    gid2, banks2 = match_groove(heads, banks1, groove_ctx, slot_idx, threshold=jnp.inf)
    assert int(gid2) == 1
    assert jnp.sum(banks2.grooves_occupied) == jnp.sum(banks1.grooves_occupied)

    # One slot occupied + threshold=-1 → creates slot 2
    gid3, banks3 = match_groove(heads, banks1, groove_ctx, slot_idx, threshold=-1.0)
    assert int(gid3) == 2
    assert bool(banks3.grooves_occupied[2])


def test_match_traces(model, step_logits):
    logits, latents = step_logits
    table_ctx = latents['table_ctx']
    heads = model.output_heads
    key = jr.PRNGKey(1)
    slot_idx = jnp.int32(0)

    banks = SongBanks.default()
    trid1, banks1 = match_trace(key, heads, banks, table_ctx, slot_idx, threshold=-1.0)
    assert int(trid1) == 1
    assert bool(banks1.tables_occupied[1])

    trid2, banks2 = match_trace(key, heads, banks1, table_ctx, slot_idx, threshold=jnp.inf)
    assert int(trid2) == 1
    assert jnp.sum(banks2.tables_occupied) == jnp.sum(banks1.tables_occupied)

    trid3, banks3 = match_trace(key, heads, banks1, table_ctx, slot_idx, threshold=-1.0)
    assert int(trid3) == 2
    assert bool(banks3.tables_occupied[2])


def test_match_tables(model, step_logits):
    logits, latents = step_logits
    table_logits = logits['table']
    table_ctx = latents['table_ctx']
    heads = model.output_heads
    key = jr.PRNGKey(2)

    banks = SongBanks.default()
    tid1, banks1 = match_table(key, heads, banks, table_logits, table_ctx, -1.0)
    assert int(tid1) == 1
    assert bool(banks1.tables_occupied[1])

    # threshold=inf → reuse an existing slot (table creation may also populate inner
    # trace slots, so we don't assert which specific slot is returned).
    tid2, banks2 = match_table(key, heads, banks1, table_logits, table_ctx, jnp.inf)
    assert bool(banks1.tables_occupied[int(tid2)])  # returned slot was already occupied
    assert jnp.sum(banks2.tables_occupied) == jnp.sum(banks1.tables_occupied)

    # threshold=-1 → always create new (no existing score can be ≤ -1).
    tid3, banks3 = match_table(key, heads, banks1, table_logits, table_ctx, -1.0)
    assert not bool(banks1.tables_occupied[int(tid3)])  # was not occupied before
    assert bool(banks3.tables_occupied[int(tid3)])


def test_match_softsynths(model, step_logits):
    logits, latents = step_logits
    softsynth_preds = logits['instr']['softsynth']
    key = jr.PRNGKey(3)

    banks = SongBanks.default()
    sid1, banks1 = match_softsynth(key, banks, softsynth_preds, threshold=-1.0)
    assert int(sid1) == 1
    assert bool(banks1.synths_occupied[1])

    sid2, banks2 = match_softsynth(key, banks1, softsynth_preds, threshold=jnp.inf)
    assert int(sid2) == 1
    assert jnp.sum(banks2.synths_occupied) == jnp.sum(banks1.synths_occupied)

    sid3, banks3 = match_softsynth(key, banks1, softsynth_preds, threshold=-1.0)
    assert int(sid3) == 2
    assert bool(banks3.synths_occupied[2])


def test_match_instruments(model, step_logits):
    logits, latents = step_logits
    instr_table_ctx = latents['instr_table_ctx']
    heads = model.output_heads
    key = jr.PRNGKey(4)

    # Force WAV type so sub-entity assertions (table + softsynth) are reliable.
    instr_preds = _force_type(logits['instr'], WAV)

    # Empty banks + threshold=-1 → creates instrument at slot 1, with sub-entities
    banks = SongBanks.default()
    iid1, banks1 = match_instrument(
        key, heads, banks, instr_preds, instr_table_ctx,
        instr_threshold=-1.0, table_threshold=-1.0, softsynth_threshold=-1.0,
    )
    assert int(iid1) == 1
    assert bool(banks1.instrs_occupied[1])
    # Both table and softsynth sub-entities must be created for a WAV instrument
    assert jnp.any(banks1.tables_occupied)
    assert jnp.any(banks1.synths_occupied)

    # One slot occupied + threshold=inf → reuses slot 1
    iid2, banks2 = match_instrument(
        key, heads, banks1, instr_preds, instr_table_ctx,
        instr_threshold=jnp.inf, table_threshold=jnp.inf, softsynth_threshold=jnp.inf,
    )
    assert int(iid2) == 1
    assert jnp.sum(banks2.instrs_occupied) == jnp.sum(banks1.instrs_occupied)

    # One slot occupied + threshold=-1 → creates slot 2
    iid3, banks3 = match_instrument(
        key, heads, banks1, instr_preds, instr_table_ctx,
        instr_threshold=-1.0, table_threshold=-1.0, softsynth_threshold=-1.0,
    )
    assert int(iid3) == 2
    assert bool(banks3.instrs_occupied[2])


# ---------------------------------------------------------------------------
# Edge case: unoccupied slots must not be matched
# ---------------------------------------------------------------------------

def test_masked_instr_not_used(model, step_logits):
    """
    A slot with content but not marked as occupied must not be returned as a match,
    even when threshold=inf (always-match). Only occupied slots are candidates.
    """
    logits, latents = step_logits
    instr_preds = logits['instr']
    instr_table_ctx = latents['instr_table_ctx']
    heads = model.output_heads
    key = jr.PRNGKey(5)

    # Slot 1: data present, NOT occupied.
    # Slot 2: zeros, IS occupied.
    banks = SongBanks.default()._replace(
        instruments=SongBanks.default().instruments.at[1].set(
            jnp.ones(INSTR_WIDTH, dtype=jnp.uint16)
        ),
        instrs_occupied=SongBanks.default().instrs_occupied.at[2].set(True),
    )

    # threshold=inf → must pick the only occupied slot (slot 2), ignoring slot 1
    iid, _ = match_instrument(
        key, heads, banks, instr_preds, instr_table_ctx,
        instr_threshold=jnp.inf, table_threshold=jnp.inf, softsynth_threshold=jnp.inf,
    )
    assert int(iid) == 2


# ---------------------------------------------------------------------------
# TYPE_ID gating: null type skips bank update; WAV type creates softsynth
# ---------------------------------------------------------------------------

def _force_type(instr_preds, type_val):
    """Return a copy of instr_preds with TYPE_ID logits biased to type_val.

    TYPE_ID is the first categorical field: offset 0, vocab 5 in ['cat'].
    Values: 0=NULL, 1=PU, 2=WAV, 3=KIT, 4=NOI.
    """
    biased = jnp.full(5, -1000.0).at[type_val].set(1000.0)
    return {**instr_preds, 'cat': instr_preds['cat'].at[:5].set(biased)}


def test_null_type_returns_null_instr_id(model, step_logits):
    """TYPE_ID=0 (NULL) must short-circuit match_instrument with id=0, banks unchanged."""
    logits, latents = step_logits
    instr_table_ctx = latents['instr_table_ctx']
    heads = model.output_heads
    key = jr.PRNGKey(10)

    null_preds = _force_type(logits['instr'], 0)
    banks = SongBanks.default()
    iid, banks_out = match_instrument(
        key, heads, banks, null_preds, instr_table_ctx,
        instr_threshold=0.05, table_threshold=0.05, softsynth_threshold=0.05,
    )
    assert int(iid) == 0
    assert not jnp.any(banks_out.instrs_occupied[1:])


# ---------------------------------------------------------------------------
# Batch generation (generate)
# ---------------------------------------------------------------------------

class TestBatchGenerate:
    S_IN = 4
    NUM_STEPS = 2

    @pytest.fixture(scope="class")
    def seed_tokens(self):
        return jnp.zeros((self.S_IN, 4, 21), dtype=jnp.uint16)

    def test_batch_shape(self, model, seed_tokens):
        M = 3
        tokens, _ = generate(model, seed_tokens, KEY, num_samples=M, num_steps=self.NUM_STEPS)
        assert tokens.shape == (M, self.S_IN + self.NUM_STEPS, 4, 21)

    def test_banks_have_batch_dim(self, model, seed_tokens):
        M = 3
        _, banks = generate(model, seed_tokens, KEY, num_samples=M, num_steps=self.NUM_STEPS)
        assert banks.instruments.shape[0] == M

    def test_samples_differ(self, model, seed_tokens):
        """Different keys must produce different token sequences."""
        M = 2
        tokens, _ = generate(model, seed_tokens, KEY, num_samples=M, num_steps=self.NUM_STEPS)
        assert not jnp.all(tokens[0] == tokens[1])

    def test_single_sample_consistent_with_generate(self, model, seed_tokens):
        """num_samples=1 should match _generate called with the same derived key."""
        sample_key = jr.split(KEY, 1)[0]
        tokens_batch, _ = generate(model, seed_tokens, KEY, num_samples=1, num_steps=self.NUM_STEPS)
        tokens_single, _ = _generate(model, seed_tokens, sample_key, num_steps=self.NUM_STEPS)
        assert jnp.array_equal(tokens_batch[0], tokens_single)

    def test_jit_smoke(self, model, seed_tokens):
        out, _ = eqx.filter_jit(generate)(model, seed_tokens, KEY, num_samples=2, num_steps=self.NUM_STEPS)
        assert out.shape == (2, self.S_IN + self.NUM_STEPS, 4, 21)


# ---------------------------------------------------------------------------
# Sliding context window (window_len)
# ---------------------------------------------------------------------------

class TestWindowLen:
    S_IN = 8
    NUM_STEPS = 2

    @pytest.fixture(scope="class")
    def seed_tokens(self):
        return jnp.zeros((self.S_IN, 4, 21), dtype=jnp.uint16)

    def test_truncate_shape(self, model, seed_tokens):
        """window_len < S_in → output has window_len + num_steps steps."""
        W = 4
        tokens, _ = _generate(model, seed_tokens, KEY, num_steps=self.NUM_STEPS, window_len=W)
        assert tokens.shape == (W + self.NUM_STEPS, 4, 21)

    def test_pad_shape(self, model, seed_tokens):
        """window_len > S_in → output has window_len + num_steps steps."""
        W = 12
        tokens, _ = _generate(model, seed_tokens, KEY, num_steps=self.NUM_STEPS, window_len=W)
        assert tokens.shape == (W + self.NUM_STEPS, 4, 21)

    def test_exact_shape(self, model, seed_tokens):
        """window_len == S_in → identical to no window_len."""
        tokens_w, _ = _generate(model, seed_tokens, KEY, num_steps=self.NUM_STEPS, window_len=self.S_IN)
        tokens_n, _ = _generate(model, seed_tokens, KEY, num_steps=self.NUM_STEPS)
        assert jnp.array_equal(tokens_w, tokens_n)

    def test_different_window_affects_output(self, model, seed_tokens):
        """A shorter window changes what the model conditions on → different output."""
        tokens_full, _ = _generate(model, seed_tokens, KEY, num_steps=self.NUM_STEPS)
        tokens_short, _ = _generate(model, seed_tokens, KEY, num_steps=self.NUM_STEPS, window_len=2)
        # The generated (new) tokens may differ when context is truncated
        assert not jnp.array_equal(tokens_full[-self.NUM_STEPS:], tokens_short[-self.NUM_STEPS:])

    def test_window_len_in_batch_generate(self, model, seed_tokens):
        """window_len threads through generate correctly."""
        W = 4
        tokens, _ = generate(model, seed_tokens, KEY, num_samples=2, num_steps=self.NUM_STEPS, window_len=W)
        assert tokens.shape == (2, W + self.NUM_STEPS, 4, 21)


def test_wav_creates_softsynth_non_wav_does_not(model, step_logits):
    """WAV instrument must allocate a softsynth slot; PU instrument must not."""
    logits, latents = step_logits
    instr_table_ctx = latents['instr_table_ctx']
    heads = model.output_heads
    key = jr.PRNGKey(11)

    wav_preds = _force_type(logits['instr'], WAV)
    banks = SongBanks.default()
    _, banks_wav = match_instrument(
        key, heads, banks, wav_preds, instr_table_ctx,
        instr_threshold=-1.0, table_threshold=-1.0, softsynth_threshold=-1.0,
    )
    assert jnp.any(banks_wav.synths_occupied[1:])

    pu_preds = _force_type(logits['instr'], PU)
    banks = SongBanks.default()
    _, banks_pu = match_instrument(
        key, heads, banks, pu_preds, instr_table_ctx,
        instr_threshold=-1.0, table_threshold=-1.0, softsynth_threshold=-1.0,
    )
    assert not jnp.any(banks_pu.synths_occupied[1:])
