import pytest
import jax.numpy as jnp
import jax.random as jr

from pe_lsdj.constants import *
from pe_lsdj.embedding.base import (
    BaseEmbedder,
    DummyEmbedder,
    EnumEmbedder,
    GatedNormedEmbedder,
    EntityEmbedder,
    SumEmbedder,
    ConcatEmbedder,
)
from pe_lsdj.embedding.fx import (
    GrooveEntityEmbedder,
    build_fx_value_embedders,
    FXValueEmbedder,
    FXEmbedder,
    TableEmbedder,
)
from pe_lsdj.embedding.instrument import (
    SoftsynthEmbedder,
    WaveframeEmbedder,
)
from pe_lsdj.embedding.song import (
    SongBanks,
    SongStepEmbedder,
    SequenceEmbedder,
    set_banks,
)
from pe_lsdj.embedding.position import (
    SinusoidalPositionEncoding,
    PhrasePositionEmbedder,
    ChannelPositionEmbedder,
)


KEY = jr.PRNGKey(0)

def _split(n):
    return jr.split(KEY, n)


@pytest.fixture(scope="module")
def song_step_embedder():
    """Construct a full SongStepEmbedder with default (zero) banks."""
    k = jr.PRNGKey(33)
    return SongStepEmbedder(k)

# ===================================================================
# Basic embedder class structures
# ===================================================================

# --- Leaf embedders ---

def test_enum_embedder_dims():
    e = EnumEmbedder(5, 16, KEY)
    assert e.in_dim == 1
    assert e.out_dim == 16
    assert e.vocab_size == 5

def test_gated_normed_embedder_dims():
    e = GatedNormedEmbedder(16, KEY, in_dim=3)
    assert e.in_dim == 3
    assert e.out_dim == 16

def test_gated_normed_embedder_default_in_dim():
    e = GatedNormedEmbedder(16, KEY)
    assert e.in_dim == 1

def test_dummy_embedder_dims():
    e = DummyEmbedder(1, 64)
    assert e.in_dim == 1
    assert e.out_dim == 64

def test_entity_embedder_dims():
    bank = jnp.zeros((10, 4))
    inner = GatedNormedEmbedder(16, KEY, in_dim=4)
    e = EntityEmbedder(bank, inner)
    assert e.in_dim == 1
    assert e.out_dim == 16
    assert e.num_entities == 10
    assert bank.shape[1] == e.embedder.in_dim

# --- Aggregator offsets ---

def test_sum_embedder_offsets():
    k1, k2, k3 = _split(3)
    e = SumEmbedder({
        'a': EnumEmbedder(5, 16, k1),
        'b': GatedNormedEmbedder(16, k2, in_dim=2),
        'c': GatedNormedEmbedder(16, k3, in_dim=1),
    })
    assert e.in_dim == 4
    assert e.offsets == {'a': 0, 'b': 1, 'c': 3}

def test_concat_embedder_offsets():
    k1, k2, k3 = _split(3)
    e = ConcatEmbedder(k3, {
        'a': EnumEmbedder(5, 16, k1),
        'b': GatedNormedEmbedder(16, k2, in_dim=2),
    }, out_dim=8)
    assert e.in_dim == 3
    assert e.offsets == {'a': 0, 'b': 1}

def test_sum_embedder_rejects_mismatched_out_dims():
    k1, k2 = _split(2)
    with pytest.raises(AssertionError, match="out_dims must match"):
        SumEmbedder({
            'a': EnumEmbedder(5, 16, k1),
            'b': EnumEmbedder(5, 32, k2),
        })

def test_concat_embedder_projection_sharing():
    k1, k2, k3, k4 = _split(4)
    e1 = ConcatEmbedder(k1, {'a': EnumEmbedder(5, 16, k2)}, out_dim=8)
    e2 = ConcatEmbedder(k3, {'a': EnumEmbedder(5, 16, k4)}, out_dim=8,
                         _projection=e1.projection)
    assert e2.projection is e1.projection

# --- FX embedder structure ---

def test_fx_value_embedder_structure():
    k1, k2 = _split(2)
    grooves = jnp.zeros((32, STEPS_PER_GROOVE * 2))
    ge = GrooveEntityEmbedder(64, k1, grooves)
    subs = build_fx_value_embedders(64, k2, ge)
    dummy = DummyEmbedder(1, 64)
    e = FXValueEmbedder(dummy, subs)
    assert e.in_dim == FX_VALUES_FEATURE_DIM
    assert e.out_dim == 64
    # All sub-embedder out_dims must be 64
    for sub in e.embedders.values():
        assert sub.out_dim == 64

def test_fx_embedder_in_dim():
    k1, k2, k3 = _split(3)
    grooves = jnp.zeros((32, STEPS_PER_GROOVE * 2))
    ge = GrooveEntityEmbedder(64, k1, grooves)
    subs = build_fx_value_embedders(64, k2, ge)
    dummy = DummyEmbedder(1, 64)
    fxv = FXValueEmbedder(dummy, subs)
    fx = FXEmbedder(k3, fxv, 128)
    # fx_cmd(1) + fx_value(17)
    assert fx.in_dim == 1 + fxv.in_dim

# --- TableEmbedder weight sharing ---

def test_table_embedder_fx_weight_sharing():
    k1, k2, k3, k4 = _split(4)
    grooves = jnp.zeros((32, STEPS_PER_GROOVE * 2))
    ge = GrooveEntityEmbedder(64, k1, grooves)
    subs = build_fx_value_embedders(64, k2, ge)
    dummy = DummyEmbedder(1, 64)
    fxv = FXValueEmbedder(dummy, subs)
    fx_emb = FXEmbedder(k3, fxv, 64)
    te = TableEmbedder(64, k4, fx_emb)
    assert te.embedders['fx1'] is te.embedders['fx2']

def test_table_embedder_in_dim_matches_table_width():
    k1, k2, k3, k4 = _split(4)
    grooves = jnp.zeros((32, STEPS_PER_GROOVE * 2))
    ge = GrooveEntityEmbedder(64, k1, grooves)
    subs = build_fx_value_embedders(64, k2, ge)
    dummy = DummyEmbedder(1, 64)
    fxv = FXValueEmbedder(dummy, subs)
    fx_emb = FXEmbedder(k3, fxv, 64)
    te = TableEmbedder(64, k4, fx_emb)
    assert te.in_dim == TABLE_WIDTH

# --- SoftsynthEmbedder weight sharing ---

def test_softsynth_param_weight_sharing():
    se = SoftsynthEmbedder(KEY)
    assert se.embedders['start_params'] is se.embedders['end_params']

# --- Entity bank / inner embedder compatibility ---

def test_entity_bank_compat(song_step_embedder):
    # Instrument entity bank columns == InstrumentEmbedder.in_dim
    ie = song_step_embedder.instrument_embedder
    assert ie.entity_bank.shape[1] == ie.embedder.in_dim

def test_table_entity_bank_compat(song_step_embedder):
    # Table entity bank columns == TableEmbedder.in_dim
    te = song_step_embedder.instrument_embedder.embedder.embedders['table']
    assert te.entity_bank.shape[1] == te.embedder.in_dim

def test_softsynth_entity_bank_compat(song_step_embedder):
    se = song_step_embedder.instrument_embedder.embedder.embedders['softsynth']
    assert se.entity_bank.shape[1] == se.embedder.in_dim

def test_waveframe_entity_bank_compat(song_step_embedder):
    we = song_step_embedder.instrument_embedder.embedder.embedders['waveframe']
    assert we.entity_bank.shape[1] == we.embedder.in_dim

def test_groove_entity_bank_compat(song_step_embedder):
    ge = song_step_embedder.fx_embedder.embedders['value'].embedders['groove']
    assert ge.entity_bank.shape[1] == ge.embedder.in_dim

# --- Two-tier table embedding projection sharing ---

def test_tier_projection_sharing(song_step_embedder):
    # Phrase FXEmbedder shares projection with tier 0
    phrase_fx = song_step_embedder.fx_embedder
    # The table entity embedder inside phrase FX value embedder
    table_entity = phrase_fx.embedders['value'].embedders['table_fx']
    # table_entity.embedder is Tier 1 TableEmbedder
    tier1_table = table_entity.embedder
    # Tier 1 FXEmbedder is inside tier1_table
    tier1_fx = tier1_table.embedders['fx1']
    # Trace entity is inside tier1 FXValueEmbedder
    trace_entity = tier1_fx.embedders['value'].embedders['table_fx']
    # trace_entity.embedder is Tier 0 TableEmbedder
    tier0_table = trace_entity.embedder
    tier0_fx = tier0_table.embedders['fx1']
    # Verify projection sharing
    assert phrase_fx.projection is tier0_fx.projection
    assert tier1_fx.projection is tier0_fx.projection
    assert tier1_table.projection is tier0_table.projection


# ===================================================================
# Forward pass shape checks
# ===================================================================

class TestForwardPassShapes:

    # --- Leaf embedders ---

    @pytest.mark.parametrize("vocab_size,out_dim", [(2, 8), (5, 16), (19, 32)])
    def test_enum_embedder(self, vocab_size, out_dim):
        e = EnumEmbedder(vocab_size, out_dim, KEY)
        out = e(jnp.array([2]))
        assert out.shape == (out_dim,)

    @pytest.mark.parametrize("in_dim", [1, 2, 4])
    def test_gated_normed_embedder(self, in_dim):
        e = GatedNormedEmbedder(16, KEY, in_dim=in_dim)
        out = e(jnp.ones((in_dim,)) * 100)
        assert out.shape == (16,)

    def test_gated_normed_zero_input(self):
        """Zero input should produce the null-gate embedding (no continuous contribution)."""
        e = GatedNormedEmbedder(16, KEY, in_dim=1)
        out = e(jnp.zeros((1,)))
        assert out.shape == (16,)

    def test_dummy_embedder(self):
        e = DummyEmbedder(1, 64)
        out = e(jnp.array([5]))
        assert out.shape == (64,)
        assert jnp.all(out == 0)

    def test_entity_embedder(self):
        bank = jnp.ones((10, 4))
        inner = GatedNormedEmbedder(16, KEY, in_dim=4)
        e = EntityEmbedder(bank, inner)
        out = e(jnp.array([3]))
        assert out.shape == (16,)

    # --- Aggregators ---

    def test_sum_embedder(self):
        k1, k2, k3 = _split(3)
        e = SumEmbedder({
            'a': EnumEmbedder(5, 16, k1),
            'b': GatedNormedEmbedder(16, k2, in_dim=2),
            'c': GatedNormedEmbedder(16, k3),
        })
        out = e(jnp.array([3, 100, 50, 42]))
        assert out.shape == (16,)

    def test_concat_embedder(self):
        k1, k2, k3 = _split(3)
        e = ConcatEmbedder(k3, {
            'a': EnumEmbedder(5, 16, k1),
            'b': GatedNormedEmbedder(32, k2, in_dim=2),
        }, out_dim=8)
        out = e(jnp.array([3, 100, 50]))
        assert out.shape == (8,)

    # --- FX chain ---

    def test_fx_value_embedder(self):
        k1, k2 = _split(2)
        grooves = jnp.ones((32, STEPS_PER_GROOVE * 2))
        ge = GrooveEntityEmbedder(64, k1, grooves)
        subs = build_fx_value_embedders(64, k2, ge)
        dummy = DummyEmbedder(1, 64)
        e = FXValueEmbedder(dummy, subs)
        x = jnp.ones((e.in_dim,))
        out = e(x)
        assert out.shape == (64,)

    def test_fx_embedder(self):
        k1, k2, k3 = _split(3)
        grooves = jnp.ones((32, STEPS_PER_GROOVE * 2))
        ge = GrooveEntityEmbedder(64, k1, grooves)
        subs = build_fx_value_embedders(64, k2, ge)
        dummy = DummyEmbedder(1, 64)
        fxv = FXValueEmbedder(dummy, subs)
        e = FXEmbedder(k3, fxv, 128)
        x = jnp.ones((e.in_dim,))
        out = e(x)
        assert out.shape == (128,)

    def test_table_embedder(self):
        k1, k2, k3, k4 = _split(4)
        grooves = jnp.ones((32, STEPS_PER_GROOVE * 2))
        ge = GrooveEntityEmbedder(64, k1, grooves)
        subs = build_fx_value_embedders(64, k2, ge)
        dummy = DummyEmbedder(1, 64)
        fxv = FXValueEmbedder(dummy, subs)
        fx_emb = FXEmbedder(k3, fxv, 64)
        e = TableEmbedder(64, k4, fx_emb)
        x = jnp.ones((e.in_dim,))
        out = e(x)
        assert out.shape == (64,)

    def test_phrase_fx_embedder(self):
        k1, k2, k3, k4, k5 = _split(5)
        grooves = jnp.ones((32, STEPS_PER_GROOVE * 2))
        ge = GrooveEntityEmbedder(64, k1, grooves)
        subs = build_fx_value_embedders(64, k2, ge)

        # Tier 0
        dummy = DummyEmbedder(1, 64)
        fxv0 = FXValueEmbedder(dummy, subs)
        fx0 = FXEmbedder(k3, fxv0, 128)
        te0 = TableEmbedder(64, k4, fx0)

        # Phrase level: table entity at position 0
        table_entity = EntityEmbedder(
            jnp.ones((NUM_TABLES, TABLE_WIDTH)), te0,
        )
        fxv_phrase = FXValueEmbedder(table_entity, subs)
        e = FXEmbedder(k5, fxv_phrase, 128, _projection=fx0.projection)
        x = jnp.ones((e.in_dim,))
        out = e(x)
        assert out.shape == (128,)

    # --- Instrument chain ---

    def test_softsynth_embedder(self):
        e = SoftsynthEmbedder(KEY)
        x = jnp.ones((e.in_dim,))
        out = e(x)
        assert out.shape == (e.out_dim,)

    def test_waveframe_embedder(self):
        e = WaveframeEmbedder(KEY, out_dim=32)
        x = jnp.ones((e.in_dim,))
        out = e(x)
        assert out.shape == (32,)

    # --- Full SongStepEmbedder ---

    def test_song_step_embedder_zero(self, song_step_embedder):
        step = jnp.zeros((4, 21))
        out = song_step_embedder(step)
        assert out.shape == (4, 256)

    def test_song_step_embedder_nonzero(self, song_step_embedder):
        step = jnp.ones((4, 21))
        out = song_step_embedder(step)
        assert out.shape == (4, 256)
        assert jnp.linalg.norm(out) > 0
    
    # --- Null value behavior ---

    def test_entity_embedder_null_entry_zero_index(self):
        """With null_entry=True, index 0 feeds zeros to the inner embedder."""
        bank = jnp.ones((10, 4))
        inner = GatedNormedEmbedder(16, KEY, in_dim=4)
        e = EntityEmbedder(bank, inner, null_entry=True)
        out_null = e(jnp.array([0]))
        # Index 0 → zero row → GatedNormedEmbedder gets all zeros → null gate
        out_real = e(jnp.array([1]))
        # Null output should differ from a real entity
        assert not jnp.allclose(out_null, out_real)

    def test_entity_embedder_null_entry_bank_size(self):
        """null_entry=True prepends a zero row, increasing num_entities by 1."""
        bank = jnp.ones((10, 4))
        inner = GatedNormedEmbedder(16, KEY, in_dim=4)
        e = EntityEmbedder(bank, inner, null_entry=True)
        assert e.num_entities == 11
        assert e.entity_bank.shape == (11, 4)
        # Row 0 should be all zeros
        assert jnp.all(e.entity_bank[0] == 0)

    def test_gated_normed_null_vs_active(self):
        """GatedNormedEmbedder: zero input (null) should differ from active input."""
        e = GatedNormedEmbedder(16, KEY, in_dim=1, null_value=0, max_value=255)
        out_null = e(jnp.zeros((1,)))
        out_active = e(jnp.array([128.0]))
        assert not jnp.allclose(out_null, out_active)

    def test_gated_normed_null_no_continuous(self):
        """GatedNormedEmbedder: null input should have no continuous component."""
        e = GatedNormedEmbedder(16, KEY, in_dim=2, null_value=0, max_value=15)
        out_null = e(jnp.zeros((2,)))
        # The gate embedding for event=0 plus zero continuous contribution
        gate_only = e.gate_embedder(jnp.array(0.0))
        assert jnp.allclose(out_null, gate_only)

    def test_dummy_embedder_always_zero(self):
        """DummyEmbedder returns zeros regardless of input."""
        e = DummyEmbedder(3, 64)
        assert jnp.all(e(jnp.array([1, 2, 3])) == 0)
        assert jnp.all(e(jnp.zeros((3,))) == 0)
        assert jnp.all(e(jnp.ones((3,)) * 255) == 0)

    # --- Soft mode ---

    def test_enum_embedder_soft_mode(self):
        """EnumEmbedder soft mode accepts a probability vector."""
        e = EnumEmbedder(5, 16, KEY)
        probs = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
        out = e(probs, soft=True)
        assert out.shape == (16,)

    def test_enum_embedder_soft_onehot_matches_hard(self):
        """Soft mode with a one-hot vector should match hard mode."""
        e = EnumEmbedder(5, 16, KEY)
        hard_out = e(jnp.array([3]))
        soft_out = e(jnp.array([0, 0, 0, 1, 0], dtype=jnp.float32), soft=True)
        assert jnp.allclose(hard_out, soft_out, atol=1e-5)

    def test_entity_embedder_soft_mode(self):
        """EntityEmbedder soft mode: probability vector selects mixture of entities."""
        bank = jnp.arange(30, dtype=jnp.float32).reshape(10, 3)
        inner = GatedNormedEmbedder(16, KEY, in_dim=3)
        e = EntityEmbedder(bank, inner)
        probs = jnp.zeros(10).at[3].set(1.0)
        out_soft = e(probs, soft=True)
        out_hard = e(jnp.array([3]))
        assert out_soft.shape == (16,)
        assert jnp.allclose(out_soft, out_hard, atol=1e-5)

    def test_entity_embedder_soft_mixture(self):
        """Soft mode with uniform weights produces a valid embedding."""
        bank = jnp.ones((10, 4))
        inner = GatedNormedEmbedder(16, KEY, in_dim=4)
        e = EntityEmbedder(bank, inner)
        probs = jnp.ones(10) / 10.0
        out = e(probs, soft=True)
        assert out.shape == (16,)

    def test_song_step_embedder_default_banks(self):
        """SongStepEmbedder with no banks arg should produce valid output."""
        step = jnp.zeros((4, 21))
        emb = SongStepEmbedder(jr.PRNGKey(99))
        out = emb(step)
        assert out.shape == (4, 256)

    # --- Real data ---

    def test_song_step_on_real_data(self, song_file):
        k = jr.PRNGKey(33)
        banks = SongBanks.from_songfile(song_file)
        song_step_embedder = SongStepEmbedder(k, banks=banks)
        out = song_step_embedder(song_file.song_tokens[0])
        assert out.shape == (4, 256)
        assert jnp.linalg.norm(out) > 0


# ===================================================================
# Positional encodings
# ===================================================================

class TestSinusoidalPositionEncoding:

    @pytest.mark.parametrize("seq_len", [1, 16, 128])
    def test_shape(self, seq_len):
        enc = SinusoidalPositionEncoding(512)
        out = enc(jnp.arange(seq_len))
        assert out.shape == (seq_len, 512)

    def test_deterministic(self):
        enc = SinusoidalPositionEncoding(64)
        pos = jnp.arange(32)
        out1 = enc(pos)
        out2 = enc(pos)
        assert jnp.allclose(out1, out2)

    def test_distinct_positions(self):
        enc = SinusoidalPositionEncoding(64)
        out = enc(jnp.arange(4))
        # Each position should produce a unique vector
        for i in range(4):
            for j in range(i + 1, 4):
                assert not jnp.allclose(out[i], out[j])


class TestPhrasePositionEmbedder:

    def test_shape(self):
        e = PhrasePositionEmbedder(64, KEY)
        out = e(jnp.arange(32))
        assert out.shape == (32, 64)

    def test_periodicity(self):
        """Position 0 at step 0 should equal position 0 at step 16."""
        e = PhrasePositionEmbedder(64, KEY)
        positions = jnp.arange(32) % STEPS_PER_PHRASE
        out = e(positions)
        assert jnp.allclose(out[0], out[16])
        assert jnp.allclose(out[1], out[17])


class TestChannelPositionEmbedder:

    def test_shape(self):
        e = ChannelPositionEmbedder(64, KEY)
        out = e()
        assert out.shape == (4, 64)

    def test_distinct_channels(self):
        e = ChannelPositionEmbedder(64, KEY)
        out = e()
        for i in range(4):
            for j in range(i + 1, 4):
                assert not jnp.allclose(out[i], out[j])


class TestSequenceEmbedder:

    def test_shape(self, song_step_embedder):
        seq_emb = SequenceEmbedder(song_step_embedder, KEY)
        tokens = jnp.zeros((32, 4, 21))
        out = seq_emb(tokens)
        assert out.shape == (32, 4, 256)

    def test_on_real_data(self, song_file):
        k1, k2 = jr.split(jr.PRNGKey(33))
        banks = SongBanks.from_songfile(song_file)
        step_emb = SongStepEmbedder(k1, banks=banks)
        seq_emb = SequenceEmbedder(step_emb, k2)
        # Embed first 32 steps
        tokens = song_file.song_tokens[:32]
        out = seq_emb(tokens)
        assert out.shape == (32, 4, 256)
        assert jnp.linalg.norm(out) > 0


# ===================================================================
# Bank swapping
# ===================================================================

class TestSongBanks:

    def test_default_shapes(self):
        banks = SongBanks.default()
        assert banks.instruments.shape == (NUM_INSTRUMENTS + 1, INSTR_WIDTH)
        assert banks.softsynths.shape == (NUM_SYNTHS + 1, SOFTSYNTH_WIDTH)
        assert banks.waveframes.shape == (NUM_SYNTHS + 1, WAVES_PER_SYNTH * FRAMES_PER_WAVE)
        assert banks.grooves.shape == (NUM_GROOVES + 1, STEPS_PER_GROOVE * 2)
        assert banks.tables.shape == (NUM_TABLES + 1, TABLE_WIDTH)
        assert banks.traces.shape == (NUM_TABLES + 1, TABLE_WIDTH)

    def test_from_songfile(self, song_file):
        banks = SongBanks.from_songfile(song_file)
        assert banks.instruments.shape == (NUM_INSTRUMENTS + 1, INSTR_WIDTH)
        assert banks.tables.shape == (NUM_TABLES + 1, TABLE_WIDTH)


class TestSetBanks:

    def test_set_banks_changes_output(self, song_step_embedder):
        """Swapping in non-zero banks should change the embedding output."""
        seq_emb = SequenceEmbedder(song_step_embedder, KEY)
        tokens = jnp.zeros((4, 4, 21))

        out_before = seq_emb(tokens)

        # Swap in banks with non-zero data
        banks = SongBanks(
            instruments=jnp.ones((NUM_INSTRUMENTS + 1, INSTR_WIDTH)),
            softsynths=jnp.ones((NUM_SYNTHS + 1, SOFTSYNTH_WIDTH)),
            waveframes=jnp.ones((NUM_SYNTHS + 1, WAVES_PER_SYNTH * FRAMES_PER_WAVE)),
            grooves=jnp.ones((NUM_GROOVES + 1, STEPS_PER_GROOVE * 2)),
            tables=jnp.ones((NUM_TABLES + 1, TABLE_WIDTH)),
            traces=jnp.ones((NUM_TABLES + 1, TABLE_WIDTH)),
        )
        seq_emb2 = seq_emb.with_banks(banks)
        out_after = seq_emb2(tokens)

        assert not jnp.allclose(out_before, out_after)

    def test_set_banks_matches_direct_construction(self, song_file):
        """set_banks should produce the same result as constructing with those banks."""
        k1, k2 = jr.split(jr.PRNGKey(42))

        # Direct construction with song banks
        banks = SongBanks.from_songfile(song_file)
        step_direct = SongStepEmbedder(k1, banks=banks)
        seq_direct = SequenceEmbedder(step_direct, k2)

        # Construction with defaults, then bank swap
        step_default = SongStepEmbedder(k1)
        seq_swapped = SequenceEmbedder(step_default, k2).with_banks(banks)

        tokens = song_file.song_tokens[:8]
        out_direct = seq_direct(tokens)
        out_swapped = seq_swapped(tokens)
        assert jnp.allclose(out_direct, out_swapped, atol=1e-5)

    def test_set_banks_preserves_learned_params(self, song_step_embedder):
        """Bank swapping should not change any learned parameters."""
        seq_emb = SequenceEmbedder(song_step_embedder, KEY)

        banks = SongBanks(
            instruments=jnp.ones((NUM_INSTRUMENTS + 1, INSTR_WIDTH)),
            softsynths=jnp.ones((NUM_SYNTHS + 1, SOFTSYNTH_WIDTH)),
            waveframes=jnp.ones((NUM_SYNTHS + 1, WAVES_PER_SYNTH * FRAMES_PER_WAVE)),
            grooves=jnp.ones((NUM_GROOVES + 1, STEPS_PER_GROOVE * 2)),
            tables=jnp.ones((NUM_TABLES + 1, TABLE_WIDTH)),
            traces=jnp.ones((NUM_TABLES + 1, TABLE_WIDTH)),
        )
        seq_emb2 = seq_emb.with_banks(banks)

        # Channel projections should be identical
        assert jnp.array_equal(
            seq_emb.step_embedder.channel_projections,
            seq_emb2.step_embedder.channel_projections,
        )
        # FX cmd embedding weights should be identical
        assert jnp.array_equal(
            seq_emb.step_embedder.fx_embedder.embedders['cmd'].projection.weight,
            seq_emb2.step_embedder.fx_embedder.embedders['cmd'].projection.weight,
        )
