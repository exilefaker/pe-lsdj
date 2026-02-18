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
from pe_lsdj.embedding.song import SongStepEmbedder


KEY = jr.PRNGKey(0)

def _split(n):
    return jr.split(KEY, n)


@pytest.fixture(scope="module")
def song_step_embedder():
    """Construct a full SongStepEmbedder with correctly-shaped mock banks."""
    k = jr.PRNGKey(33)

    return SongStepEmbedder(
        k,
        instruments=jnp.zeros((64, INSTR_WIDTH)),
        softsynths=jnp.zeros((16, SOFTSYNTH_WIDTH)),
        waveframes=jnp.zeros((16, WAVES_PER_SYNTH * FRAMES_PER_WAVE)),
        grooves=jnp.zeros((32, STEPS_PER_GROOVE * 2)),
        tables=jnp.zeros((64, TABLE_WIDTH)),
        traces=jnp.zeros((64, TABLE_WIDTH)),
    )

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
    e = SumEmbedder([
        EnumEmbedder(5, 16, k1),
        GatedNormedEmbedder(16, k2, in_dim=2),
        GatedNormedEmbedder(16, k3, in_dim=1),
    ])
    assert e.offsets[-1] == e.in_dim
    assert (0, 1, 3, 4) == e.offsets

def test_concat_embedder_offsets():
    k1, k2, k3 = _split(3)
    e = ConcatEmbedder(k3, [
        EnumEmbedder(5, 16, k1),
        GatedNormedEmbedder(16, k2, in_dim=2),
    ], out_dim=8)
    assert e.offsets[-1] == e.in_dim
    assert (0, 1, 3) == e.offsets

def test_sum_embedder_rejects_mismatched_out_dims():
    k1, k2 = _split(2)
    with pytest.raises(AssertionError, match="out_dims must match"):
        SumEmbedder([
            EnumEmbedder(5, 16, k1),
            EnumEmbedder(5, 32, k2),
        ])

def test_concat_embedder_projection_sharing():
    k1, k2, k3, k4 = _split(4)
    e1 = ConcatEmbedder(k1, [EnumEmbedder(5, 16, k2)], out_dim=8)
    e2 = ConcatEmbedder(k3, [EnumEmbedder(5, 16, k4)], out_dim=8,
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
    assert e.offsets[-1] == e.in_dim
    assert e.in_dim == FX_VALUES_FEATURE_DIM
    assert e.out_dim == 64
    # All sub-embedder out_dims must be 64
    for sub in e.embedders:
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
    assert te.embedders[te.fx1_idx] is te.embedders[te.fx2_idx]

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
    # start_params and end_params (indices 5 and 6) share weights
    assert se.embedders[5] is se.embedders[6]

# --- Entity bank / inner embedder compatibility ---

def test_entity_bank_compat(song_step_embedder):
    # Instrument entity bank columns == InstrumentEmbedder.in_dim
    ie = song_step_embedder.instrument_embedder
    assert ie.entity_bank.shape[1] == ie.embedder.in_dim

def test_table_entity_bank_compat(song_step_embedder):
    # Table entity bank columns == TableEmbedder.in_dim
    te = song_step_embedder.instrument_embedder.embedder.embedders[1]
    assert te.entity_bank.shape[1] == te.embedder.in_dim

def test_softsynth_entity_bank_compat(song_step_embedder):
    se = song_step_embedder.instrument_embedder.embedder.embedders[17]
    assert se.entity_bank.shape[1] == se.embedder.in_dim

def test_waveframe_entity_bank_compat(song_step_embedder):
    we = song_step_embedder.instrument_embedder.embedder.embedders[-1]
    assert we.entity_bank.shape[1] == we.embedder.in_dim

def test_groove_entity_bank_compat(song_step_embedder):
    # Groove embedder is inside fx_embedder -> fx_value -> embedders[1]
    ge = song_step_embedder.fx_embedder.embedders[1].embedders[1]
    assert ge.entity_bank.shape[1] == ge.embedder.in_dim

# --- Two-tier table embedding projection sharing ---

def test_tier_projection_sharing(song_step_embedder):
    # Phrase FXEmbedder shares projection with tier 0
    phrase_fx = song_step_embedder.fx_embedder
    # The table entity embedder inside phrase FX value embedder
    table_entity_embedder = phrase_fx.embedders[1].embedders[0]
    # table_entity.embedder is Tier 1 TableEmbedder
    tier1_table = table_entity_embedder.embedder
    # Tier 1 FXEmbedder is inside tier1_table
    tier1_fx = tier1_table.embedders[tier1_table.fx1_idx]
    # Trace entity is inside tier1 FXValueEmbedder
    trace_entity_embedder = tier1_fx.embedders[1].embedders[0]
    # trace_entity.embedder is Tier 0 TableEmbedder
    tier0_table = trace_entity_embedder.embedder
    tier0_fx = tier0_table.embedders[tier0_table.fx1_idx]
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
        e = SumEmbedder([
            EnumEmbedder(5, 16, k1),
            GatedNormedEmbedder(16, k2, in_dim=2),
            GatedNormedEmbedder(16, k3),
        ])
        out = e(jnp.array([3, 100, 50, 42]))
        assert out.shape == (16,)

    def test_concat_embedder(self):
        k1, k2, k3 = _split(3)
        e = ConcatEmbedder(k3, [
            EnumEmbedder(5, 16, k1),
            GatedNormedEmbedder(32, k2, in_dim=2),
        ], out_dim=8)
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
        table_entity = EntityEmbedder(jnp.ones((64, TABLE_WIDTH)), te0)
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
        assert out.shape == (512,)

    def test_song_step_embedder_nonzero(self, song_step_embedder):
        step = jnp.ones((4, 21))
        out = song_step_embedder(step)
        assert out.shape == (512,)
        assert jnp.linalg.norm(out) > 0
    
    # Real data
    def test_song_step_on_real_data(self, song_file):
        k = jr.PRNGKey(33)
        song_step_embedder = SongStepEmbedder(
            k,
            song_file.instruments_array,
            song_file.softsynths_array,
            song_file.waveframes_array,
            song_file.grooves_array,
            song_file.tables_array,
            song_file.traces_array,
        )
        out = song_step_embedder(song_file.song_tokens[0])
        assert out.shape == (512,)
        assert jnp.linalg.norm(out) > 0
