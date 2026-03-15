import pytest
import jax.numpy as jnp
import jax.random as jr

from pe_lsdj.constants import *
from pe_lsdj.embedding.base import (
    BaseEmbedder,
    DummyEmbedder,
    EnumEmbedder,
    EntityEmbedder,
    EntityType,
    GatedNormedEmbedder,
    HelixEmbedder,
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
    SynthWavesEmbedder,
    WaveframeEmbedder,
)
from pe_lsdj.embedding.song import (
    SongBanks,
    SongStepEmbedder,
    SequenceEmbedder,
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
    """Construct a full SongStepEmbedder (no banks stored at construction time)."""
    k = jr.PRNGKey(33)
    return SongStepEmbedder(k)


@pytest.fixture(scope="module")
def default_banks():
    return SongBanks.default()


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
    inner = GatedNormedEmbedder(16, KEY, in_dim=STEPS_PER_GROOVE * 2)
    e = EntityEmbedder(EntityType.GROOVES, inner)
    assert e.in_dim == 1
    assert e.out_dim == 16
    assert e.entity_type == EntityType.GROOVES

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
    ge = GrooveEntityEmbedder(64, k1)
    subs = build_fx_value_embedders(64, k2, ge)
    dummy = DummyEmbedder(1, 64)
    e = FXValueEmbedder(dummy, subs)
    assert e.in_dim == FX_VALUES_FEATURE_DIM
    assert e.out_dim == 64
    for sub in e.embedders.values():
        assert sub.out_dim == 64

def test_fx_embedder_in_dim():
    k1, k2, k3 = _split(3)
    ge = GrooveEntityEmbedder(64, k1)
    subs = build_fx_value_embedders(64, k2, ge)
    dummy = DummyEmbedder(1, 64)
    fxv = FXValueEmbedder(dummy, subs)
    fx = FXEmbedder(k3, fxv, 128)
    assert fx.in_dim == 1 + fxv.in_dim

# --- TableEmbedder weight sharing ---

def test_table_embedder_fx_weight_sharing():
    k1, k2, k3, k4 = _split(4)
    ge = GrooveEntityEmbedder(64, k1)
    subs = build_fx_value_embedders(64, k2, ge)
    dummy = DummyEmbedder(1, 64)
    fxv = FXValueEmbedder(dummy, subs)
    fx_emb = FXEmbedder(k3, fxv, 64)
    te = TableEmbedder(64, k4, fx_emb)
    assert te.embedders['fx1'] is te.embedders['fx2']

def test_table_embedder_in_dim_matches_table_width():
    k1, k2, k3, k4 = _split(4)
    ge = GrooveEntityEmbedder(64, k1)
    subs = build_fx_value_embedders(64, k2, ge)
    dummy = DummyEmbedder(1, 64)
    fxv = FXValueEmbedder(dummy, subs)
    fx_emb = FXEmbedder(k3, fxv, 64)
    te = TableEmbedder(64, k4, fx_emb)
    assert te.in_dim == TABLE_WIDTH

# --- SynthWavesEmbedder weight sharing ---

def test_synth_waves_param_weight_sharing():
    sw = SynthWavesEmbedder(KEY)
    assert sw.embedders['start_params'] is sw.embedders['end_params']

def test_synth_waves_embedder_in_dim():
    sw = SynthWavesEmbedder(KEY)
    assert sw.in_dim == SOFTSYNTH_WIDTH + WAVEFRAME_DIM

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

    def test_entity_embedder(self, default_banks):
        inner = GatedNormedEmbedder(16, KEY, in_dim=STEPS_PER_GROOVE * 2)
        e = EntityEmbedder(EntityType.GROOVES, inner)
        out = e(jnp.array([3]), default_banks)
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

    def test_fx_value_embedder(self, default_banks):
        k1, k2 = _split(2)
        ge = GrooveEntityEmbedder(64, k1)
        subs = build_fx_value_embedders(64, k2, ge)
        dummy = DummyEmbedder(1, 64)
        e = FXValueEmbedder(dummy, subs)
        x = jnp.ones((e.in_dim,))
        out = e(x, default_banks)
        assert out.shape == (64,)

    def test_fx_embedder(self, default_banks):
        k1, k2, k3 = _split(3)
        ge = GrooveEntityEmbedder(64, k1)
        subs = build_fx_value_embedders(64, k2, ge)
        dummy = DummyEmbedder(1, 64)
        fxv = FXValueEmbedder(dummy, subs)
        e = FXEmbedder(k3, fxv, 128)
        x = jnp.ones((e.in_dim,))
        out = e(x, default_banks)
        assert out.shape == (128,)

    def test_table_embedder(self, default_banks):
        k1, k2, k3, k4 = _split(4)
        ge = GrooveEntityEmbedder(64, k1)
        subs = build_fx_value_embedders(64, k2, ge)
        dummy = DummyEmbedder(1, 64)
        fxv = FXValueEmbedder(dummy, subs)
        fx_emb = FXEmbedder(k3, fxv, 64)
        e = TableEmbedder(64, k4, fx_emb)
        x = jnp.ones((e.in_dim,))
        out = e(x, default_banks)
        assert out.shape == (64,)

    def test_phrase_fx_embedder(self, default_banks):
        k1, k2, k3, k4, k5 = _split(5)
        ge = GrooveEntityEmbedder(64, k1)
        subs = build_fx_value_embedders(64, k2, ge)

        # Tier 0
        dummy = DummyEmbedder(1, 64)
        fxv0 = FXValueEmbedder(dummy, subs)
        fx0 = FXEmbedder(k3, fxv0, 128)
        te0 = TableEmbedder(64, k4, fx0)

        # Phrase level: table entity
        table_entity = EntityEmbedder(EntityType.TABLES, te0)
        fxv_phrase = FXValueEmbedder(table_entity, subs)
        e = FXEmbedder(k5, fxv_phrase, 128, _projection=fx0.projection)
        x = jnp.ones((e.in_dim,))
        out = e(x, default_banks)
        assert out.shape == (128,)

    # --- Instrument chain ---

    def test_synth_waves_embedder(self):
        e = SynthWavesEmbedder(KEY)
        x = jnp.ones((e.in_dim,))
        out = e(x)
        assert out.shape == (e.out_dim,)

    def test_waveframe_embedder(self):
        e = WaveframeEmbedder(KEY, out_dim=32)
        x = jnp.ones((e.in_dim,))
        out = e(x)
        assert out.shape == (32,)

    # --- Full SongStepEmbedder ---

    def test_song_step_embedder_zero(self, song_step_embedder, default_banks):
        step = jnp.zeros((4, 21))
        out = song_step_embedder(step, default_banks)
        assert out.shape == (4, 256)

    def test_song_step_embedder_nonzero(self, song_step_embedder, default_banks):
        step = jnp.ones((4, 21))
        out = song_step_embedder(step, default_banks)
        assert out.shape == (4, 256)
        assert jnp.linalg.norm(out) > 0

    # --- Null value behavior ---

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

    def test_entity_embedder_soft_mode(self, default_banks):
        """EntityEmbedder soft mode: one-hot probs should match hard-index result."""
        inner = GatedNormedEmbedder(16, KEY, in_dim=STEPS_PER_GROOVE * 2)
        e = EntityEmbedder(EntityType.GROOVES, inner)
        n = default_banks.grooves.shape[0]
        probs = jnp.zeros(n).at[3].set(1.0)
        out_soft = e(probs, default_banks, soft=True)
        out_hard = e(jnp.array([3]), default_banks)
        assert out_soft.shape == (16,)
        assert jnp.allclose(out_soft, out_hard, atol=1e-5)

    def test_entity_embedder_soft_mixture(self, default_banks):
        """Soft mode with uniform weights produces a valid embedding."""
        inner = GatedNormedEmbedder(16, KEY, in_dim=STEPS_PER_GROOVE * 2)
        e = EntityEmbedder(EntityType.GROOVES, inner)
        n = default_banks.grooves.shape[0]
        probs = jnp.ones(n) / n
        out = e(probs, default_banks, soft=True)
        assert out.shape == (16,)

    def test_song_step_embedder_default_banks(self):
        """SongStepEmbedder should produce valid output with default banks."""
        step = jnp.zeros((4, 21))
        emb = SongStepEmbedder(jr.PRNGKey(99))
        out = emb(step, SongBanks.default())
        assert out.shape == (4, 256)

    # --- Real data ---

    def test_song_step_on_real_data(self, song_file):
        k = jr.PRNGKey(33)
        banks = SongBanks.from_songfile(song_file)
        step_emb = SongStepEmbedder(k)
        out = step_emb(song_file.song_tokens[0], banks)
        assert out.shape == (4, 256)
        assert jnp.linalg.norm(out) > 0


# ===================================================================
# HelixEmbedder
# ===================================================================

class TestHelixEmbedder:
    # LSDJ token mapping (token = NOTES_index + 1):
    #   NOTES[0]='---', NOTES[1]='C 3', NOTES[13]='C 4', NOTES[25]='C 5'
    #   NOTES[24]='B 4' → token 25,  NOTES[25]='C 5' → token 26
    #   NOTES[26]='C#5' → token 27,  NOTES[27]='D 5' → token 28
    #   NOTES[13]='C 4' → token 14,  NOTES[15]='D 4' → token 16

    def test_output_shape(self):
        e = HelixEmbedder(32, KEY, period=12, num_values=NUM_NOTES)
        out = e(jnp.array([1], dtype=jnp.uint16))
        assert out.shape == (32,)

    def test_null_is_zero_vector(self):
        """NULL token (0) → zero vector regardless of projection weights."""
        e = HelixEmbedder(32, KEY, period=12, num_values=NUM_NOTES)
        out = e(jnp.array([0], dtype=jnp.uint16))
        assert jnp.all(out == 0.0)

    def test_valid_note_nonzero(self):
        """Any non-null note should produce a non-zero embedding."""
        e = HelixEmbedder(32, KEY, period=12, num_values=NUM_NOTES)
        out = e(jnp.array([14], dtype=jnp.uint16))  # C 4
        assert jnp.linalg.norm(out) > 0

    def test_enharmonic_distance(self):
        """B4→C5 (1 semitone) is closer in helix space than B4→C#5 (2 semitones)."""
        e = HelixEmbedder(32, KEY, period=12, num_values=NUM_NOTES)
        # B4=token 25, C5=token 26, C#5=token 27
        emb_b4  = e(jnp.array([25], dtype=jnp.uint16))
        emb_c5  = e(jnp.array([26], dtype=jnp.uint16))
        emb_cs5 = e(jnp.array([27], dtype=jnp.uint16))
        dist_b4_c5  = jnp.linalg.norm(emb_b4 - emb_c5)
        dist_b4_cs5 = jnp.linalg.norm(emb_b4 - emb_cs5)
        assert dist_b4_c5 < dist_b4_cs5

    def test_same_chroma_different_octave(self):
        """C4 and C5 (same chroma, adjacent octave) are closer than C4 and D4 (2-semitone chroma step)."""
        e = HelixEmbedder(32, KEY, period=12, num_values=NUM_NOTES)
        # C4=token 14, C5=token 26, D4=token 16
        emb_c4 = e(jnp.array([14], dtype=jnp.uint16))
        emb_c5 = e(jnp.array([26], dtype=jnp.uint16))
        emb_d4 = e(jnp.array([16], dtype=jnp.uint16))
        dist_octave = jnp.linalg.norm(emb_c4 - emb_c5)
        dist_chroma = jnp.linalg.norm(emb_c4 - emb_d4)
        assert dist_octave < dist_chroma

    def test_distinct_notes_distinct_outputs(self):
        """Different notes produce different embeddings."""
        e = HelixEmbedder(32, KEY, period=12, num_values=NUM_NOTES)
        out1 = e(jnp.array([14], dtype=jnp.uint16))  # C 4
        out2 = e(jnp.array([26], dtype=jnp.uint16))  # C 5
        assert not jnp.allclose(out1, out2)


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

    def test_shape(self, song_step_embedder, default_banks):
        seq_emb = SequenceEmbedder(song_step_embedder, KEY)
        tokens = jnp.zeros((32, 4, 21))
        out = seq_emb(tokens, default_banks)
        assert out.shape == (32, 4, 256)

    def test_on_real_data(self, song_file):
        k1, k2 = jr.split(jr.PRNGKey(33))
        banks = SongBanks.from_songfile(song_file)
        step_emb = SongStepEmbedder(k1)
        seq_emb = SequenceEmbedder(step_emb, k2)
        tokens = song_file.song_tokens[:32]
        out = seq_emb(tokens, banks)
        assert out.shape == (32, 4, 256)
        assert jnp.linalg.norm(out) > 0

    def test_progress_different_positions(self, song_step_embedder, default_banks):
        """Different absolute positions → different embedding (progress signal active)."""
        seq_emb = SequenceEmbedder(song_step_embedder, KEY)
        tokens = jnp.zeros((8, 4, 21))
        out_early = seq_emb(tokens, default_banks, positions=jnp.arange(8),       song_length=64)
        out_late  = seq_emb(tokens, default_banks, positions=jnp.arange(8) + 32,  song_length=64)
        assert not jnp.allclose(out_early, out_late)

    def test_progress_different_song_length(self, song_step_embedder, default_banks):
        """Same positions but different song_length → different progress fraction → different embedding."""
        seq_emb = SequenceEmbedder(song_step_embedder, KEY)
        tokens = jnp.zeros((8, 4, 21))
        positions = jnp.arange(8)
        out_short = seq_emb(tokens, default_banks, positions=positions, song_length=16)
        out_long  = seq_emb(tokens, default_banks, positions=positions, song_length=256)
        assert not jnp.allclose(out_short, out_long)

    def test_progress_none_song_length_finite(self, song_step_embedder, default_banks):
        """song_length=None falls back to crop length; output should be finite."""
        seq_emb = SequenceEmbedder(song_step_embedder, KEY)
        tokens = jnp.zeros((8, 4, 21))
        out = seq_emb(tokens, default_banks)
        assert jnp.all(jnp.isfinite(out))

    def test_progress_crop_start_shifts_output(self, song_step_embedder, default_banks):
        """Crops at different song offsets (same content) should produce different embeddings."""
        seq_emb = SequenceEmbedder(song_step_embedder, KEY)
        tokens = jnp.zeros((8, 4, 21))
        # simulate two crops from same song at different offsets
        out_start = seq_emb(tokens, default_banks, positions=jnp.arange(8),      song_length=128)
        out_mid   = seq_emb(tokens, default_banks, positions=jnp.arange(8) + 64, song_length=128)
        assert not jnp.allclose(out_start, out_mid)


# ===================================================================
# SongBanks
# ===================================================================

class TestSongBanks:

    def test_default_shapes(self):
        banks = SongBanks.default()
        assert banks.instruments.shape == (NUM_INSTRUMENTS + 1, INSTR_WIDTH)
        assert banks.synth_waves.shape == (NUM_SYNTHS + 1, SOFTSYNTH_WIDTH + WAVEFRAME_DIM)
        assert banks.grooves.shape == (NUM_GROOVES + 1, STEPS_PER_GROOVE * 2)
        assert banks.tables.shape == (NUM_TABLES + 1, TABLE_WIDTH)
        assert banks.traces.shape == (NUM_TABLES + 1, TABLE_WIDTH)

    def test_from_songfile(self, song_file):
        banks = SongBanks.from_songfile(song_file)
        assert banks.instruments.shape == (NUM_INSTRUMENTS + 1, INSTR_WIDTH)
        assert banks.tables.shape == (NUM_TABLES + 1, TABLE_WIDTH)
        assert banks.synth_waves.shape == (NUM_SYNTHS + 1, SOFTSYNTH_WIDTH + WAVEFRAME_DIM)


# ===================================================================
# Runtime banks API
# ===================================================================

class TestRuntimeBanks:
    """Banks are passed at call time — not stored in the model tree."""

    def test_different_banks_produce_different_output(self, song_step_embedder):
        """Calling with different banks should change the embedding output."""
        seq_emb = SequenceEmbedder(song_step_embedder, KEY)
        tokens = jnp.zeros((4, 4, 21))

        banks_zero = SongBanks.default()
        banks_ones = SongBanks(
            instruments=jnp.ones((NUM_INSTRUMENTS + 1, INSTR_WIDTH), dtype=jnp.uint16),
            grooves=jnp.ones((NUM_GROOVES + 1, STEPS_PER_GROOVE * 2), dtype=jnp.uint16),
            tables=jnp.ones((NUM_TABLES + 1, TABLE_WIDTH), dtype=jnp.uint16),
            traces=jnp.ones((NUM_TABLES + 1, TABLE_WIDTH), dtype=jnp.uint16),
            synth_waves=jnp.ones((NUM_SYNTHS + 1, SOFTSYNTH_WIDTH + WAVEFRAME_DIM), dtype=jnp.uint16),
            instrs_occupied=jnp.zeros(NUM_INSTRUMENTS + 1, dtype=jnp.bool_),
            grooves_occupied=jnp.zeros(NUM_GROOVES + 1, dtype=jnp.bool_),
            tables_occupied=jnp.zeros(NUM_TABLES + 1, dtype=jnp.bool_),
            synths_occupied=jnp.zeros(NUM_SYNTHS + 1, dtype=jnp.bool_),
        )

        out_zero = seq_emb(tokens, banks_zero)
        out_ones = seq_emb(tokens, banks_ones)
        assert not jnp.allclose(out_zero, out_ones)

    def test_same_banks_deterministic(self, song_step_embedder, default_banks):
        """Calling twice with the same banks produces identical output."""
        seq_emb = SequenceEmbedder(song_step_embedder, KEY)
        tokens = jnp.zeros((4, 4, 21))
        out1 = seq_emb(tokens, default_banks)
        out2 = seq_emb(tokens, default_banks)
        assert jnp.allclose(out1, out2)

    def test_banks_not_in_model_tree(self, song_step_embedder):
        """No uint16 arrays (bank data) should appear in the model parameter tree."""
        import jax
        for leaf in jax.tree.leaves(song_step_embedder):
            if hasattr(leaf, 'dtype'):
                assert leaf.dtype != jnp.uint16, (
                    f"uint16 leaf found in model tree (likely a leaked bank): {leaf.shape}"
                )

    def test_on_real_data(self, song_file):
        k1, k2 = jr.split(jr.PRNGKey(42))
        banks = SongBanks.from_songfile(song_file)
        step_emb = SongStepEmbedder(k1)
        seq_emb = SequenceEmbedder(step_emb, k2)
        tokens = song_file.song_tokens[:8]
        out = seq_emb(tokens, banks)
        assert out.shape == (8, 4, step_emb.per_ch_dim)
        assert jnp.linalg.norm(out) > 0
